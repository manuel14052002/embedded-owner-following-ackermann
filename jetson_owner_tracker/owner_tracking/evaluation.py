"""Owner-centric evaluation metrics.

The task is *single-target* tracking of a pre-designated identity, so metrics
are computed for the owner only. Each frame provides:

* ``gt``: ``{person_id: bbox}`` for every annotated visible person, and
* ``pred``: the tracker's asserted owner box, or ``None``.

A prediction is matched to the ground-truth person it overlaps most with
``IoU >= iou_threshold`` (``None`` if it overlaps nobody).

Definitions (per frame):

* **TP** - prediction matches the owner.
* **FP** - prediction present but does not match the owner. If it matches
  another annotated person it is also a **false owner frame**.
* **FN** - owner visible but not correctly predicted (a wrong prediction is
  both FP and FN, as in CLEAR-MOT).
* **ID switch** - the matched identity of consecutive (non-empty) predictions
  changes. **Incorrect owner switch** - the tracker starts reporting a person
  other than the owner (count of wrong-person segments). This is the
  headline metric: it must be zero.
* **MOTA** = 1 - (FN + FP + IDSW) / #owner-visible-frames.
* **IDF1** = 2 IDTP / (2 IDTP + IDFP + IDFN). The tracker's single identity is
  bound to the owner by the operator's click, so IDTP = TP.
* **HOTA** - with one GT and one predicted identity, association accuracy
  equals detection accuracy (AssA = TP/(TP+FN+FP) = DetA), so
  HOTA_a = sqrt(DetA_a * AssA_a) = DetA_a, averaged over the 19 localisation
  thresholds a = 0.05 ... 0.95 as in Luiten et al. (2021).
* **Loss episodes** - maximal runs of frames without a TP (after the first
  TP) during which the owner is visible at least once; invisible frames inside
  the run are included, so leaving and returning is one episode. Duration is
  the run length. **Re-ID success rate** - fraction of episodes of at least
  ``min_reid_gap`` frames that end with a correct re-acquisition, with no
  wrong-person report inside the episode. Episodes cut off by the end of the
  sequence count as failures (conservative).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from owner_tracking.geometry import iou

GT = dict[int, np.ndarray]


@dataclass
class OwnerMetrics:
    frames: int = 0
    owner_visible_frames: int = 0
    predicted_frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    false_owner_frames: int = 0
    id_switches: int = 0
    incorrect_owner_switches: int = 0
    precision: float = 0.0
    recall: float = 0.0
    mota: float = 0.0
    idf1: float = 0.0
    hota: float = 0.0
    loss_episodes: int = 0
    mean_loss_frames: float = 0.0
    max_loss_frames: int = 0
    reid_opportunities: int = 0
    reid_successes: int = 0
    reid_success_rate: float = float("nan")
    mean_reacquisition_frames: float = float("nan")
    fps: float = float("nan")
    mean_latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("extras")
        d.update(self.extras)
        return d


def _match(pred: Optional[np.ndarray], gt: GT, thr: float) -> tuple[Optional[int], float]:
    """Return ``(person_id, iou)`` of the GT box best matching ``pred``."""
    if pred is None or not gt:
        return None, 0.0
    best_id, best = None, 0.0
    for pid, box in gt.items():
        v = iou(pred, box)
        if v > best:
            best_id, best = pid, v
    return (best_id, best) if best >= thr else (None, best)


def evaluate_owner_track(gt_frames: Sequence[GT], predictions: Sequence[Optional[np.ndarray]],
                         owner_id: int, iou_threshold: float = 0.5, min_reid_gap: int = 5,
                         latencies_ms: Optional[Sequence[float]] = None) -> OwnerMetrics:
    """Compute owner-centric metrics. ``gt_frames`` and ``predictions`` are aligned per frame."""
    if len(gt_frames) != len(predictions):
        raise ValueError("gt_frames and predictions must have the same length")
    m = OwnerMetrics(frames=len(gt_frames))
    status: list[str] = []          # per frame: "tp", "fn", "none" (owner invisible)
    prev_identity: Optional[int] = None
    prev_wrong = False
    owner_ious: list[float] = []    # IoU with owner for every owner-visible frame
    wrong_flags: list[bool] = []
    for gt, pred in zip(gt_frames, predictions):
        visible = owner_id in gt
        m.owner_visible_frames += visible
        matched, _ = _match(pred, gt, iou_threshold)
        if pred is not None:
            m.predicted_frames += 1
        owner_iou = iou(pred, gt[owner_id]) if (visible and pred is not None) else 0.0
        if visible:
            owner_ious.append(owner_iou)
        is_tp = matched == owner_id and pred is not None
        if pred is not None and not is_tp:
            m.fp += 1
            if matched is not None:
                m.false_owner_frames += 1
        if is_tp:
            m.tp += 1
        elif visible:
            m.fn += 1
        wrong = pred is not None and matched is not None and matched != owner_id
        if wrong and not prev_wrong:
            m.incorrect_owner_switches += 1
        prev_wrong = wrong
        wrong_flags.append(wrong)
        if pred is not None and matched is not None:
            if prev_identity is not None and matched != prev_identity:
                m.id_switches += 1
            prev_identity = matched
        status.append("tp" if is_tp else ("fn" if visible else "none"))

    m.precision = m.tp / (m.tp + m.fp) if (m.tp + m.fp) else 0.0
    m.recall = m.tp / m.owner_visible_frames if m.owner_visible_frames else 0.0
    m.mota = (1.0 - (m.fn + m.fp + m.id_switches) / m.owner_visible_frames
              if m.owner_visible_frames else 0.0)
    idtp, idfp, idfn = m.tp, m.predicted_frames - m.tp, m.owner_visible_frames - m.tp
    m.idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (idtp + idfp + idfn) else 0.0
    # HOTA (single target): mean DetA over localisation thresholds.
    ious = np.asarray(owner_ious)
    alphas = np.arange(0.05, 0.96, 0.05)
    hotas = []
    for a in alphas:
        tp_a = int(np.sum(ious >= a))
        fn_a = m.owner_visible_frames - tp_a
        fp_a = m.predicted_frames - tp_a
        denom = tp_a + fn_a + fp_a
        hotas.append(tp_a / denom if denom else 0.0)
    m.hota = float(np.mean(hotas))

    # Loss episodes: maximal runs without a TP, starting after the first TP.
    # A run spans frames where the owner is invisible, so an exit and return is
    # one episode. Only runs in which the owner is visible at least once count.
    episodes: list[tuple[int, bool]] = []   # (length_in_frames, recovered)
    reacq: list[int] = []
    n = len(status)
    i = next((k for k, s in enumerate(status) if s == "tp"), n)
    while i < n:
        if status[i] == "tp":
            i += 1
            continue
        start, first_visible, wrong_first = i, None, False
        while i < n and status[i] != "tp":
            if status[i] == "fn" and first_visible is None:
                first_visible = i
            wrong_first = wrong_first or wrong_flags[i]
            i += 1
        if first_visible is None:
            continue            # owner never visible: nothing to re-identify (sequence ended)
        recovered = i < n and not wrong_first
        episodes.append((i - start, recovered))
        if recovered:
            reacq.append(i - first_visible)
    if episodes:
        lengths = [e[0] for e in episodes]
        m.loss_episodes = len(episodes)
        m.mean_loss_frames = float(np.mean(lengths))
        m.max_loss_frames = int(max(lengths))
    long_eps = [e for e in episodes if e[0] >= min_reid_gap]
    m.reid_opportunities = len(long_eps)
    m.reid_successes = sum(e[1] for e in long_eps)
    if long_eps:
        m.reid_success_rate = m.reid_successes / len(long_eps)
    if reacq:
        m.mean_reacquisition_frames = float(np.mean(reacq))
    if latencies_ms is not None and len(latencies_ms):
        lat = np.asarray(latencies_ms, dtype=float)
        m.mean_latency_ms = float(lat.mean())
        m.p95_latency_ms = float(np.percentile(lat, 95))
        m.fps = 1000.0 / m.mean_latency_ms if m.mean_latency_ms > 0 else float("nan")
    return m


# ----------------------------------------------------------------- loaders
def load_ground_truth(path: Union[str, Path], num_frames: Optional[int] = None) -> list[GT]:
    """Load ``frame,person_id,x1,y1,x2,y2`` CSV (header optional, frames 0-based)."""
    rows: dict[int, GT] = {}
    max_frame = -1
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.reader(fh):
            if not rec or rec[0].strip().lower() in ("frame", "frame_number", "#"):
                continue
            f, pid = int(float(rec[0])), int(float(rec[1]))
            box = np.array([float(v) for v in rec[2:6]], dtype=np.float32)
            rows.setdefault(f, {})[pid] = box
            max_frame = max(max_frame, f)
    n = num_frames if num_frames is not None else max_frame + 1
    return [rows.get(f, {}) for f in range(n)]


def load_tracker_log(path: Union[str, Path]) -> tuple[list[Optional[np.ndarray]], list[float]]:
    """Load a :class:`~owner_tracking.logging_utils.CsvFrameLogger` file.

    Returns per-frame predictions (indexed by ``frame_number``) and latencies.
    The log must have been written with ``interval: 1``.
    """
    preds: dict[int, Optional[np.ndarray]] = {}
    lat: list[float] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame_number"])
            box = row.get("owner_bbox", "").strip()
            preds[f] = np.array([float(v) for v in box.split()], dtype=np.float32) if box else None
            if row.get("processing_ms"):
                lat.append(float(row["processing_ms"]))
    n = max(preds) + 1 if preds else 0
    return [preds.get(f) for f in range(n)], lat


def evaluate_scenario_results(results: Sequence, owner_id: int, **kwargs) -> OwnerMetrics:
    """Evaluate the ``(ScenarioFrame, FrameResult)`` list returned by ``run_scenario``."""
    gt = [sf.ground_truth for sf, _ in results]
    preds = [r.owner_bbox for _, r in results]
    lat = [r.processing_ms for _, r in results]
    m = evaluate_owner_track(gt, preds, owner_id, latencies_ms=lat, **kwargs)
    m.extras["blocked_switch_attempts"] = sum(r.identity_switch_attempt for _, r in results)
    return m
