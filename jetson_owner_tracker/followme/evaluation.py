"""Metrics and evaluation protocols of the paper's experiments (Sect. VI).

* Table I: per-class precision / recall / F1 and accuracy from a confusion
  matrix (rows = true label, columns = predicted label).
* Sect. VI-B Re-ID protocol: each subject is calibrated in turn (2/3 of its
  calibration images for mu/sigma, 1/3 for lambda_d); the others are
  distractors. Test set 1 (everyone present): correct iff the target is
  selected. Test set 2 (everyone but the target): correct iff nobody is
  selected. Confusion matrices are averaged over subjects (Fig. 6b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from followme.calibration import TargetModel, fit_target_model, identify
from followme.config import CalibrationConfig


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_report(cm: np.ndarray, labels: Sequence[str]) -> dict:
    """Precision, recall, F1 per class and overall accuracy (Table I)."""
    out: dict = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1}
    out["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return out


def format_report(cm: np.ndarray, labels: Sequence[str], title: str) -> str:
    rep = classification_report(cm, labels)
    w = max(10, max(len(l) for l in labels) + 2)
    lines = [title, "confusion (rows = true, cols = predicted):",
             " " * w + "".join(f"{l:>{w}}" for l in labels)]
    for i, l in enumerate(labels):
        lines.append(f"{l:<{w}}" + "".join(f"{v:>{w}.1f}" for v in cm[i]))
    lines.append("")
    lines.append(" " * w + "".join(f"{l:>{w}}" for l in labels))
    for m in ("precision", "recall", "f1"):
        lines.append(f"{m:<{w}}" + "".join(f"{rep[l][m]:>{w}.2f}" for l in labels))
    lines.append(f"{'accuracy':<{w}}{rep['accuracy']:>{w}.2f}")
    return "\n".join(lines)


@dataclass
class ImageInstances:
    """Pre-extracted per-image data: one feature row and box per segmented person."""

    features: np.ndarray          # (N, D)
    boxes: np.ndarray             # (N, 4)
    target_box: Optional[np.ndarray] = None   # GT box of the evaluated subject (test set 1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ReIDSubjectResult:
    subject: str
    model: TargetModel
    cm: np.ndarray                                    # 2x2: [Target, No Target]
    wrong_person: int = 0                             # set 1: someone else selected
    target_distances: list = field(default_factory=list)       # threshold-set d (Fig. 2 green)
    distractor_distances: list = field(default_factory=list)   # other people's d (Fig. 2 red)


REID_LABELS = ("Target", "No Target")


def evaluate_reid_subject(subject: str, calib_features: np.ndarray, present: Sequence[ImageInstances],
                          absent: Sequence[ImageInstances], cfg: CalibrationConfig,
                          iou_match: float = 0.5, seed: int = 0) -> ReIDSubjectResult:
    """Sect. VI-B protocol for one calibrated subject."""
    rng = np.random.default_rng(seed)
    model = fit_target_model(calib_features, cfg, rng)
    y_true, y_pred = [], []
    wrong = 0
    distractor_d: list[float] = []
    for img in present:
        y_true.append("Target")
        if len(img.features) == 0:
            y_pred.append("No Target")
            continue
        ident = identify(model, img.features)
        hit = (ident.target_index is not None and img.target_box is not None
               and _iou(img.boxes[ident.target_index], img.target_box) >= iou_match)
        if ident.target_index is not None and not hit:
            wrong += 1
        y_pred.append("Target" if hit else "No Target")
        if img.target_box is not None:
            for i, b in enumerate(img.boxes):
                if _iou(b, img.target_box) < iou_match:
                    distractor_d.append(float(ident.distances[i]))
    for img in absent:
        y_true.append("No Target")
        ident = identify(model, img.features) if len(img.features) else None
        y_pred.append("No Target" if ident is None or ident.target_index is None else "Target")
        if ident is not None:
            distractor_d.extend(float(d) for d in ident.distances)
    # Fig. 2 green curve: distances of held-out calibration samples of the subject
    order = np.random.default_rng(seed).permutation(len(calib_features))
    thresh_idx = order[model.n_calibration:]
    target_d = model.distances(calib_features[thresh_idx]).tolist()
    return ReIDSubjectResult(subject, model, confusion_matrix(y_true, y_pred, REID_LABELS), wrong,
                             target_d, distractor_d)
