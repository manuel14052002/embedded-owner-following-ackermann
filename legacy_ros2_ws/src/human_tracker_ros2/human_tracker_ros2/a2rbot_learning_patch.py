"""
a2rbot_learning_patch.py  (v2)
==============================
Opt-in learning extension for multicue_owner.MultiCueMatcher.

Nothing in this module runs unless the matcher is constructed with
learning_mode="extended" and/or leg_mode="shadow"/"on" (or the env vars
A2RBOT_LEARNING / A2RBOT_LEG_CUE are set). Defaults reproduce the legacy
behaviour exactly — verified by verify_learning_patch.py.

  EnrollmentPhase    >= 120 s enrollment, quality-weighted anchors
  ContinuousLearner  >= 300 s ACTIVE learning while following, then slow drift
  compute_leg_profile / leg_similarity
                     depth-based lower-body (leg) signature from the D435

Evidence tags: [OBS] in source  [DER] arithmetic  [INF] design choice
               [UNK] must be measured before it is trusted
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("a2rbot.learning")

# ── Enrollment ──────────────────────────────────────────────────────────────
ENROLL_MIN_DURATION    = 120.0   # s  — requirement
ENROLL_MIN_GOOD_FRAMES = 600     # [DER] ~40 s of accepted frames at 15 Hz
ENROLL_MAX_FRAMES      = 4000    # memory cap: 4000 x 512 x 4 B = 8.2 MB [DER]
ENROLL_QUALITY_FLOOR   = 0.40    # [INF]
ENROLL_DEPTH_MIN_M     = 0.40    # [INF]
ENROLL_DEPTH_MAX_M     = 4.00    # [INF]
ENROLL_GALLERY_SEED    = 15      # matches legacy deque(maxlen=15) [OBS]

# ── Continuous learning ────────────────────────────────────────────────────
LEARN_ACTIVE_DURATION  = 300.0   # s of VERIFIED following — requirement
LEARN_ACTIVE_RATE_HZ   = 0.2     # [DER] 300 s x 0.2 Hz = 60 samples
LEARN_DRIFT_RATE_HZ    = 0.05    # after active phase: 1 sample / 20 s
LEARN_GALLERY_MAXLEN   = 60      # [DER] holds the whole active phase
LEARN_SCORE_THR        = 0.82    # [OBS] unchanged legacy gate
LEARN_MAX_GAP_S        = 1.0     # gaps longer than this don't count as following

# ── Leg cue ────────────────────────────────────────────────────────────────
LEG_BAND_TOP     = 0.55          # fraction of bbox height  [INF]
LEG_BAND_BOTTOM  = 0.95
LEG_BINS         = 8
LEG_FG_TOL_MM    = 350.0         # pixel belongs to person if within ±0.35 m [INF]
LEG_MIN_FG_PX    = 150           # [INF]
LEG_TAU          = 0.15          # [UNK] UNCALIBRATED — calibrate from shadow logs
LEG_CONF_CAP     = 0.45          # < 0.5 → never counts toward quorum until calibrated
LEG_EMA_KEEP     = 0.90          # slow adaptation of the leg anchor [INF]


# ===========================================================================
class EnrollmentPhase:
    """Time-gated (>=120 s), quality-weighted owner enrollment."""

    _W_REID, _W_DEPTH, _W_BBOX = 0.50, 0.30, 0.20

    def __init__(self):
        self._start_ts: Optional[float] = None
        self._embs, self._hists, self._bbox_h, self._depth, self._q = [], [], [], [], []
        self._frame_h = 480
        self._emb_sum: Optional[np.ndarray] = None

    def start(self) -> None:
        self.__init__()
        self._start_ts = time.monotonic()
        logger.info("[ENROLL] started, %.0f s required", ENROLL_MIN_DURATION)

    @property
    def started(self) -> bool:
        return self._start_ts is not None

    @property
    def elapsed(self) -> float:
        return 0.0 if self._start_ts is None else time.monotonic() - self._start_ts

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed / ENROLL_MIN_DURATION)

    @property
    def n_accepted(self) -> int:
        return len(self._q)

    @property
    def ready(self) -> bool:
        return (self.elapsed >= ENROLL_MIN_DURATION
                and self.n_accepted >= ENROLL_MIN_GOOD_FRAMES)

    def feed(self, emb, hist, bbox, depth_m, frame_h) -> Tuple[bool, float]:
        if self._start_ts is None:          # robust: auto-start on first frame
            self.start()
        if emb is None or hist is None or self.n_accepted >= ENROLL_MAX_FRAMES:
            return False, 0.0
        if depth_m is not None and not (ENROLL_DEPTH_MIN_M <= depth_m <= ENROLL_DEPTH_MAX_M):
            return False, 0.0
        self._frame_h = frame_h
        emb = emb.ravel().astype(np.float32)
        bbox_h = float(bbox[3] - bbox[1])

        # quality: agreement with the RUNNING MEAN of accepted embeddings
        if self._emb_sum is None:
            q_reid = 1.0
        else:
            m = self._emb_sum / (np.linalg.norm(self._emb_sum) + 1e-6)
            q_reid = float(np.clip(0.5 * (float(emb @ m) / (np.linalg.norm(emb) + 1e-6) + 1.0), 0, 1))
        if depth_m is None or not self._depth:
            q_depth = 0.5
        else:
            ref = float(np.median(self._depth[-60:]))
            q_depth = float(np.clip(1.0 - abs(depth_m - ref) / max(ref, 0.01), 0, 1))
        q_bbox = float(np.clip(bbox_h / (0.4 * frame_h), 0.3, 1.0))
        q = self._W_REID * q_reid + self._W_DEPTH * q_depth + self._W_BBOX * q_bbox
        if q < ENROLL_QUALITY_FLOOR:
            return False, q

        self._emb_sum = emb.copy() if self._emb_sum is None else self._emb_sum + emb
        self._embs.append(emb)
        self._hists.append(hist.ravel().astype(np.float32))
        self._bbox_h.append(bbox_h)
        if depth_m is not None:
            self._depth.append(float(depth_m))
        self._q.append(q)
        return True, q

    def finalize(self) -> Optional[dict]:
        if not self.ready:
            return None
        w = np.asarray(self._q, dtype=np.float64); w /= w.sum()
        E = np.stack(self._embs); H = np.stack(self._hists)
        anchor_emb = (w[:, None] * E).sum(0)
        anchor_emb /= (np.linalg.norm(anchor_emb) + 1e-6)
        anchor_hist = (w[:, None] * H).sum(0).astype(np.float32)
        bbox_h = float(np.average(self._bbox_h, weights=w))
        depth = float(np.median(self._depth)) if self._depth else None
        top = np.argsort(self._q)[-ENROLL_GALLERY_SEED:]          # best frames
        return dict(
            anchor_emb=anchor_emb.astype(np.float32), anchor_hist=anchor_hist,
            bbox_h_px=bbox_h, height_ratio=bbox_h / self._frame_h,
            depth_m=depth, seed_embs=[E[i] for i in top], seed_hists=[H[i] for i in top],
            n_frames=self.n_accepted, duration_s=self.elapsed,
        )


# ===========================================================================
class ContinuousLearner:
    """
    Learning during follow mode.

    Phase ACTIVE : first LEARN_ACTIVE_DURATION s of *verified following*
                   (time only accumulates while the owner is confirmed;
                   gaps > 1 s are not counted). Gallery updates at 0.2 Hz.
    Phase DRIFT  : afterwards, learning continues at 0.05 Hz to track
                   slow appearance change (jacket on/off, lighting).

    The fixed enrollment anchor is never modified — only the galleries,
    which enter ReID/colour scoring at 0.2 weight [OBS], so the anchor
    stays dominant and identity cannot drift onto another person.
    """

    def __init__(self):
        self.learned_s = 0.0
        self.n_updates = 0
        self._last_seen: Optional[float] = None
        self._last_update = -1e9

    @property
    def phase(self) -> str:
        return "active" if self.learned_s < LEARN_ACTIVE_DURATION else "drift"

    @property
    def progress(self) -> float:
        return min(1.0, self.learned_s / LEARN_ACTIVE_DURATION)

    def observe(self) -> None:
        """Call each frame the owner is verified (from update_galleries)."""
        now = time.monotonic()
        if self._last_seen is not None:
            gap = now - self._last_seen
            if gap <= LEARN_MAX_GAP_S:
                self.learned_s += gap
        self._last_seen = now

    def should_update(self, score: float) -> bool:
        if score < LEARN_SCORE_THR:
            return False
        rate = LEARN_ACTIVE_RATE_HZ if self.phase == "active" else LEARN_DRIFT_RATE_HZ
        now = time.monotonic()
        if now - self._last_update < 1.0 / rate:
            return False
        self._last_update = now
        self.n_updates += 1
        return True

    def status(self) -> dict:
        return dict(phase=self.phase, learned_s=round(self.learned_s, 1),
                    progress=round(self.progress, 3), n_updates=self.n_updates)


# ===========================================================================
def compute_leg_profile(depth_frame, bbox, distance_m=None) -> Tuple[Optional[np.ndarray], int]:
    """
    Lower-body occupancy signature from the aligned D435 depth (uint16 mm).

    The leg band (55-95 % of bbox height) is segmented by DEPTH: a pixel
    belongs to the person if it lies within ±0.35 m of the person's depth.
    The feature is the per-column fraction of person pixels (8 bins).

    Why occupancy and not raw depth: raw depth encodes WHERE the person is,
    so the owner at 3 m would fail against a 1.5 m enrollment. Occupancy is
    a shape fraction — the bbox scales with distance, so the feature is
    approximately distance-invariant. [INF]

    Returns (profile, n_person_px) or (None, n) on abstention.
    """
    if depth_frame is None:
        return None, 0
    H, W = depth_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # legs cut off by the bottom of the image -> no reliable signature
    if y2 >= H - 2:
        return None, 0
    x1, x2 = max(0, x1), min(W, x2)
    bh = y2 - y1
    if bh < 40 or (x2 - x1) < LEG_BINS * 2:
        return None, 0
    ly1 = max(0, y1 + int(LEG_BAND_TOP * bh))
    ly2 = min(H, y1 + int(LEG_BAND_BOTTOM * bh))
    roi = depth_frame[ly1:ly2, x1:x2].astype(np.float32)
    valid = (roi > 200) & (roi < 8000)                       # [OBS] same window as _get_3d
    if valid.sum() < LEG_MIN_FG_PX:
        return None, int(valid.sum())
    z_mm = distance_m * 1000.0 if distance_m else float(np.median(roi[valid]))
    fg = valid & (np.abs(roi - z_mm) < LEG_FG_TOL_MM)
    n_fg = int(fg.sum())
    if n_fg < LEG_MIN_FG_PX:
        return None, n_fg
    cols = np.array_split(fg, LEG_BINS, axis=1)
    prof = np.array([c.mean() for c in cols], dtype=np.float32)
    return prof, n_fg


def leg_similarity(profile, anchor, n_fg) -> Tuple[float, float]:
    """(sim, conf). sim = exp(-mean|p-a| / tau). tau is UNCALIBRATED [UNK]."""
    if profile is None or anchor is None:
        return 0.0, 0.0
    d = float(np.mean(np.abs(profile - anchor)))
    sim = float(np.exp(-d / LEG_TAU))
    conf = float(np.clip(n_fg / 2000.0, 0.10, LEG_CONF_CAP))
    return sim, conf
