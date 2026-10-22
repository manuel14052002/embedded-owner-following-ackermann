"""Naive baselines used to quantify what the anti-switch machinery buys.

:class:`NearestNeighborBaseline` follows the detection closest to the last
owner box (IoU first, then centre distance) with no appearance model,
margin rule or confirmation - the "proximity over identity" behaviour the
specification forbids. It shares ``select_owner``/``update``/``has_owner`` with
:class:`~owner_tracking.tracker.OwnerTracker` so ``run_scenario`` works on both.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from owner_tracking.datatypes import ConfidenceReport, Decision, Detection, FrameResult, TrackState
from owner_tracking.geometry import bbox_center, iou
from owner_tracking.tracker import find_clicked_detection


class NearestNeighborBaseline:
    """Proximity-only tracker (reference point, not for deployment)."""

    def __init__(self, max_jump_heights: float = 1.0) -> None:
        self.max_jump_heights = max_jump_heights
        self.owner_id: Optional[str] = None
        self._last: Optional[np.ndarray] = None
        self.frame_index = -1

    @property
    def has_owner(self) -> bool:
        return self.owner_id is not None

    def reset(self) -> None:
        self.owner_id, self._last = None, None

    def select_owner(self, frame: np.ndarray, click: tuple[float, float],
                     detections: Sequence[Detection]) -> tuple[bool, str]:
        index, reason = find_clicked_detection(list(detections), click, 0.0)
        if index is None:
            return False, reason
        self.owner_id, self._last = "001", detections[index].bbox.copy()
        return True, "ok"

    def update(self, frame: np.ndarray, detections: Sequence[Detection],
               timestamp: Optional[float] = None) -> FrameResult:
        start = time.perf_counter()
        self.frame_index += 1
        dets = list(detections)
        chosen: Optional[int] = None
        if self._last is not None and dets:
            h = max(float(self._last[3] - self._last[1]), 1.0)
            cx, cy = bbox_center(self._last)

            def key(i: int) -> tuple[float, float]:
                c = bbox_center(dets[i].bbox)
                return (-iou(dets[i].bbox, self._last), float(np.hypot(c[0] - cx, c[1] - cy)))

            best = min(range(len(dets)), key=key)
            c = bbox_center(dets[best].bbox)
            if np.hypot(c[0] - cx, c[1] - cy) <= self.max_jump_heights * h:
                chosen = best
                self._last = dets[best].bbox.copy()
        state = (TrackState.UNINITIALIZED if self.owner_id is None
                 else TrackState.TRACKING if chosen is not None else TrackState.LOST)
        return FrameResult(
            frame_index=self.frame_index, timestamp=time.time() if timestamp is None else timestamp,
            state=state, decision=Decision.ACCEPT if chosen is not None else Decision.MISS,
            owner_id=self.owner_id, owner_bbox=None if chosen is None else dets[chosen].bbox.copy(),
            predicted_bbox=None, owner_detection_index=chosen, confidence=ConfidenceReport(),
            best_candidate=None, second_best_score=0.0, score_margin=0.0,
            identity_switch_attempt=False, reidentified=False, detections=dets,
            processing_ms=(time.perf_counter() - start) * 1000.0)
