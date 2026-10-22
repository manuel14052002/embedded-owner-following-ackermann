"""CSV logging of per-frame tracker output."""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Optional, Union

from owner_tracking.datatypes import FrameResult
from owner_tracking.geometry import iou

LOGGER = logging.getLogger(__name__)

#: Columns required by the specification, followed by diagnostic extras.
CSV_COLUMNS = [
    "timestamp", "frame_number", "owner_bbox", "detection_confidence", "appearance_similarity",
    "IoU", "motion_score", "association_score", "tracking_state", "identity_switch_attempt",
    # extras
    "owner_id", "decision", "reported", "score_margin", "second_best_score",
    "overall_confidence", "appearance_conf", "motion_conf", "spatial_conf", "detection_conf",
    "identity_conf", "num_detections", "gallery_size", "reidentified", "processing_ms",
    "top_candidate_rejected", "rejections",
]


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.{digits}f}"


def result_to_row(result: FrameResult) -> dict[str, str]:
    """Flatten a :class:`FrameResult` into CSV cells."""
    best = result.best_candidate if result.reported else None
    c = result.confidence
    bbox = result.owner_bbox
    iou_val: Optional[float] = None
    if bbox is not None and result.predicted_bbox is not None:
        iou_val = iou(bbox, result.predicted_bbox)
    return {
        "timestamp": f"{result.timestamp:.6f}",
        "frame_number": str(result.frame_index),
        "owner_bbox": "" if bbox is None else " ".join(f"{v:.1f}" for v in bbox),
        "detection_confidence": _num(best.detection_confidence if best else None),
        "appearance_similarity": _num(result.best_candidate.appearance_similarity
                                      if result.best_candidate else None),
        "IoU": _num(iou_val),
        "motion_score": _num(best.motion_score if best else None),
        "association_score": _num(result.best_candidate.total if result.best_candidate else None),
        "tracking_state": result.state.value,
        "identity_switch_attempt": str(int(result.identity_switch_attempt)),
        "owner_id": result.owner_id or "",
        "decision": result.decision.value,
        "reported": str(int(result.reported)),
        "score_margin": _num(result.score_margin),
        "second_best_score": _num(result.second_best_score),
        "overall_confidence": _num(c.overall), "appearance_conf": _num(c.appearance),
        "motion_conf": _num(c.motion), "spatial_conf": _num(c.spatial),
        "detection_conf": _num(c.detection), "identity_conf": _num(c.identity),
        "num_detections": str(len(result.detections)),
        "gallery_size": str(result.gallery_size),
        "reidentified": str(int(result.reidentified)),
        "processing_ms": _num(result.processing_ms, 3),
        # Why candidates were refused this frame (diagnoses misses / LOST events).
        "top_candidate_rejected": _top_rejection(result),
        "rejections": ";".join(f"{k}:{v}" for k, v in sorted(Counter(
            c.rejected_reason for c in result.candidates if c.rejected_reason).items())),
    }


def _top_rejection(result: FrameResult) -> str:
    if not result.candidates:
        return ""
    top = max(result.candidates, key=lambda c: c.total)
    return top.rejected_reason or ""


class CsvFrameLogger:
    """Buffered CSV writer. Use as a context manager or call :meth:`close`."""

    def __init__(self, path: Union[str, Path], interval: int = 1, flush_every: int = 30) -> None:
        self.path = Path(path)
        self.interval = max(1, int(interval))
        self.flush_every = max(1, int(flush_every))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._since_flush = 0
        LOGGER.info("Logging tracker output to %s", self.path)

    def log(self, result: FrameResult) -> None:
        # Always log switch attempts and state changes, regardless of interval.
        if result.frame_index % self.interval and not result.identity_switch_attempt:
            return
        self._writer.writerow(result_to_row(result))
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._fh.flush()
            self._since_flush = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "CsvFrameLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
