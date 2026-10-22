"""Core data types shared across the owner-tracking pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TrackState(str, Enum):
    """Externally visible state of the owner track."""

    UNINITIALIZED = "UNINITIALIZED"
    TRACKING = "TRACKING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    REIDENTIFYING = "REIDENTIFYING"
    LOST = "LOST"


#: When two sources of evidence propose different states, the more severe
#: (less trusting) state wins.
STATE_SEVERITY: dict[TrackState, int] = {
    TrackState.UNINITIALIZED: -1,
    TrackState.TRACKING: 0,
    TrackState.LOW_CONFIDENCE: 1,
    TrackState.AMBIGUOUS: 2,
    TrackState.REIDENTIFYING: 3,
    TrackState.LOST: 4,
}


class Decision(str, Enum):
    """Per-frame association decision."""

    NO_OWNER = "NO_OWNER"      # no owner selected yet
    ACCEPT = "ACCEPT"          # a detection was confirmed as the owner
    PENDING = "PENDING"        # a candidate is collecting temporal evidence
    AMBIGUOUS = "AMBIGUOUS"    # best and second-best too close: choose neither
    MISS = "MISS"              # no candidate met the identity threshold


@dataclass
class Detection:
    """A person detection.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` pixels, float32.
        confidence: detector confidence in ``[0, 1]``.
        source_id: ground-truth identity for simulation/evaluation only.
            The tracker never reads it.
    """

    bbox: np.ndarray
    confidence: float
    source_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.confidence = float(self.confidence)


@dataclass
class CandidateScore:
    """Association evidence for one detection in one frame.

    ``*_score`` fields are normalised to ``[0, 1]``; ``nan`` marks an
    unavailable component, which is excluded from the weighted sum.
    """

    index: int
    bbox: np.ndarray
    detection_confidence: float
    appearance_similarity: float      # raw cosine similarity to the owner model
    distractor_similarity: float      # raw cosine similarity to known other people
    appearance_score: float
    iou_score: float
    position_score: float
    scale_score: float
    motion_score: float
    velocity_z: float                 # normalised velocity disagreement (nan = unknown)
    mahalanobis_d2: float
    total: float = 0.0
    rejected_reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.rejected_reason is None


@dataclass
class ConfidenceReport:
    """Separate confidence channels plus the fused, smoothed overall value."""

    appearance: float = 0.0
    motion: float = 0.0
    spatial: float = 0.0
    detection: float = 0.0
    identity: float = 0.0
    overall: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FrameResult:
    """Everything the tracker concluded for one frame."""

    frame_index: int
    timestamp: float
    state: TrackState
    decision: Decision
    owner_id: Optional[str]
    owner_bbox: Optional[np.ndarray]          # set only when output is trusted
    predicted_bbox: Optional[np.ndarray]
    owner_detection_index: Optional[int]
    confidence: ConfidenceReport
    best_candidate: Optional[CandidateScore]
    second_best_score: float
    score_margin: float
    identity_switch_attempt: bool
    reidentified: bool
    candidates: list[CandidateScore] = field(default_factory=list)
    detections: list = field(default_factory=list)
    gallery_size: int = 0
    processing_ms: float = 0.0

    @property
    def reported(self) -> bool:
        """True if the tracker asserts ``owner_bbox`` is the owner."""
        return self.owner_bbox is not None
