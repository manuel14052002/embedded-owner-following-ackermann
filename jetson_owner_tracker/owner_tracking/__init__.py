"""Identity-preserving single-owner person tracking.

Pipeline of interchangeable components::

    PersonDetector -> PersonReID -> MotionPredictor -> OwnerTracker
                   -> IdentityManager -> ConfidenceManager -> Visualizer

Central design rule: *it is better to temporarily lose the owner than to
confidently track the wrong person.*
"""

from owner_tracking.config import TrackerConfig
from owner_tracking.datatypes import Decision, Detection, FrameResult, TrackState
from owner_tracking.tracker import OwnerTracker

__all__ = ["Decision", "Detection", "FrameResult", "OwnerTracker", "TrackState", "TrackerConfig"]
__version__ = "1.0.0"
