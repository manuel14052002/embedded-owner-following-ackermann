"""FollowMe: person following with visual Re-ID and hand gestures.

Implementation of Rollo et al., "FollowMe: a Robust Person Following
Framework Based on Visual Re-Identification and Gestures", arXiv:2311.12992.

    segmentation -> reid -> calibration (Eq. 1, lambda_d) -> localization (KF, Eq. 2)
    gestures (MediaPipe + SVM, xi filter) -> state_machine (Fig. 5) -> navigation (Eq. 3)
"""

from followme.config import FollowMeConfig

__all__ = ["FollowMeConfig"]
__version__ = "1.0.0"
