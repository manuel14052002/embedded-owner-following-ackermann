"""Build the FollowMe perception stack from a config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from followme.calibration import TargetModel
from followme.config import FollowMeConfig
from followme.gestures import GestureClassifier, GestureRecognizer
from followme.localization import CameraIntrinsics
from followme.pipeline import FollowMePerception
from followme.reid import build_feature_extractor
from followme.segmentation import build_segmenter

LOGGER = logging.getLogger(__name__)


def build_gestures(cfg: FollowMeConfig) -> Optional[GestureRecognizer]:
    g = cfg.gestures
    if not g.enabled:
        return None
    if not Path(g.classifier_path).is_file():
        LOGGER.warning("Gesture SVM '%s' not found: gestures disabled (train it with "
                       "scripts/followme_train_gestures.py); use keyboard F/W instead", g.classifier_path)
        return None
    try:
        detector = None
        if g.hand_process:
            from followme.gestures import ProcessHandDetector
            detector = ProcessHandDetector(g, static_images=g.only_target_person)
        rec = GestureRecognizer(g, GestureClassifier.load(g.classifier_path), detector)
        if g.async_mode:
            from followme.gestures import AsyncGestureRecognizer
            rec = AsyncGestureRecognizer(rec)
        return rec
    except Exception as exc:  # noqa: BLE001 - optional module, report and continue
        LOGGER.warning("Gesture module unavailable: %s", exc)
        return None


def build_perception(cfg: FollowMeConfig, device: str, intrinsics: Optional[CameraIntrinsics],
                     target_model_path: Optional[str] = None) -> FollowMePerception:
    model = None
    if target_model_path and Path(target_model_path).is_file():
        model = TargetModel.load(target_model_path)
        LOGGER.info("Loaded target model %s (lambda_d=%.3f)", target_model_path, model.threshold)
    return FollowMePerception(cfg, build_segmenter(cfg.segmentation, device),
                              build_feature_extractor(cfg.reid, device), intrinsics,
                              build_gestures(cfg), model)
