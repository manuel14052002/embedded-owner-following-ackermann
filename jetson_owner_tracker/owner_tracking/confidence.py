"""Confidence channels, temporal smoothing and the confidence-to-state map."""

from __future__ import annotations

import math

from owner_tracking.config import ConfidenceConfig, StateConfig
from owner_tracking.datatypes import ConfidenceReport, TrackState


def _finite(value: float, default: float) -> float:
    return value if math.isfinite(value) else default


class ConfidenceManager:
    """Fuses per-channel confidences into one smoothed overall confidence.

    Channels: appearance, motion, spatial (IoU + position), detection and
    identity (score and separation from the runner-up). ``overall`` is an
    exponential moving average, so a single good frame after an ambiguous
    period cannot instantly restore full trust; frames without an accepted
    measurement decay confidence multiplicatively.
    """

    def __init__(self, config: ConfidenceConfig, state_config: StateConfig) -> None:
        self.config = config
        self.state_config = state_config
        self.reset()

    def reset(self) -> None:
        self.report = ConfidenceReport()

    def initialize(self, detection_confidence: float) -> ConfidenceReport:
        self.report = ConfidenceReport(1.0, 1.0, 1.0, detection_confidence, 1.0, 1.0)
        return self.report

    def calculate_confidence(self, *, appearance: float, motion: float, spatial: float,
                             detection: float, association_score: float,
                             margin: float) -> ConfidenceReport:
        c = self.config
        identity = 0.5 * association_score + 0.5 * min(1.0, max(0.0, margin / c.margin_saturation))
        channels = {
            "appearance": _finite(appearance, 0.0),
            "motion": _finite(motion, 0.5),     # unknown motion is neutral
            "spatial": _finite(spatial, 0.5),
            "detection": _finite(detection, 0.0),
            "identity": identity,
        }
        weights = {
            "appearance": c.appearance_weight, "motion": c.motion_weight,
            "spatial": c.spatial_weight, "detection": c.detection_weight,
            "identity": c.identity_weight,
        }
        total_w = sum(weights.values())
        raw = sum(weights[k] * channels[k] for k in channels) / total_w
        overall = c.smoothing * raw + (1.0 - c.smoothing) * self.report.overall
        self.report = ConfidenceReport(overall=overall, **channels)
        return self.report

    def decay(self, factor: float) -> ConfidenceReport:
        r = self.report
        self.report = ConfidenceReport(
            appearance=r.appearance * factor, motion=r.motion * factor, spatial=r.spatial * factor,
            detection=r.detection * factor, identity=r.identity * factor, overall=r.overall * factor)
        return self.report

    def state_for(self, overall: float) -> TrackState:
        """Configurable confidence bands -> state."""
        s = self.state_config
        if overall >= s.tracking_threshold:
            return TrackState.TRACKING
        if overall >= s.low_confidence_threshold:
            return TrackState.LOW_CONFIDENCE
        if overall >= s.ambiguous_threshold:
            return TrackState.AMBIGUOUS
        return TrackState.LOST
