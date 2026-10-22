"""Typed, YAML-loadable configuration for every tracker component.

All thresholds live here so experiments can be reproduced from a single file.
Unknown keys raise immediately: a typo in an experiment config must never be
silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class DetectorConfig:
    """Person detector settings."""

    backend: str = "ultralytics"
    model_path: str = "yolo11n.pt"          # .pt, .onnx or TensorRT .engine
    confidence_threshold: float = 0.30
    nms_iou_threshold: float = 0.50
    image_size: int = 640
    half_precision: bool = True             # only honoured on CUDA
    max_detections: int = 50
    min_box_height: float = 24.0
    person_class_id: int = 0


@dataclass
class ReIDConfig:
    """Appearance embedding settings."""

    backend: str = "torchreid"              # torchreid | torchvision | histogram
    model_name: str = "osnet_x0_25"
    model_path: str = ""                    # ReID-trained weights (strongly recommended)
    input_height: int = 256
    input_width: int = 128
    batch_size: int = 32
    half_precision: bool = True
    fallback_backends: list[str] = field(default_factory=lambda: ["torchvision", "histogram"])
    prefilter_by_motion_gate: bool = False  # skip embedding far-away detections


@dataclass
class MotionConfig:
    """Kalman filter and motion-consistency settings."""

    std_weight_position: float = 0.05       # process noise, relative to box height
    std_weight_velocity: float = 0.00625
    measurement_std_weight: float = 0.05
    gating_chi2: float = 9.4877             # 95% quantile, chi-square 4 dof
    gate_growth_per_frame: float = 0.05     # linear search-radius growth when unseen
    max_gate_multiplier: float = 4.0
    velocity_damping_when_lost: float = 0.97
    velocity_sigma: float = 0.01            # expected velocity noise, x box height/frame
    velocity_sigma_min_px: float = 1.0
    velocity_gate_z: float = 3.0
    velocity_memory_frames: int = 30        # trust owner velocity this long after loss
    min_frames_for_reliable_velocity: int = 3
    tracklet_match_iou: float = 0.50
    tracklet_ambiguity_gap: float = 0.10
    tracklet_velocity_smoothing: float = 0.5
    tracklet_max_missed_frames: int = 5
    tracklet_isolation_frames: int = 3          # refuse links this soon after an overlap
    velocity_strikes_to_exclude: int = 8        # 0 disables; see README
    max_speed_heights_per_frame: float = 0.06   # physical reachability (~30 fps)
    reachability_base_heights: float = 0.5


@dataclass
class AssociationWeights:
    """Weights of the multi-factor association score (renormalised at runtime)."""

    appearance: float = 0.40
    iou: float = 0.20
    position: float = 0.15
    scale: float = 0.10
    motion: float = 0.10
    detection: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class AssociationConfig:
    """Per-candidate scoring settings."""

    tracking_weights: AssociationWeights = field(default_factory=AssociationWeights)
    reid_weights: AssociationWeights = field(
        default_factory=lambda: AssociationWeights(
            appearance=0.70, iou=0.0, position=0.05, scale=0.10, motion=0.10, detection=0.05))
    appearance_similarity_floor: float = 0.20
    appearance_similarity_ceiling: float = 0.85
    position_sigma: float = 0.5             # in units of predicted box height
    scale_sigma: float = 0.25               # in log-ratio units
    min_appearance_similarity: float = 0.35  # hard floor while tracking
    distractor_margin: float = 0.05


@dataclass
class IdentityConfig:
    """Identity decision, anti-switch and gallery settings."""

    identity_threshold: float = 0.55
    min_score_margin: float = 0.12
    reid_threshold: float = 0.60            # raw cosine floor while LOST
    reid_score_threshold: float = 0.60
    reid_min_score_margin: float = 0.15
    reid_min_motion_score: float = 0.05     # while a prediction still exists
    confirmation_frames: int = 4
    reid_confirmation_frames: int = 5
    continuity_iou: float = 0.30
    pending_match_iou: float = 0.30
    short_gap_frames: int = 2
    click_ambiguity_ratio: float = 0.15
    gallery_size: int = 30
    gallery_top_k: int = 5
    anchor_weight: float = 0.40
    gallery_update_confidence: float = 0.80
    gallery_update_min_similarity: float = 0.55
    gallery_update_min_margin: float = 0.20
    gallery_dedup_similarity: float = 0.97
    gallery_update_interval: int = 5
    gallery_min_stable_frames: int = 10
    gallery_isolation_iou: float = 0.10
    distractor_bank_size: int = 64
    distractor_isolation_iou: float = 0.05
    distractor_max_owner_similarity: float = 0.80  # look-alikes are not useful hard negatives


@dataclass
class ConfidenceConfig:
    """Fusion of the separate confidence channels."""

    appearance_weight: float = 0.35
    motion_weight: float = 0.15
    spatial_weight: float = 0.15
    detection_weight: float = 0.05
    identity_weight: float = 0.30
    margin_saturation: float = 0.30
    smoothing: float = 0.60                 # EMA weight of the newest observation
    miss_decay: float = 0.85
    ambiguous_decay: float = 0.95


@dataclass
class StateConfig:
    """State-machine thresholds."""

    tracking_threshold: float = 0.75
    low_confidence_threshold: float = 0.50
    ambiguous_threshold: float = 0.30
    lost_after_missed_frames: int = 10
    lost_after_unresolved_frames: int = 30
    max_prediction_frames: int = 60


@dataclass
class LoggingConfig:
    enabled: bool = True
    csv_path: str = "logs/owner_track.csv"
    interval: int = 1
    flush_every: int = 30


@dataclass
class RuntimeConfig:
    device: str = "auto"                    # auto | cpu | cuda | cuda:N
    seed: int = 0
    deterministic: bool = True


@dataclass
class TrackerConfig:
    """Root configuration object."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    state: StateConfig = field(default_factory=StateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "TrackerConfig":
        cfg = cls()
        _update_dataclass(cfg, data or {}, prefix="")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrackerConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root in {path} must be a mapping")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Raise ``ValueError`` for internally inconsistent settings."""
        s = self.state
        if not (0.0 <= s.ambiguous_threshold <= s.low_confidence_threshold
                <= s.tracking_threshold <= 1.0):
            raise ValueError("State thresholds must satisfy 0 <= ambiguous <= low <= tracking <= 1")
        for name in ("tracking_weights", "reid_weights"):
            weights = getattr(self.association, name).as_dict()
            if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError(f"association.{name} must be non-negative with positive sum")
        a = self.association
        if a.appearance_similarity_ceiling <= a.appearance_similarity_floor:
            raise ValueError("appearance_similarity_ceiling must exceed the floor")
        i = self.identity
        if i.confirmation_frames < 1 or i.reid_confirmation_frames < 1:
            raise ValueError("confirmation frame counts must be >= 1")
        if not 0.0 <= i.anchor_weight <= 1.0:
            raise ValueError("identity.anchor_weight must be in [0, 1]")
        if self.logging.interval < 1:
            raise ValueError("logging.interval must be >= 1")


def _update_dataclass(obj: Any, data: dict[str, Any], prefix: str) -> None:
    valid = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown configuration key '{prefix}{key}'")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)
