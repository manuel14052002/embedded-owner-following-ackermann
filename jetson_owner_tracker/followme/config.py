"""Typed configuration for the FollowMe pipeline (Rollo et al., arXiv:2311.12992).

Every paper constant is a default here and is marked with the section it
comes from. Unknown YAML keys raise, exactly like ``owner_tracking.config``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from owner_tracking.config import _update_dataclass


@dataclass
class SegmentationConfig:
    """Person instance segmentation (Sect. III-A).

    The paper uses YOLACT++ pre-trained on COCO. YOLACT++ needs a DCNv2 CUDA
    extension that does not build on current PyTorch / JetPack, so the default
    backend is an Ultralytics ``*-seg`` model, which produces the same output
    (per-person box + binary mask, COCO "person" class) in real time.
    """

    backend: str = "ultralytics"            # ultralytics (.pt / .onnx / .engine)
    model_path: str = "yolo11n-seg.pt"      # .pt / .onnx / TensorRT .engine
    confidence_threshold: float = 0.40
    nms_iou_threshold: float = 0.50
    image_size: int = 640
    half_precision: bool = True             # only honoured on CUDA
    max_detections: int = 20
    min_box_height: float = 48.0            # smaller people give unusable Re-ID crops
    min_mask_fraction: float = 0.05         # mask pixels / box pixels
    person_class_id: int = 0
    obstacle_class_ids: list[int] = field(default_factory=list)  # COCO ids kept as obstacles from the
                                            # same YOLO run: 13 bench, 56 chair, 57 couch, 60 dining table
    obstacle_confidence: float = 0.35
    remove_background: bool = True          # Sect. III-A: image (element-wise) x mask


@dataclass
class ReIDConfig:
    """Appearance features (Sect. III-B, VI).

    Paper: MMT pre-training, IBN-ResNet-50 trained on MSMT17, D = 256.
    """

    backend: str = "mmt"                    # mmt | onnx | tensorrt | torchreid | histogram
    model_path: str = ""                    # MMT checkpoint (.pth.tar) or .onnx
    feature_dim: int = 256                  # D (Sect. VI)
    input_height: int = 256
    input_width: int = 128
    batch_size: int = 16
    half_precision: bool = False
    l2_normalize: bool = True               # MMT normalises features at test time
    torchreid_model_name: str = "osnet_x0_25"


@dataclass
class CalibrationConfig:
    """Calibration phase (Sect. III-B)."""

    num_samples: int = 150                  # N_c + N_lambda
    calibration_fraction: float = 2.0 / 3.0  # Sect. VI-B: 2/3 for mu/sigma, 1/3 for lambda
    threshold_k: float = 2.0                # lambda_d = mu_d + k * sigma_d
    metric: str = "eq1"                     # eq1 (paper) | cosine | both | gallery
    min_interval_s: float = 0.066           # spacing between accepted samples
    duration_s: Optional[float] = None      # e.g. 30: spread num_samples evenly over this many seconds
    require_single_person: bool = True      # calibration frames must show only the target
    sigma_floor_relative: float = 0.05      # sigma_i >= this * mean(sigma) (numerical safety)
    sigma_floor_absolute: float = 1e-6
    distance_bins_m: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.5])
    min_bins_covered: int = 2               # warn if the target stayed at one distance
    shuffle_seed: int = 0


@dataclass
class IdentityConfig:
    """Multi-view gallery + negative bank + online update (``calibration.metric: gallery``).

    Not in the paper (its future work iii suggests continual learning).
    """

    top_k: int = 3                          # similarity = mean of the k closest gallery views
    negative_margin: float = 0.03           # sim(you) must beat sim(best known stranger) by this
    max_negatives: int = 300
    negative_max_iou: float = 0.05          # only clearly separate people become negatives
    negative_confident_factor: float = 0.8  # ...and only while the target is confidently identified
    online_update: bool = True
    max_online: int = 150
    online_interval_s: float = 1.0
    online_confident_factor: float = 0.7    # add a view only if d <= this * threshold
    online_min_negative_margin: float = 0.10
    online_max_iou: float = 0.10            # nobody overlapping the target (clean crop)
    online_duplicate_sim: float = 0.97      # skip near-duplicates of existing views
    online_from_tracking: bool = False      # also learn views of a tracked owner who is ALONE in view
                                            # (up to tracking_threshold_factor*threshold): adapts to
                                            # new rooms/lighting that calibration never saw
    continuity_iou: float = 0.0             # >0: last frame's owner box overlapping by this much,
                                            # with nobody else near it, stays the owner
    continuity_factor: float = 2.5          # ...if its Re-ID distance is below factor*threshold
    tracking_threshold_factor: float = 1.0  # >1: while tracked, accept up to factor*threshold
                                            # for a candidate inside the KF gate (needs gate_m)


@dataclass
class OwnershipConfig:
    """Claim / release the robot with a raised open palm (not in the paper).

    NO_OWNER --one person holds a raised open palm claim_hold_s--> CLAIMING (auto-calibration)
    CLAIMING --calibration done--> OWNED (FollowMe) ; claimant lost claim_lost_s --> NO_OWNER
    OWNED --owner holds a raised open palm release_hold_s, or unseen owner_lost_release_s--> NO_OWNER
    Needs the gesture classifier (open palm = WAIT class).
    """

    enabled: bool = False
    claim_hold_s: float = 2.0
    release_hold_s: float = 3.0
    raised_fraction: float = 0.25          # wrist in the top 25 % of the person box = raised
    gesture_gap_s: float = 0.6             # tolerated hand-detection drop-out while holding
    max_candidates: int = 4                # hands checked on the N largest people (NO_OWNER)
    hands_per_frame: int = 0               # >0: check only this many of them per frame, rotating
    hand_check_every: int = 1              # run the hand checks only every Nth frame while free
    hand_async: bool = False               # hand checks in a background thread (loop never waits)
                                           # (MediaPipe runs on the CPU: ~25 ms per person)
    calibration_s: float = 20.0            # claimant calibration duration
    claim_lost_s: float = 3.0              # abort the claim if the claimant disappears this long
    owner_lost_release_s: float = 60.0     # auto-release; 0 disables
    release_cooldown_s: float = 3.0        # no new claim right after a release
    start_following: bool = True           # FOLLOW as soon as the new owner is identified
    follow_timeout_s: float = 5.0
    voice_arm_s: float = 10.0              # after a voice 'enroll'/'follow me': window to raise a hand
    voice_claim_hold_s: float = 0.5        # ...and the raised palm only needs this long


@dataclass
class LocalizationConfig:
    """Target localisation and Kalman filtering (Sect. III-C)."""

    min_depth_m: float = 0.3
    max_depth_m: float = 8.0
    mask_erosion_px: int = 5                # keeps depth from the mask border out of the cloud
    pixel_stride: int = 2                   # subsample the mask for speed
    min_points: int = 50
    centroid: str = "mean"                  # mean (paper) | median (robust option)
    depth_inlier_band_m: float = 0.0        # >0: keep points within band of median depth
    expiration_time_s: float = 3.0          # t_exp (Sect. VI)
    mirror_x: bool = False                  # flip left/right at the source if the camera reads mirrored
    bbox_range_fallback: bool = False       # RGB-only testing: range from box height (not in paper)
    assumed_person_height_m: float = 1.7
    confirm_frames: int = 1                 # >1: new target must pass N consecutive frames (not in paper)
    confirm_radius_m: float = 0.5           # ...at a consistent 3-D position
    gate_m: Optional[float] = None          # while tracked: accept only near the KF prediction (not in paper)
    max_person_speed_mps: float = 1.5       # gate growth per second unseen
    process_noise_accel: float = 1.0        # m/s^2, constant-velocity KF
    measurement_noise_m: float = 0.10


@dataclass
class GestureConfig:
    """Hand gesture detection (Sect. III-D)."""

    enabled: bool = True
    backend: str = "auto"                   # auto | solutions | tasks
    task_model_path: str = "hand_landmarker.task"   # only for the MediaPipe Tasks API
    classifier_path: str = "models/gesture_svm.joblib"
    consecutive_required: int = 8           # xi (Sect. III-D); a2r_step3: was 5 - an open hand while walking
                                            # stopped the car (WAIT). ~0.7 s of the same gesture now.
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    only_target_person: bool = True         # run on the re-identified person's crop only
    async_mode: bool = False                # run MediaPipe in a background thread (no loop blocking)
    hand_process: bool = False              # run MediaPipe in a separate PROCESS (no GIL sharing)
    crop_padding: float = 0.15
    mirror_left_hands: bool = True
    svm_c: float = 10.0
    svm_gamma: str = "scale"


@dataclass
class NavigationConfig:
    """Navigation and safety (Sect. IV, VI)."""

    max_velocity_mps: float = 0.3           # v_max (Sect. VI)
    safety_factor: float = 1.4              # Eq. 3
    safety_radius_m: Optional[float] = None  # None -> Eq. 3 (1.4 * 0.3 * 3 = 1.26 m)
    goal_update_distance_m: float = 0.20    # resend a goal only when the target moved this much
    goal_update_period_s: float = 1.0
    goal_standoff_m: float = 0.0            # 0 = paper (goal at the target position)
    search_angular_speed: float = 0.5       # rad/s
    search_linear_speed: float = 0.0        # >0 for Ackermann bases that cannot spin in place
    search_max_angle_rad: float = 6.283185307179586   # "at most one complete turn"
    ackermann_wheelbase_m: float = 0.0      # >0: car-like base; search angle uses v*tan(steer)/L
    ackermann_max_steer_deg: float = 30.0   # mechanical steering limit (same as the serial bridge)
    search_move_s: float = 0.0              # >0: stop-and-look search: turn this long...
    search_pause_s: float = 0.0             # ...then stand still this long (sharp images for Re-ID)
    # direct cmd_vel following (ROS control_mode:=cmd_vel), instead of Nav2 goals
    follow_distance_m: Optional[float] = None   # None -> safety radius + 0.3 m
    k_linear: float = 0.8                   # v = k_linear * (distance - follow_distance)
    k_angular: float = 1.2                  # w = k_angular * bearing
    max_angular_rps: float = 0.8
    accel_mps2: float = 0.4                 # speed-up limit
    decel_mps2: float = 1.0                 # slow-down limit (braking is allowed to be quicker)
    turn_slowdown: bool = True              # v *= cos(bearing): do not rush while turning
    pid_kp: float = 1.0                     # steering_law pid: delta = kp*a + ki*int(a) + kd*da/dt
    pid_ki: float = 0.3
    pid_kd: float = 0.0
    pid_i_limit_rad: float = 0.5            # bound of the integral term (anti-windup)
    # steering_law pi_goal: the Raspberry Pi follow pipeline's goal law (smart_planner v2.2)
    goal_k_lin: float = 0.8                 # v = k * (distance - stand-off), then capped
    goal_max_speed: float = 0.24            # m/s while following
    goal_turn_speed: float = 0.199          # m/s on a full-lock arc toward an owner far to the side
    goal_turn_bearing_deg: float = 75.0     # owner further round than this -> full-lock arc
    goal_bearing_deadband_deg: float = 3.0  # tracking jitter below this -> drive straight
    goal_pp_lookahead_m: float = 1.2        # pure pursuit: never aim closer than this
    goal_k_ang: float = 1.0                 # scale on the pure-pursuit curvature
    goal_tol_m: float = 0.10                # this close to the stand-off point -> hold
    min_drive_speed: float = 0.168          # the ESC stalls below this: never crawl slower
    owner_smooth_alpha: float = 1.0         # low-pass on the owner position (1 = off)
    goal_trim_ki: float = 0.0               # >0: slow trim integral (removes servo offset only)
    goal_trim_band_deg: float = 12.0        # ...integrates only while the owner is this close to centre
    goal_trim_max_deg: float = 6.0          # ...and never adds more steering than this
    goal_trim_leak_s: float = 30.0          # ...slowly forgets (time constant)
    pp_reference: str = "camera"            # pure pursuit seen from the camera (keeps the owner in
                                            # view) | base (rear-axle textbook form)
    steering_law: str = "proportional"      # proportional (w = k*bearing) | pure_pursuit (Ackermann,
                                            # uses ackermann_wheelbase_m / ackermann_max_steer_deg)
    apf_enabled: bool = False               # artificial potential field obstacle avoidance (lidar)
    apf_influence_m: float = 1.2            # obstacles farther than this do not repel
    apf_k_rep: float = 0.6                  # repulsion gain
    apf_owner_exclusion_m: float = 0.5      # lidar points this close to the owner are the owner
    apf_lookahead_m: float = 1.2            # distance of the virtual steering target
    hold_hysteresis_m: float = 0.0          # >0: after stopping, restart beyond follow distance + this
    steer_smoothing: float = 0.0            # 0..0.9: weight of the previous steering angle (EMA)
    steer_rate_limit_dps: float = 0.0       # >0: max change of the steering angle per second
    steer_from_measurement: bool = False    # True: aim at this frame's raw detection (no filter lag);
                                            # the Kalman estimate is used only when no detection
    min_steer_speed_mps: float = 0.0        # >0: bases that only steer while rolling (many Ackermann
                                            # ESP32/RC setups) keep this minimum speed whenever they
                                            # must turn toward the owner (may exceed max_velocity_mps)
    cmd_timeout_s: float = 0.5              # watchdog: zero velocity if no fresh command


@dataclass
class StateMachineConfig:
    """Decision making (Sect. V)."""

    initial_state: str = "WAIT"             # experiments start waiting for a Follow gesture
    steady_requires_follow_gesture: bool = False
    wait_from_any_state: bool = True


@dataclass
class FollowMeConfig:
    """Root configuration."""

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    ownership: OwnershipConfig = field(default_factory=OwnershipConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "FollowMeConfig":
        cfg = cls()
        _update_dataclass(cfg, data or {}, prefix="")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FollowMeConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root in {path} must be a mapping")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        c = self.calibration
        if not 0.0 < c.calibration_fraction < 1.0:
            raise ValueError("calibration.calibration_fraction must be in (0, 1)")
        if c.num_samples < 6:
            raise ValueError("calibration.num_samples must be >= 6")
        if self.gestures.consecutive_required < 1:
            raise ValueError("gestures.consecutive_required must be >= 1")
        if self.localization.expiration_time_s <= 0:
            raise ValueError("localization.expiration_time_s must be > 0")
        if c.metric not in ("eq1", "cosine", "both", "gallery"):
            raise ValueError("calibration.metric must be eq1 | cosine | both | gallery")
        if self.localization.centroid not in ("mean", "median"):
            raise ValueError("localization.centroid must be 'mean' or 'median'")
        if self.state_machine.initial_state not in ("STEADY", "FOLLOW", "SEARCH", "WAIT"):
            raise ValueError("state_machine.initial_state must be STEADY|FOLLOW|SEARCH|WAIT")
        if self.navigation.steering_law not in ("proportional", "pure_pursuit", "pid", "pi_goal"):
            raise ValueError("navigation.steering_law must be proportional, pure_pursuit, pid or pi_goal")
        if self.navigation.max_velocity_mps <= 0:
            raise ValueError("navigation.max_velocity_mps must be > 0")
