"""Motion models: a constant-velocity Kalman filter for the owner and a
short-term, identity-agnostic velocity estimator for every detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from owner_tracking.config import MotionConfig
from owner_tracking.geometry import cxcywh_to_xyxy, iou_matrix, xyxy_to_cxcywh


class MotionPredictor(ABC):
    """Interface for the owner motion model."""

    @abstractmethod
    def initialize(self, bbox: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Advance one frame and return the predicted xyxy box."""

    @abstractmethod
    def update(self, bbox: np.ndarray) -> None:
        """Correct the state with a confirmed owner measurement."""

    @abstractmethod
    def mahalanobis(self, bboxes: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of each box to the latest prediction."""

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def is_active(self) -> bool: ...

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Current centre velocity ``(vx, vy)`` in pixels/frame."""

    def damp_velocity(self, factor: float) -> None:  # optional
        """Shrink the velocity (used while the owner is lost)."""


class KalmanMotionPredictor(MotionPredictor):
    """8-state constant-velocity Kalman filter on ``(cx, cy, w, h)``.

    Process and measurement noise scale with the box height (as in DeepSORT),
    so gating behaves consistently for near and far people. Implemented with
    ``cv2.KalmanFilter`` in float64.
    """

    def __init__(self, config: MotionConfig, dt: float = 1.0) -> None:
        self.config = config
        self.dt = dt
        self._kf: Optional[cv2.KalmanFilter] = None
        self._pred_mean: Optional[np.ndarray] = None
        self._pred_cov: Optional[np.ndarray] = None
        self._H = np.zeros((4, 8), dtype=np.float64)
        self._H[:, :4] = np.eye(4)

    @property
    def is_active(self) -> bool:
        return self._kf is not None

    @property
    def velocity(self) -> np.ndarray:
        if self._kf is None:
            return np.zeros(2)
        return self._kf.statePost[4:6, 0].copy()

    def reset(self) -> None:
        self._kf = None
        self._pred_mean = None
        self._pred_cov = None

    def initialize(self, bbox: np.ndarray) -> None:
        cfg = self.config
        kf = cv2.KalmanFilter(8, 4, 0, cv2.CV_64F)
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = self.dt * np.eye(4)
        kf.transitionMatrix = transition
        kf.measurementMatrix = self._H.copy()
        z = xyxy_to_cxcywh(bbox)
        kf.statePost = np.concatenate([z, np.zeros(4)]).reshape(8, 1)
        h = max(z[3], 1.0)
        std = np.array([2 * cfg.std_weight_position * h] * 4 + [10 * cfg.std_weight_velocity * h] * 4)
        kf.errorCovPost = np.diag(std ** 2)
        kf.processNoiseCov = np.eye(8)
        kf.measurementNoiseCov = np.eye(4)
        self._kf = kf
        self._pred_mean = kf.statePost.copy()
        self._pred_cov = kf.errorCovPost.copy()

    def _require(self) -> cv2.KalmanFilter:
        if self._kf is None:
            raise RuntimeError("Motion model used before initialize()")
        return self._kf

    def predict(self) -> np.ndarray:
        kf = self._require()
        cfg = self.config
        h = max(float(kf.statePost[3, 0]), 1.0)
        std = np.array([cfg.std_weight_position * h] * 4 + [cfg.std_weight_velocity * h] * 4)
        kf.processNoiseCov = np.diag(std ** 2)
        state = kf.predict().copy()
        state[2, 0] = max(state[2, 0], 1.0)
        state[3, 0] = max(state[3, 0], 1.0)
        kf.statePre = state
        kf.statePost = state.copy()   # coasting: post == pre until a measurement arrives
        self._pred_mean = state
        self._pred_cov = kf.errorCovPre.copy()
        return cxcywh_to_xyxy(state[:4, 0]).astype(np.float32)

    def _measurement_noise(self) -> np.ndarray:
        h = max(float(self._pred_mean[3, 0]), 1.0)
        return np.diag((np.full(4, self.config.measurement_std_weight * h)) ** 2)

    def update(self, bbox: np.ndarray) -> None:
        kf = self._require()
        kf.measurementNoiseCov = self._measurement_noise()
        kf.correct(xyxy_to_cxcywh(bbox).reshape(4, 1))

    def mahalanobis(self, bboxes: np.ndarray) -> np.ndarray:
        bboxes = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        if self._kf is None or self._pred_mean is None or len(bboxes) == 0:
            return np.full(len(bboxes), np.inf)
        innovation_cov = self._H @ self._pred_cov @ self._H.T + self._measurement_noise()
        diff = xyxy_to_cxcywh(bboxes) - self._pred_mean[:4, 0]
        chol = np.linalg.cholesky(innovation_cov)
        solved = np.linalg.solve(chol, diff.T)
        return np.sum(solved ** 2, axis=0)

    def damp_velocity(self, factor: float) -> None:
        if self._kf is None:
            return
        state = self._kf.statePost.copy()
        state[4:, 0] *= factor
        self._kf.statePost = state


class ShortTermTracklets:
    """Identity-agnostic, *conservative* frame-to-frame links between detections.

    A detection is linked to a detection of the previous frame only if the IoU
    match is mutual-best and clearly better than the runner-up; otherwise a
    new tracklet starts. Tracklets therefore break at every crossing rather
    than risk swapping. They are never used to assign the owner identity;
    they provide two *negative* cues:

    * velocity: is this candidate moving the way the owner was moving?
    * co-existence: a tracklet that was visible as a *different* detection
      while the owner was confirmed in the same frame cannot be the owner.
    """

    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._boxes = np.zeros((0, 4), dtype=np.float32)
        self._velocities = np.zeros((0, 2), dtype=np.float64)
        self._ages = np.zeros(0, dtype=np.int32)
        self._ids = np.zeros(0, dtype=np.int64)
        self._missed = np.zeros(0, dtype=np.int32)   # frames since last detected
        self._crowded_ago = np.zeros(0, dtype=np.int32)  # frames since it overlapped someone
        self._next_id = 1

    def update(self, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(velocities (N, 2), ages (N,), tracklet_ids (N,))``.

        Unknown velocities are ``nan``; ``age`` counts successful links.
        Tracklets survive up to ``tracklet_max_missed_frames`` detector
        drop-outs so a brief miss does not erase what we know about a person.
        """
        cfg = self.config
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        n = len(bboxes)
        velocities = np.full((n, 2), np.nan)
        ages = np.zeros(n, dtype=np.int32)
        ids = np.zeros(n, dtype=np.int64)
        linked = np.zeros(n, dtype=bool)
        used = np.zeros(len(self._boxes), dtype=bool)
        self_overlap = iou_matrix(bboxes, bboxes)
        np.fill_diagonal(self_overlap, 0.0)
        overlapping_now = self_overlap.max(axis=1) > 0.0 if n > 1 else np.zeros(n, dtype=bool)
        crowded_ago = np.full(n, 10 ** 6, dtype=np.int32)
        if n and len(self._boxes):
            # Motion-compensated IoU: also compare against each previous box
            # advanced by its own velocity and keep the better overlap. At low
            # frame rates a walking person moves more than half a box width per
            # frame, so plain IoU alone would break every tracklet and force
            # constant re-confirmation. Plain IoU is kept because a noisy
            # velocity estimate of a nearly stationary person can push the
            # shifted box off them (which would drop their exclusion history).
            overlap = np.maximum(iou_matrix(bboxes, self._boxes),
                                 iou_matrix(bboxes, self._shifted_boxes()))
            centers = xyxy_to_cxcywh(bboxes)[:, :2]
            prev_centers = xyxy_to_cxcywh(self._boxes)[:, :2]
            for i in range(n):
                j = int(np.argmax(overlap[i]))
                best = overlap[i, j]
                if best < cfg.tracklet_match_iou or int(np.argmax(overlap[:, j])) != i:
                    continue
                row_second = np.partition(overlap[i], -2)[-2] if overlap.shape[1] > 1 else 0.0
                col_second = np.partition(overlap[:, j], -2)[-2] if overlap.shape[0] > 1 else 0.0
                if best - max(row_second, col_second) < cfg.tracklet_ambiguity_gap:
                    continue
                # A person who was overlapping someone within the last few frames
                # may have been swapped with them (e.g. the other one went
                # undetected). Such links are refused: better a new tracklet
                # than an inherited identity.
                if self._crowded_ago[j] < cfg.tracklet_isolation_frames:
                    continue
                displacement = (centers[i] - prev_centers[j]) / float(self._missed[j] + 1)
                prev_v = self._velocities[j]
                if np.all(np.isfinite(prev_v)):
                    a = cfg.tracklet_velocity_smoothing
                    displacement = a * displacement + (1 - a) * prev_v
                velocities[i] = displacement
                ages[i] = self._ages[j] + 1
                ids[i] = self._ids[j]
                crowded_ago[i] = self._crowded_ago[j] + 1
                linked[i] = True
                used[j] = True
        for i in np.flatnonzero(~linked):
            ids[i] = self._next_id
            self._next_id += 1
        crowded_ago[overlapping_now] = 0
        # Carry unmatched tracklets forward for a few frames - but only if they
        # are genuinely unobserved. A tracklet that overlaps a current detection
        # it did not link to has probably merged with / been hidden by someone
        # else; carrying it would risk a later identity swap, so it is ended.
        if n and len(self._boxes):
            merged = overlap.max(axis=0) >= 0.5 * cfg.tracklet_match_iou
        else:
            merged = np.zeros(len(self._boxes), dtype=bool)
        keep = ((~used) & (~merged) & (self._missed + 1 <= cfg.tracklet_max_missed_frames)
                & (self._crowded_ago >= cfg.tracklet_isolation_frames))
        self._boxes = np.concatenate([bboxes, self._boxes[keep]])
        self._velocities = np.concatenate([velocities, self._velocities[keep]])
        self._ages = np.concatenate([ages, self._ages[keep]])
        self._ids = np.concatenate([ids, self._ids[keep]])
        self._missed = np.concatenate([np.zeros(n, np.int32), self._missed[keep] + 1])
        self._crowded_ago = np.concatenate([crowded_ago, self._crowded_ago[keep] + 1])
        return velocities, ages, ids

    def _shifted_boxes(self) -> np.ndarray:
        """Previous boxes advanced by their velocity (unchanged if unknown)."""
        reliable = (np.all(np.isfinite(self._velocities), axis=1)
                    & (self._ages >= self.config.min_frames_for_reliable_velocity))
        shift = np.where(reliable[:, None], np.nan_to_num(self._velocities), 0.0)
        shift = shift * (self._missed[:, None] + 1)
        out = self._boxes.astype(np.float64).copy()
        out[:, [0, 2]] += shift[:, :1]
        out[:, [1, 3]] += shift[:, 1:2]
        return out.astype(np.float32)

    @property
    def live_ids(self) -> set[int]:
        """Ids of tracklets that are visible or only briefly missing."""
        return set(self._ids.tolist())


#: Backwards-compatible name.
DetectionVelocityEstimator = ShortTermTracklets
