"""
owner_tracking_upgrade.py
─────────────────────────────────────────────────────────────────────────────
Drop-in upgrade module for human_tracker_ros2/person_tracker.py, implementing
techniques from:

 [P1] "A person-following method based on monocular camera for quadruped
      robots" (Biomimetic Intelligence and Robotics, 2022):
      Kalman-filter target tracking, online-adaptive appearance model,
      motion+appearance target recapture.

 [P2] "The human-following strategy for mobile robots in mixed environments"
      (Robotics and Autonomous Systems, 2022):
      FOV-constraint servo control, smooth distance/orientation keeping.

Classes:
  KalmanCV2D       — constant-velocity KF on image-plane target center.
                     Predicts through YOLO dropouts / brief occlusion. [P1]
  AdaptiveGallery  — enrollment anchor + rolling gallery of recent owner
                     embeddings; adapts to lighting/pose drift without
                     forgetting the enrolled identity.                 [P1]
  RecaptureManager — after target loss: score candidates by
                     appearance × motion-consistency; re-lock only when
                     both agree.                                       [P1]
  fov_servo        — angular-gain shaping that prioritizes keeping the
                     target inside the camera FOV.                     [P2]

No external deps beyond numpy. All classes unit-testable standalone.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import math
from collections import deque
from typing import Optional, Tuple, List

import numpy as np


# ═════════════════════════════════════════════════════════════════════════
#  [P1] Kalman filter — constant-velocity model on (cx, cy) pixel center
# ═════════════════════════════════════════════════════════════════════════
class KalmanCV2D:
    """
    State: [x, y, vx, vy].  Measurement: [x, y] (bbox center, pixels).
    predict(dt) advances state; update(z) fuses a detection.
    gate(z) returns Mahalanobis distance for association gating.
    """

    def __init__(self,
                 process_var: float = 400.0,     # px²/s² — high for fast human motion
                 meas_var: float = 25.0):        # px² detection noise
        self.x = np.zeros(4)                     # state
        self.P = np.eye(4) * 500.0               # covariance
        self.q = process_var
        self.R = np.eye(2) * meas_var
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.initialized = False
        self._last_t = time.monotonic()

    def reset(self):
        self.initialized = False
        self.P = np.eye(4) * 500.0

    def init(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.initialized = True
        self._last_t = time.monotonic()

    def _F_Q(self, dt: float):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        # White-accel process noise
        q = self.q
        Q = np.array([
            [dt**4/4,       0, dt**3/2,       0],
            [      0, dt**4/4,       0, dt**3/2],
            [dt**3/2,       0,   dt**2,       0],
            [      0, dt**3/2,       0,   dt**2]]) * q
        return F, Q

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """Advance state by dt (auto-computed if None). Returns predicted (x,y)."""
        if not self.initialized:
            return (0.0, 0.0)
        now = time.monotonic()
        if dt is None:
            dt = max(1e-3, now - self._last_t)
        self._last_t = now
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return (float(self.x[0]), float(self.x[1]))

    def gate(self, cx: float, cy: float) -> float:
        """Mahalanobis distance of measurement to predicted position."""
        if not self.initialized:
            return 0.0
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            return float(math.sqrt(y @ np.linalg.solve(S, y)))
        except np.linalg.LinAlgError:
            return 1e9

    def update(self, cx: float, cy: float):
        if not self.initialized:
            self.init(cx, cy)
            return
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular covariance - skip this update rather than crash the
            # whole tracker. Next frame re-initialises cleanly.
            return
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def speed_px(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


# ═════════════════════════════════════════════════════════════════════════
#  [P1] Online-adaptive appearance model
#       (practical analogue of CCF + online boosting for an OSNet pipeline)
# ═════════════════════════════════════════════════════════════════════════
class AdaptiveGallery:
    """
    Keeps the ENROLLMENT embedding as a permanent anchor (identity never
    drifts away from what was enrolled) plus a rolling gallery of recent
    high-confidence embeddings (adapts to lighting / pose / viewpoint).

    match() score = max(anchor_sim, mean of top-k gallery sims)
    update() only accepts embeddings when the match was confident AND the
    scene was unambiguous — this is the contamination guard the paper's
    online-boosting selectivity provides.
    """

    def __init__(self,
                 anchor_embedding: np.ndarray,
                 gallery_size: int = 12,
                 top_k: int = 3,
                 update_threshold: float = 0.72):
        a = np.asarray(anchor_embedding, dtype=np.float32).ravel()
        self.anchor = a / (np.linalg.norm(a) + 1e-9)
        self.gallery: deque = deque(maxlen=gallery_size)
        self.top_k = top_k
        self.update_threshold = update_threshold

    @staticmethod
    def _norm(e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float32).ravel()
        return e / (np.linalg.norm(e) + 1e-9)

    def match(self, embedding: np.ndarray) -> float:
        """Cosine similarity of candidate vs anchor + adaptive gallery."""
        e = self._norm(embedding)
        anchor_sim = float(self.anchor @ e)
        if not self.gallery:
            return anchor_sim
        sims = sorted((float(g @ e) for g in self.gallery), reverse=True)
        top = sims[: self.top_k]
        gallery_sim = sum(top) / len(top)
        return max(anchor_sim, gallery_sim)

    def update(self, embedding: np.ndarray,
               match_score: float,
               scene_unambiguous: bool):
        """
        Add to gallery only when:
          - the frame matched confidently (score > update_threshold), and
          - only one person candidate was in view (no contamination risk).
        """
        if scene_unambiguous and match_score >= self.update_threshold:
            self.gallery.append(self._norm(embedding))

    def stats(self) -> str:
        return f"gallery={len(self.gallery)}"


# ═════════════════════════════════════════════════════════════════════════
#  [P1] Recapture: motion prediction × appearance after target loss
# ═════════════════════════════════════════════════════════════════════════
class RecaptureManager:
    """
    While the target is lost, the KF keeps predicting (coast). Candidates
    are scored:  score = w_app·appearance + w_mot·exp(-d_pred / sigma)
    Re-lock requires BOTH a decent appearance match and motion consistency
    early in the loss; as loss duration grows, motion prediction decays and
    appearance dominates (person may have changed direction).
    """

    def __init__(self,
                 coast_time: float = 0.6,       # s of pure KF coasting (brief blur only)
                 recapture_window: float = 8.0, # s before giving up entirely
                 w_appearance: float = 0.65,
                 w_motion: float = 0.35,
                 sigma_px: float = 90.0,
                 accept_threshold: float = 0.50,
                 appearance_floor: float = 0.50):
        self.coast_time = coast_time
        self.window = recapture_window
        self.w_app = w_appearance
        self.w_mot = w_motion
        self.sigma = sigma_px
        self.accept = accept_threshold
        self.app_floor = appearance_floor
        self._lost_since: Optional[float] = None

    def mark_lost(self):
        if self._lost_since is None:
            self._lost_since = time.monotonic()

    def mark_found(self):
        self._lost_since = None

    @property
    def lost_duration(self) -> float:
        return 0.0 if self._lost_since is None else time.monotonic() - self._lost_since

    @property
    def coasting(self) -> bool:
        """True during the short window where KF prediction is trustworthy."""
        return self._lost_since is not None and self.lost_duration <= self.coast_time

    @property
    def expired(self) -> bool:
        return self._lost_since is not None and self.lost_duration > self.window

    def score(self,
              appearance_sim: float,
              cand_cx: float, cand_cy: float,
              pred_cx: float, pred_cy: float) -> float:
        """Combined recapture score for one candidate."""
        if appearance_sim < self.app_floor:
            return 0.0                                  # hard identity floor
        d = math.hypot(cand_cx - pred_cx, cand_cy - pred_cy)
        motion = math.exp(-d / self.sigma)
        # Motion trust decays with loss duration (person may have turned)
        trust = max(0.0, 1.0 - self.lost_duration / self.window)
        w_mot = self.w_mot * trust
        w_app = 1.0 - w_mot
        return w_app * appearance_sim + w_mot * motion

    def try_recapture(self,
                      candidates: List[Tuple[float, float, float]],
                      pred_cx: float, pred_cy: float
                      ) -> Optional[int]:
        """
        candidates: list of (appearance_sim, cx, cy).
        Returns index of accepted candidate or None.
        """
        best_i, best_s = None, self.accept
        for i, (sim, cx, cy) in enumerate(candidates):
            s = self.score(sim, cx, cy, pred_cx, pred_cy)
            if s > best_s:
                best_s, best_i = s, i
        return best_i


# ═════════════════════════════════════════════════════════════════════════
#  [P2] FOV servo — keep the target inside the camera field of view
# ═════════════════════════════════════════════════════════════════════════
def fov_servo(bearing_rad: float,
              base_angular: float,
              fov_half_rad: float = math.radians(34.7),   # D435 color HFOV/2
              margin: float = 0.72,
              edge_gain: float = 2.2,
              max_angular: float = 1.2) -> float:
    """
    Shapes the angular command so FOV retention dominates near the image
    edges [P2]. Inside `margin`·FOV the base PID output passes through;
    beyond it, gain ramps quadratically so the robot turns hard before the
    person exits the frame (a fixed camera cannot recover once they do).
    """
    edge = abs(bearing_rad) / fov_half_rad          # 0 center … 1 at edge
    if edge <= margin:
        out = base_angular
    else:
        # Quadratic ramp from 1× at margin to edge_gain× at FOV edge
        t = (edge - margin) / (1.0 - margin)
        boost = 1.0 + (edge_gain - 1.0) * t * t
        # Boosted command always steers toward the person
        toward = -math.copysign(1.0, bearing_rad)
        out = toward * max(abs(base_angular) * boost,
                           0.35 * boost)             # floor: act even if PID small
    return max(-max_angular, min(max_angular, out))


# ═════════════════════════════════════════════════════════════════════════
#  Smooth velocity shaping [P2] — trapezoidal accel limit
# ═════════════════════════════════════════════════════════════════════════
class VelocityShaper:
    def __init__(self, max_accel: float = 0.6, max_ang_accel: float = 2.0):
        self.a = max_accel
        self.aw = max_ang_accel
        self._lin = 0.0
        self._ang = 0.0
        self._t = time.monotonic()

    def reset(self):
        self._lin = self._ang = 0.0
        self._t = time.monotonic()

    def shape(self, lin_target: float, ang_target: float) -> Tuple[float, float]:
        now = time.monotonic()
        dt = max(1e-3, now - self._t)
        self._t = now
        self._lin += max(-self.a * dt,  min(self.a * dt,  lin_target - self._lin))
        self._ang += max(-self.aw * dt, min(self.aw * dt, ang_target - self._ang))
        return self._lin, self._ang
