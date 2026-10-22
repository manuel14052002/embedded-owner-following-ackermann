"""Target localisation and filtering (paper Sect. III-C, IV).

1. The re-identified person's mask selects depth pixels, which are
   deprojected into a segmented point cloud in the camera optical frame
   (x right, y down, z forward); ``c_p`` is its 3-D centroid (Fig. 1d).
2. A constant-velocity Kalman filter smooths ``c_p``. It stops integrating
   when no measurement arrived for the expiration time ``t_exp``; the
   filtered position ``c_p_filt`` is then no longer reported.
3. ``m_p_filt = m_T_c * c_p_filt`` (Eq. 2); the 2-D goal is its projection on
   the map XY plane, and ``theta`` is the heading of the target seen from the
   camera.
4. Safety radius ``d_safe = 1.4 * v_max * t_exp`` (Eq. 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from followme.config import LocalizationConfig


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics of the colour image (depth must be aligned to it)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_realsense(cls, intr: Any) -> "CameraIntrinsics":
        return cls(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height)

    @classmethod
    def from_camera_info(cls, k: Any, width: int, height: int) -> "CameraIntrinsics":
        """From a ROS ``sensor_msgs/CameraInfo.k`` (row-major 3x3)."""
        return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]), int(width), int(height))

    def deproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels + depth (m) -> ``(N, 3)`` points in the camera optical frame."""
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=1)


def safety_radius(v_max: float, t_exp: float, factor: float = 1.4) -> float:
    """Eq. 3: distance travelled at ``v_max`` during ``t_exp`` plus 40 %."""
    return factor * v_max * t_exp


def segmented_point_cloud(depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics,
                          cfg: LocalizationConfig) -> np.ndarray:
    """Target point cloud from the aligned depth image and the person mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return np.zeros((0, 3))
    cols = np.flatnonzero(mask.any(axis=0))
    s = max(1, cfg.pixel_stride)
    pad = max(0, cfg.mask_erosion_px) + 1               # erode inside a padded region = full-frame result
    H, W = mask.shape
    y0 = (max(0, rows[0] - pad) // s) * s               # start on the stride grid (same samples)
    x0 = (max(0, cols[0] - pad) // s) * s
    y1, x1 = min(H, rows[-1] + 1 + pad), min(W, cols[-1] + 1 + pad)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    if cfg.mask_erosion_px > 0:
        k = 2 * cfg.mask_erosion_px + 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sub_mask = m[::s, ::s].astype(bool)
    vs, us = np.nonzero(sub_mask)
    z = depth_m[y0:y1:s, x0:x1:s][vs, us].astype(np.float64)
    vs = vs * s + y0
    us = us * s + x0
    good = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    if cfg.depth_inlier_band_m > 0 and good.any():
        med = np.median(z[good])
        good &= np.abs(z - med) <= cfg.depth_inlier_band_m
    return intr.deproject(us[good].astype(np.float64), vs[good].astype(np.float64), z[good])


def target_position(depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics,
                    cfg: LocalizationConfig) -> Optional[np.ndarray]:
    """``c_p``: 3-D centroid of the segmented cloud, or None if too few points."""
    cloud = segmented_point_cloud(depth_m, mask, intr, cfg)
    if len(cloud) < cfg.min_points:
        return None
    p = np.median(cloud, axis=0) if cfg.centroid == "median" else cloud.mean(axis=0)
    if cfg.mirror_x:
        p = p.copy()
        p[0] = -p[0]
    return p


def position_from_bbox(bbox: np.ndarray, intr: CameraIntrinsics, person_height_m: float = 1.7,
                       mirror_x: bool = False) -> np.ndarray:
    """RGB-only fallback (NOT in the paper): pinhole range from the box height.

    ``z = fy * H / h_px``; only for testing the pipeline on a webcam without
    depth. Partial occlusion or a crouching person corrupts it.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    z = intr.fy * person_height_m / max(1.0, y2 - y1)
    u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    p = intr.deproject(np.array([u]), np.array([v]), np.array([z]))[0]
    if mirror_x:
        p = p.copy()
        p[0] = -p[0]
    return p


def intrinsics_from_fov(width: int, height: int, hfov_deg: float = 69.0) -> CameraIntrinsics:
    """Approximate intrinsics from the horizontal field of view (webcams)."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return CameraIntrinsics(f, f, width / 2, height / 2, width, height)


def heading_angle(p_cam: np.ndarray) -> float:
    """``theta``: bearing of the target from the camera axis, positive to the
    LEFT (ROS convention, counter-clockwise seen from above)."""
    return math.atan2(-float(p_cam[0]), float(p_cam[2]))


class TargetKalmanFilter:
    """Constant-velocity KF on ``c_p`` with expiration time ``t_exp``.

    State ``[x, y, z, vx, vy, vz]`` in the camera frame. Time steps come from
    frame timestamps, so the filter is correct at any (varying) frame rate.
    """

    def __init__(self, cfg: LocalizationConfig) -> None:
        self.cfg = cfg
        self.x: Optional[np.ndarray] = None
        self.P = np.eye(6)
        self.t_state = 0.0
        self.t_measure = -math.inf
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.R = np.eye(3) * cfg.measurement_noise_m ** 2

    def reset(self) -> None:
        self.x = None
        self.t_measure = -math.inf

    def expired(self, now: float) -> bool:
        return self.x is None or (now - self.t_measure) > self.cfg.expiration_time_s

    def _predict_to(self, now: float) -> None:
        dt = max(0.0, now - self.t_state)
        if dt == 0.0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.cfg.process_noise_accel ** 2
        g = np.array([0.5 * dt * dt, dt])
        Q1 = q * np.outer(g, g)                 # white-acceleration model, per axis
        Q = np.zeros((6, 6))
        for i in range(3):
            idx = [i, i + 3]
            Q[np.ix_(idx, idx)] = Q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_state = now

    def update(self, p_cam: np.ndarray, now: float) -> np.ndarray:
        z = np.asarray(p_cam, np.float64).reshape(3)
        if self.expired(now):
            self.x = np.concatenate([z, np.zeros(3)])
            self.P = np.diag([self.cfg.measurement_noise_m ** 2] * 3 + [1.0] * 3)
            self.t_state = now
        else:
            self._predict_to(now)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = np.linalg.solve(S, self.H @ self.P).T
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
        self.t_measure = now
        return self.x[:3].copy()

    def estimate(self, now: float) -> Optional[np.ndarray]:
        """``c_p_filt`` at ``now``; None once ``t_exp`` elapsed without measurements."""
        if self.expired(now):
            return None
        self._predict_to(now)
        return self.x[:3].copy()


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Eq. 2: apply a 4x4 homogeneous transform to a 3-D point."""
    return (T[:3, :3] @ np.asarray(p, np.float64)) + T[:3, 3]


@dataclass
class Goal2D:
    x: float
    y: float
    yaw: float          # facing the target, from the camera position
    distance: float     # planar camera->target distance


def compute_goal(p_map: np.ndarray, camera_in_map: np.ndarray, standoff_m: float = 0.0) -> Goal2D:
    """Project ``m_p_filt`` on the map plane; orient the goal toward the target.

    ``standoff_m = 0`` reproduces the paper (goal on the target position; the
    safety circle stops the robot). A positive stand-off moves the goal back
    toward the robot along the line of sight.
    """
    dx = float(p_map[0] - camera_in_map[0])
    dy = float(p_map[1] - camera_in_map[1])
    dist = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    gx, gy = float(p_map[0]), float(p_map[1])
    if standoff_m > 0 and dist > 1e-6:
        back = min(standoff_m, dist)
        gx -= back * dx / dist
        gy -= back * dy / dist
    return Goal2D(gx, gy, yaw, dist)
