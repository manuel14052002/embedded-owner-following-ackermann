"""Direct velocity following (``control_mode: cmd_vel``) - alternative to Nav2 goals.

The goal comes from the same pipeline (KF-filtered owner position, Eq. 2) but expressed
in the robot base frame, so ``goal.distance`` is the planar range and ``goal.yaw`` the
bearing of the owner. A proportional law with acceleration limits turns it into a Twist:

    v = clip(k_linear * (distance - follow_distance), 0, v_max) * max(0, cos(bearing))
    w = clip(k_angular * bearing, -w_max, w_max)

``/cmd_vel`` is then converted to Ackermann steering by the base driver (serial bridge).
There is no obstacle avoidance in this mode: the safety circle (state machine beta) and
the follow distance are the only protection. Use Nav2 mode where obstacles matter.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from followme.config import NavigationConfig
from followme.localization import Goal2D


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def apf_target(gx: float, gy: float, obstacles: np.ndarray, c: NavigationConfig) -> tuple[float, float]:
    """Artificial potential field (Khatib 1986): unit attraction toward the owner plus
    repulsion k*(1/d - 1/d0)/d^2 from every obstacle point within d0 (points near the owner
    are the owner and are ignored). Returns a virtual steering target at the look-ahead distance."""
    dist_goal = math.hypot(gx, gy)
    if dist_goal < 1e-6:
        return gx, gy
    fx, fy = gx / dist_goal, gy / dist_goal                  # attraction (unit)
    if obstacles is not None and len(obstacles):
        pts = np.asarray(obstacles, dtype=np.float64)
        far_from_owner = np.hypot(pts[:, 0] - gx, pts[:, 1] - gy) > c.apf_owner_exclusion_m
        d = np.hypot(pts[:, 0], pts[:, 1])
        m = far_from_owner & (d > 1e-3) & (d < c.apf_influence_m)
        if m.any():
            d_m, p_m = d[m], pts[m]
            mag = c.apf_k_rep * (1.0 / d_m - 1.0 / c.apf_influence_m) / d_m ** 2
            fx -= float(np.sum(mag * p_m[:, 0] / d_m))       # push away from each point
            fy -= float(np.sum(mag * p_m[:, 1] / d_m))
    n = math.hypot(fx, fy)
    if n < 1e-6:
        return gx, gy
    look = min(c.apf_lookahead_m, dist_goal)
    return look * fx / n, look * fy / n


class DirectFollowController:
    def __init__(self, cfg: NavigationConfig, safety_radius: float) -> None:
        self.cfg = cfg
        self.follow_distance = (cfg.follow_distance_m if cfg.follow_distance_m is not None
                                else safety_radius + 0.3)
        self.v = 0.0
        self.w = 0.0
        self._t: Optional[float] = None
        self.delta = 0.0
        self.holding = False
        self.apf_target = (0.0, 0.0)
        self._i = 0.0
        self._prev_alpha = None
        self._d_filt = 0.0
        self._owner_s = None
        self._trim = 0.0

    def reset(self) -> None:
        self.v = self.w = 0.0
        self._t = None
        self.delta = 0.0
        self.holding = False
        self._i = 0.0
        self._prev_alpha = None
        self._d_filt = 0.0
        self._owner_s = None
        self._trim = 0.0

    # obstacle points in base_link (N, 2) and their receive time, set by the ROS node
    obstacles: Optional[np.ndarray] = None
    obstacles_t: float = -1e9

    def command(self, goal: Goal2D, now: float) -> tuple[float, float]:
        c = self.cfg
        if c.steering_law == "pi_goal" and c.ackermann_wheelbase_m > 0:
            return self._pi_goal(goal, now)
        if c.steering_law in ("pure_pursuit", "pid") and c.ackermann_wheelbase_m > 0:
            return self._pure_pursuit(goal, now)
        bearing = goal.yaw
        v_des = clamp(c.k_linear * (goal.distance - self.follow_distance), 0.0, c.max_velocity_mps)
        if c.turn_slowdown:
            v_des *= max(0.0, math.cos(bearing))
        w_des = clamp(c.k_angular * bearing, -c.max_angular_rps, c.max_angular_rps)
        if (c.min_steer_speed_mps > 0 and goal.distance > self.follow_distance
                and abs(w_des) > 0.05):
            # steering only works while rolling: never let the slowdown stop us mid-turn
            v_des = max(v_des, c.min_steer_speed_mps)
        dt = 0.1 if self._t is None else clamp(now - self._t, 0.0, 0.5)
        self._t = now
        dv = v_des - self.v
        self.v += clamp(dv, -c.decel_mps2 * dt, c.accel_mps2 * dt)
        self.w = w_des
        return self.v, self.w

    def _pid(self, alpha: float, now: float, dmax: float) -> float:
        """PI(D) on the bearing. Integrates only while the car rolls (steering has no effect at
        standstill) and only if that does not push further into saturation (anti-windup)."""
        c = self.cfg
        dt = 0.1 if self._t is None else clamp(now - self._t, 0.0, 0.5)
        if self._prev_alpha is None:
            da = 0.0
        else:
            da = (alpha - self._prev_alpha) / dt if dt > 0 else 0.0
        self._prev_alpha = alpha
        self._d_filt = 0.7 * self._d_filt + 0.3 * da
        unclamped = c.pid_kp * alpha + c.pid_ki * self._i + c.pid_kd * self._d_filt
        saturated = abs(unclamped) >= dmax and math.copysign(1.0, unclamped) == math.copysign(1.0, alpha)
        if self.v > 0.0 and not saturated:
            self._i = clamp(self._i + alpha * dt, -c.pid_i_limit_rad, c.pid_i_limit_rad)
        return clamp(c.pid_kp * alpha + c.pid_ki * self._i + c.pid_kd * self._d_filt, -dmax, dmax)

    def _pure_pursuit(self, goal: Goal2D, now: float) -> tuple[float, float]:
        """Ackermann pure pursuit toward the owner, from the base_link origin (rear axle).

        kappa = 2 sin(alpha) / d ; delta = atan(L kappa), clamped to +/- delta_max ;
        omega = v tan(delta) / L  (the yaw rate the car will actually have at speed v).
        Owner behind the robot (|alpha| > 90 deg): full lock toward that side.
        """
        c = self.cfg
        L = c.ackermann_wheelbase_m
        dmax = math.radians(c.ackermann_max_steer_deg)
        tx, ty = goal.x, goal.y
        if c.apf_enabled and self.obstacles is not None and now - self.obstacles_t < 0.5:
            tx, ty = apf_target(goal.x, goal.y, self.obstacles, c)
        self.apf_target = (tx, ty)
        ox = oy = 0.0
        if c.pp_reference == "camera":
            # camera position in the goal's frame: goal point minus the camera->owner vector
            ox = goal.x - goal.distance * math.cos(goal.yaw)
            oy = goal.y - goal.distance * math.sin(goal.yaw)
        alpha = math.atan2(ty - oy, tx - ox)
        d = max(math.hypot(tx - ox, ty - oy), 1e-3)
        if c.steering_law == "pid":
            delta = self._pid(alpha, now, dmax)
        elif abs(alpha) > math.pi / 2:
            delta = math.copysign(dmax, alpha)
        else:
            delta = clamp(math.atan(L * 2.0 * math.sin(alpha) / d), -dmax, dmax)
        # hold hysteresis: stopped at the follow distance -> restart only beyond distance + margin
        if goal.distance <= self.follow_distance:
            self.holding = True
        elif goal.distance > self.follow_distance + c.hold_hysteresis_m:
            self.holding = False
        v_des = 0.0 if (self.holding and c.hold_hysteresis_m > 0) else \
            clamp(c.k_linear * (goal.distance - self.follow_distance), 0.0, c.max_velocity_mps)
        if (c.min_steer_speed_mps > 0 and goal.distance > self.follow_distance
                and not (self.holding and c.hold_hysteresis_m > 0)
                and abs(delta) > math.radians(1.0)):
            v_des = max(v_des, c.min_steer_speed_mps)          # this base only steers while rolling
        dt = 0.1 if self._t is None else clamp(now - self._t, 0.0, 0.5)
        self._t = now
        self.v += clamp(v_des - self.v, -c.decel_mps2 * dt, c.accel_mps2 * dt)
        # light steering filter + rate limit (smooth without the ~1 s lag of heavy smoothing)
        prev = getattr(self, "delta", 0.0)
        if c.steer_smoothing > 0:
            delta = (1.0 - c.steer_smoothing) * delta + c.steer_smoothing * prev
        if c.steer_rate_limit_dps > 0:
            step = math.radians(c.steer_rate_limit_dps) * dt
            delta = clamp(delta, prev - step, prev + step)
        self.delta = delta
        self.w = self.v * math.tan(delta) / L                    # physically achievable yaw rate
        return self.v, self.w

    def _pi_goal(self, goal: Goal2D, now: float) -> tuple[float, float]:
        """The Raspberry Pi follow pipeline's goal law (smart_planner._goal_cmd, v2.2), seen from
        the camera so the owner stays in view:
          goal distance  dg = owner distance - stand-off
          dg < tol                 -> hold
          |bearing| > turn bearing -> full-lock arc at turn speed (Ackermann cannot turn in place)
          v = min(vmax, k_lin * dg) * cos(bearing), never below min_drive_speed while moving
          |bearing| < deadband     -> straight (no jitter steering)
          w = k_ang * v * 2 sin(bearing) / max(dg, lookahead)
        then the steering angle is slew-limited (servo) and w is recomputed from it."""
        c = self.cfg
        L = c.ackermann_wheelbase_m
        dmax = math.radians(c.ackermann_max_steer_deg)
        dt = 0.1 if self._t is None else clamp(now - self._t, 0.0, 0.5)
        self._t = now
        ox = goal.x - goal.distance * math.cos(goal.yaw)        # camera position (see pure pursuit)
        oy = goal.y - goal.distance * math.sin(goal.yaw)
        tx, ty = goal.x, goal.y
        if c.apf_enabled and self.obstacles is not None and now - self.obstacles_t < 0.5:
            tx, ty = apf_target(goal.x, goal.y, self.obstacles, c)
        rx, ry = tx - ox, ty - oy
        a = c.owner_smooth_alpha
        prev = getattr(self, "_owner_s", None)
        if 0.0 < a < 1.0 and prev is not None:
            rx, ry = prev[0] + a * (rx - prev[0]), prev[1] + a * (ry - prev[1])
        self._owner_s = (rx, ry)
        d = math.hypot(rx, ry)
        b = math.atan2(ry, rx)
        dg = d - self.follow_distance
        if dg < c.goal_tol_m:
            v_des, delta_des = 0.0, 0.0
        elif abs(b) > math.radians(c.goal_turn_bearing_deg):
            v_des, delta_des = c.goal_turn_speed, math.copysign(dmax, b)
        else:
            v_des = min(c.goal_max_speed, c.goal_k_lin * dg) * max(0.0, math.cos(b))
            if v_des < 1e-3:
                v_des = 0.0
            elif c.min_drive_speed > 0:
                v_des = max(v_des, c.min_drive_speed)
            if v_des == 0.0 or abs(b) < math.radians(c.goal_bearing_deadband_deg):
                delta_des = 0.0
            else:
                kappa = c.goal_k_ang * 2.0 * math.sin(b) / max(dg, c.goal_pp_lookahead_m)
                delta_des = clamp(math.atan(L * kappa), -dmax, dmax)
        # slow trim integral: removes a constant offset (servo trim) without chasing noise
        trim = getattr(self, "_trim", 0.0)
        if c.goal_trim_ki > 0:
            if c.goal_trim_leak_s > 0:
                trim *= max(0.0, 1.0 - dt / c.goal_trim_leak_s)
            if self.v > 0 and abs(b) < math.radians(c.goal_trim_band_deg) and dg >= c.goal_tol_m:
                lim = math.radians(c.goal_trim_max_deg)
                trim = clamp(trim + c.goal_trim_ki * b * dt, -lim, lim)
            if v_des > 0:
                delta_des = clamp(delta_des + trim, -dmax, dmax)
        self._trim = trim
        self.v += clamp(v_des - self.v, -c.decel_mps2 * dt, c.accel_mps2 * dt)
        if v_des > 0 and c.min_drive_speed > 0 and 0 < self.v < c.min_drive_speed:
            self.v = c.min_drive_speed                         # start straight above the stall speed
        prev_d = getattr(self, "delta", 0.0)
        if c.steer_rate_limit_dps > 0:
            step = math.radians(c.steer_rate_limit_dps) * dt
            delta_des = clamp(delta_des, prev_d - step, prev_d + step)
        self.delta = delta_des
        self.w = self.v * math.tan(delta_des) / L
        return self.v, self.w
