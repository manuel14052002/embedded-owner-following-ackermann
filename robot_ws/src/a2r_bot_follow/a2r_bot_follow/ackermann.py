#!/usr/bin/env python3
"""
ackermann.py - ONE kinematic model and ONE footprint for the whole follow stack
===============================================================================
The planner, the collision monitor and the escape manoeuvre used to carry
three private copies of "how does this car move and how big is it", and they
disagreed: the planner treated the car as a 0.195 m circle round the centre,
the monitor as a 0.45 x 0.33 m box, and both integrated the rear-axle arc as
if it were the centre's. The planner therefore kept choosing arcs the monitor
then refused, and the car sat still in front of things instead of avoiding
them. Everything now calls this module, so they can no longer disagree.

MEASURED GEOMETRY (hand-measured sheet, Sep 2026)
    tyre-to-tyre length 400 mm, tyre diameter 120 mm  -> wheelbase 280 mm
    tyre-to-tyre width  310 mm                        -> half width 155 mm
    base plate I 420 x 150 mm, centred                -> body x = +/-210 mm
    front sonar barrel protrudes to ~215 mm           -> front edge 215 mm
    max steer 30 deg (servo travel, unchanged)

    minimum turning radius, rear axle   L / tan(30)   = 0.485 m
                            base_link                 = 0.505 m
    outer front corner sweeps                         = 0.732 m
  so a full-lock U-turn needs about 1.46 m wall to wall.

MODEL
Kinematic bicycle referenced at base_link, which the URDF puts midway between
the axles (lr = L/2 ahead of the rear axle):

    beta   = atan(lr * tan(delta) / L)          slip angle of base_link
    dpsi   = cos(beta) * tan(delta) / L * ds    heading change per metre
    dx, dy = ds * (cos(psi + beta), sin(psi + beta))

ds is SIGNED arc length of base_link (negative = reversing), so reversing
with left lock correctly swings the heading clockwise. The steering angle may
ramp (servo slew) along the path instead of being assumed instant.
"""
import math

import numpy as np

# Measured values. Nodes take these as parameter defaults.
WHEELBASE_M = 0.280
TRACK_M = 0.250                 # 310 mm outside - 60 mm tyre width (tyre width not measured)
WHEEL_RADIUS_M = 0.060
MAX_STEER_DEG = 30.0
FOOT_FRONT_M = 0.215
FOOT_REAR_M = -0.210
FOOT_HALF_W_M = 0.155
LIDAR_X_M = -0.140              # rear of the upper deck (= URDF lidar_x, from the 09-25 photos); confirm with a tape measure
LIDAR_Y_M = 0.001               # 55 mm right / 53 mm left standoff gaps -> 1 mm left


class Ackermann:
    def __init__(self, wheelbase=WHEELBASE_M, max_steer=math.radians(MAX_STEER_DEG),
                 front=FOOT_FRONT_M, rear=FOOT_REAR_M, half_w=FOOT_HALF_W_M,
                 ref_from_rear=None):
        self.L = float(wheelbase)
        self.max_steer = float(max_steer)
        self.f, self.r, self.w = float(front), float(rear), float(half_w)
        # base_link distance ahead of the rear axle (URDF: mid-wheelbase)
        self.lr = 0.5 * self.L if ref_from_rear is None else float(ref_from_rear)

    # ── scalar helpers ─────────────────────────────────────────────────
    def slip(self, steer):
        return math.atan(self.lr * math.tan(steer) / self.L)

    def curvature(self, steer):
        """Heading change per metre travelled by base_link."""
        return math.cos(self.slip(steer)) * math.tan(steer) / self.L

    def radii(self, steer=None):
        """(rear axle, base_link, outer front corner) turning radii."""
        steer = self.max_steer if steer is None else abs(steer)
        if steer < 1e-6:
            return math.inf, math.inf, math.inf
        rr = self.L / math.tan(steer)
        rc = math.hypot(rr, self.lr)
        ro = math.hypot(rr + self.w, self.lr + self.f)
        return rr, rc, ro

    # ── path ───────────────────────────────────────────────────────────
    def poses(self, steer, dist, step, steer0=None, slew_per_m=0.0,
              x0=0.0, y0=0.0, yaw0=0.0):
        """Sample the path of base_link over signed distance `dist`.

        Returns (s, x, y, yaw) numpy arrays, s[0] = 0 at the start pose.
        With steer0 and slew_per_m (rad per metre travelled) the steering
        ramps from steer0 to `steer` instead of jumping there, which is what
        the servo physically does and why tight arcs start late."""
        n = max(1, int(math.ceil(abs(dist) / step)))
        ds = math.copysign(abs(dist) / n, dist if dist != 0 else 1.0)
        target = max(-self.max_steer, min(self.max_steer, float(steer)))
        if steer0 is None or slew_per_m <= 0.0:
            delta = np.full(n, target)
        else:
            d0 = max(-self.max_steer, min(self.max_steer, float(steer0)))
            travelled = (np.arange(n) + 0.5) * abs(ds)
            change = np.clip(target - d0, -slew_per_m * travelled, slew_per_m * travelled)
            delta = d0 + change
        tan_d = np.tan(delta)
        beta = np.arctan(self.lr * tan_d / self.L)
        kappa = np.cos(beta) * tan_d / self.L
        dpsi = kappa * ds
        psi_end = yaw0 + np.cumsum(dpsi)
        psi_mid = psi_end - 0.5 * dpsi                  # midpoint rule, 2nd order
        th = psi_mid + beta
        x = x0 + np.cumsum(ds * np.cos(th))
        y = y0 + np.cumsum(ds * np.sin(th))
        s = np.arange(0, n + 1) * ds
        return (s,
                np.concatenate(([x0], x)),
                np.concatenate(([y0], y)),
                np.concatenate(([yaw0], psi_end)))

    # ── footprint ──────────────────────────────────────────────────────
    def inside(self, pts, x, y, yaw, margin=0.0):
        """Boolean mask of points inside the (margin-grown) footprint at one pose."""
        if pts is None or len(pts) == 0:
            return np.zeros(0, dtype=bool)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        return (bx < self.f + margin) & (bx > self.r - margin) & (np.abs(by) < self.w + margin)

    def first_hit(self, pts, s, x, y, yaw, margin=0.0, chunk=16):
        """Signed-free distance |s| travelled before any obstacle point enters
        the footprint grown by `margin`. inf if the path is clear.

        A point ALREADY inside the margin band at the start pose only counts
        once it reaches the bare footprint. Without this, a car that came to
        rest 2 cm from a wall found every command - including reversing away
        from the wall - blocked at distance 0, and never moved again.

        a2r_merge: same answer, ~5-10x less work. The poses are tested in chunks of
        `chunk`, in order, each against only the points that can reach it (a point
        can only be inside the footprint if it lies within the footprint's
        circumradius of the pose centre), and the search stops at the first chunk
        with a hit. It used to build a (poses x points) matrix for the whole path:
        ~8 ms per call on the Pi, x 24-36 calls per plan cycle in a corridor."""
        if pts is None or len(pts) == 0 or len(s) == 0:
            return math.inf
        reach = abs(s[-1]) + max(self.f, -self.r) + self.w + margin + 0.05
        near = np.hypot(pts[:, 0] - x[0], pts[:, 1] - y[0]) < reach
        p = pts[near]
        if len(p) == 0:
            return math.inf
        mm = np.where(self.inside(p, x[0], y[0], yaw[0], margin), 0.0, margin)
        rc = math.hypot(max(self.f, -self.r), self.w) + margin + 1e-6
        px, py = p[:, 0], p[:, 1]
        n = len(s)
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            xs, ys = x[i0:i1], y[i0:i1]
            k = ((px > xs.min() - rc) & (px < xs.max() + rc)
                 & (py > ys.min() - rc) & (py < ys.max() + rc))
            if not k.any():
                continue
            qx, qy, m = px[k][None, :], py[k][None, :], mm[k][None, :]
            c, sn = np.cos(yaw[i0:i1])[:, None], np.sin(yaw[i0:i1])[:, None]
            dx = qx - xs[:, None]
            dy = qy - ys[:, None]
            bx = c * dx + sn * dy
            by = -sn * dx + c * dy
            hit = ((bx < self.f + m) & (bx > self.r - m) & (np.abs(by) < self.w + m)).any(axis=1)
            idx = np.flatnonzero(hit)
            if len(idx):
                return float(abs(s[i0 + idx[0]]))
        return math.inf

    def body_mask(self, pts, pad=0.01):
        """Points that are the robot itself (inside the footprint + pad) at the origin."""
        return self.inside(pts, 0.0, 0.0, 0.0, pad)
