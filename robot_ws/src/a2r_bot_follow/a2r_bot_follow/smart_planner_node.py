#!/usr/bin/env python3
"""
smart_planner_node.py                       (Raspberry Pi — a2r_bot_follow)
═══════════════════════════════════════════════════════════════════════════════
SMART VARIANT of local_planner_bridge_node.py.

  0) SHARED DRIVER-SKILL LAYER               (see "SHARED DRIVER-SKILL LAYER")
     Every skill below used to live inside _plan_dwa, so switching the
     PLANNER DISPATCH line silently dropped all of them and the other seven
     planners ran the plain 2024 logic. They now live in five shared methods
     — _situation, _bias, _commit_update, _finish, _score_arc — and each
     planner plugs into them THROUGH ITS OWN COMPUTATION:

       skill              DWA           APF             VFH           FGM
       ------------------------------------------------------------------
       cmd fade           score weight  K_att gain      target cost   aim blend
       early turn         prox term     forward probe   sector depth  gap depth
       anti-oscillation   commit filter bias force      VFH+ prev     aim hyst.
       escape exit        score bonus   exit force      sector bonus  aim shift
       VFH+T trap exit    score bonus   exit attractor  sector bonus  gap pref

     TEB uses homotopy-class hysteresis, Pure Pursuit an adaptive lookahead,
     Lattice its second segment. The seven still disagree about *how* to
     decide; they no longer disagree about how much they may know — which is
     what makes the comparison study fair.

     Shared and therefore identical for all eight: costmap + inflation, the
     arc-length rollout, the speed law, approach easing, the steering slew
     limit, the pinned-at-lock stall signal, trap and orbit detection, and
     the reverse-escape manager.

  A) HEADING MEMORY / RE-CENTRING
     Tracks how far the car's heading has drifted from the direction the
     Jetson asked for, and adds a gentle DWA term that prefers arcs which
     unwind that drift. After an S-shaped dodge around obstacles the car
     curves back onto the original bearing by itself. It is a bias, not a
     setpoint — clearance always wins, so it never feels like a global path.
     Yaw rate comes from the IMU when the IMU is publishing real data, and
     otherwise from the bicycle model  psi_dot = v / L * tan(delta)  applied
     to what we actually commanded.

  B) DRIVER-STYLE REVERSE
     Instead of backing up at whatever small angle happens to be cheapest,
     the car now pre-steers to near full lock BEFORE reversing (motor idle,
     servo swinging — exactly what a driver does), backs up on that lock,
     then leaves the pocket with opposite lock: a real three-point turn.
     Repeated escapes alternate sides and lengthen, instead of nibbling the
     same 20 cm twenty times.

  C) VFH+T TRAP DETECTION  (Neamah, Elek & Korondi 2024)
     _detect_trap() groups adjacent scan points (threshold 2*rr, paper 4.3),
     tests the largest group's azimuth span, and confirms hollowness with the
     concaveness criterion (eq. 51). It returns the exit bearing, which now
     biases ALL SEVEN planners via _bias(). A companion orbit detector
     catches the case the concavity test cannot see: a car circling inside a
     pocket at high steer never trips the stall detector because the planner
     keeps finding "valid" arcs.

The original file is untouched; switch between them in follow.launch.py.

OWNER-FOLLOWING SERIAL BRIDGE WITH SWITCHABLE LOCAL PATH PLANNERS
for a 1:10 Ackermann RC car (Tamiya Wild Dagger WR-01 class chassis).

Pipeline
────────
  Jetson Nano (camera_node.py → tracker_node.py, same ROS_DOMAIN_ID + WiFi)
        │ /cmd_vel  (geometry_msgs/Twist)
        │   • points the car toward the owner
        │   • zero / silence when owner is inside the follow threshold
        ▼
  THIS NODE (RPi)
        /scan (LiDAR) ─► rolling local costmap + footprint-based INFLATION LAYER
        LOCAL PLANNER  ─► collision-free (v, steering) — 7 algorithms, switch by
                          comment/uncomment at "PLANNER DISPATCH" below:
                            1. DWA            (ACTIVE)
                            2. APF            Artificial Potential Field
                            3. TEB-style      Timed-Elastic-Band path deformation
                            4. VFH            Vector Field Histogram
                            5. Pure Pursuit   geometric lookahead tracking
                            6. FGM            Follow-the-Gap Method
                            7. LATTICE        two-segment arc lattice (predictive)
        SPEED / STALL MANAGER
            • speed never collapses below `min_drive_speed` while a path exists
              (linear clearance scaling with a floor — NOT exponential decay)
            • stall detector + REVERSE-ESCAPE: if boxed in or stalled, the car
              backs out along the least-cost reverse arc, swinging its nose
              toward the open side, then replans forward
        ▼
  USB serial ─► Arduino Nano ESP32   {"cmd":"DRIVE","v":..,"s":..}\n   @ 20 Hz
               (PCA9685 steering servo · BTS7960 motor · MPU6050 IMU telemetry)

RViz topics (see follow.launch.py / follow.rviz)
────────────────────────────────────────────────
  /local_costmap                 OccupancyGrid  obstacles + inflation layer
  /local_planner/path            nav_msgs/Path  chosen trajectory
  /local_planner/trajectories    MarkerArray    candidate arcs (grey/red)
  /local_planner/best_traj       MarkerArray    winning arc (green)
  /local_planner/desired_cmd     Marker         Jetson's raw wish (blue arrow)
  /local_planner/state           Marker         FOLLOW / ESCAPE-REVERSE text

Safety
──────
  • zero desired Twist → STOP (owner at threshold — the Jetson decides this)
  • /cmd_vel silence > cmd_timeout_s → STOP (tracker lost owner / link down)
  • every trajectory lethal AND reverse blocked → STOP (truly boxed in)
  • explicit double-STOP on shutdown; dry_run:=true for bench testing
═══════════════════════════════════════════════════════════════════════════════
"""

import array
import heapq
import json
import math
import glob
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
try:                                                    # a2r_v7 edt
    from scipy.ndimage import distance_transform_edt as _a2r_edt
except Exception:                                       # no scipy: old loop
    _a2r_edt = None

try:
    from a2r_bot_follow.ackermann import (Ackermann, WHEELBASE_M, FOOT_FRONT_M,
                                          FOOT_REAR_M, FOOT_HALF_W_M)
except ImportError:                       # run straight from the source tree
    from ackermann import (Ackermann, WHEELBASE_M, FOOT_FRONT_M,
                           FOOT_REAR_M, FOOT_HALF_W_M)

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy, qos_profile_sensor_data)

from geometry_msgs.msg import Twist, Point, PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Path, Odometry      # a2r_pi_update2: Odometry
from sensor_msgs.msg import LaserScan, Imu, BatteryState
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray

try:
    import serial  # pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

_ESP32_HINTS = [
    ("303a", "1001"),   # Espressif native USB (Nano ESP32)
    ("2341", "0070"),   # Arduino VID for Nano ESP32
    ("10c4", "ea60"),   # CP2102
    ("1a86", "7523"),   # CH340
    ("0403", "6001"),   # FT232
]

COST_FREE      = 0
COST_INSCRIBED = 99
COST_LETHAL    = 100

# Planner state machine
ST_FOLLOW   = 0
ST_REVERSE  = 1
ST_PRESTEER = 2      # servo swinging to lock, motor idle, before reversing
ST_MANEUVER = 3      # a2r_pi_update2: planned multi-point turn (built in below)


# ══════════════════════════════════════════════════════════════════════
#  MULTI-POINT TURNS (a2r_merge: was maneuver.py, now part of this node)
# ══════════════════════════════════════════════════════════════════════
# Pure pursuit drives forward only and the stall escape is ONE reverse arc, so in a
# corridor narrower than a full-lock U-turn (~1.46 m wall to wall) or with the owner
# behind the car, the car could never turn round. This is the K-turn a driver makes:
# a small hybrid-A* over (x, y, heading) with forward/reverse x left/straight/right
# primitives, on the scan the collision monitor sees, with
#   * the SAME kinematic model and footprint as the planner and the monitor (ackermann.py);
#   * every primitive also checked stop_buffer metres FURTHER along its arc (the monitor
#     STOPs the car ~0.2 m before an obstacle at crawl speed);
#   * obstacles already inside the margin at the start pose only counting against the
#     bare body (same rule as Ackermann.first_hit), so a car parked at a wall can leave;
#   * a2r_merge, OWNER-AWARE REVERSING: reverse travel is capped per leg (max_rev_leg_m)
#     and per plan (max_reverse_m); the owner (lidar leg track / phone / goal) has a keep
#     zone the plan never enters; a reverse step that moves AWAY from the owner costs
#     extra. The old planner backed 0.8-2.1 m, right up to an owner standing behind;
#   * best effort: no full route within the budget -> the partial plan that turned the
#     car furthest toward the goal; the car drives it and re-plans with a fresh scan.
# Goals (base_link of the start pose):
#   drive : (gx, gy) = the follow goal. Done when within ang_tol of straight ahead AND
#           the car could drive toward it for free_m.
#   face  : (gx, gy) = where the owner stands. Done when within ang_tol of straight ahead.

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ManeuverGoal:
    mode: str                 # "drive" | "face"
    gx: float
    gy: float
    ang_tol: float = math.radians(30.0)
    free_m: float = 0.6       # drive: clear run toward the goal needed at the end
    ox: Optional[float] = None    # owner position (base_link of the start pose), if known:
    oy: Optional[float] = None    #   the plan keeps clear of them and weighs reverse against them


@dataclass
class Plan:
    prims: List[Tuple[int, float, float]]           # (dir +1/-1, steer rad, length m)
    poses: np.ndarray                               # (n, 3) x, y, yaw along the plan
    expanded: int = 0
    ms: float = 0.0
    legs: List[List[Tuple[int, float, float]]] = field(default_factory=list)
    partial: bool = False                           # best effort: turned toward the goal, not there yet
    gain_deg: float = 0.0                           # partial: heading error removed (deg)

    def summary(self) -> str:
        parts = []
        for leg in self.legs:
            d = leg[0][0]
            parts.append(("F" if d > 0 else "R") + f"{sum(p[2] for p in leg):.2f}")
        return " ".join(parts)


def split_legs(prims):
    legs, cur = [], []
    for p in prims:
        if cur and p[0] != cur[-1][0]:
            legs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        legs.append(cur)
    return legs


def merge_prims(prims):
    """Consecutive primitives with the same direction and steering -> one segment."""
    out = []
    for d, s, ln in prims:
        if out and out[-1][0] == d and abs(out[-1][1] - s) < 1e-9:
            out[-1] = (d, s, out[-1][2] + ln)
        else:
            out.append((d, s, ln))
    return out


class ManeuverPlanner:
    def __init__(self, car, margin: float = 0.04, res: float = 0.04, step: float = 0.20,
                 sample: float = 0.05, stop_buffer: float = 0.30, area: float = 2.2,
                 xy_bin: float = 0.08, yaw_bin_deg: float = 10.0, reverse_w: float = 1.5,
                 switch_w: float = 0.6, steer_w: float = 0.05, max_expand: int = 3000,
                 time_budget_s: float = 0.6, verify_margin: float = 0.03,
                 half_steer: bool = False, h_weight: float = 1.5,
                 partial_min_gain_deg: float = 20.0, max_reverse_m: float = 1.2, max_rev_leg_m: float = 0.5, min_leg_m: float = 0.2,
                 owner_keep_m: float = 0.75, owner_w: float = 2.0, away_w: float = 0.8,
                 rev_leg_k: float = 1.0, rev_total_k: float = 1.5):
        self.car = car
        # a2r_step3: the reverse budget grows with the turn still needed (see plan())
        self.rev_leg_k, self.rev_total_k, self.last_leg_cap = rev_leg_k, rev_total_k, max_rev_leg_m
        self.margin, self.res, self.step, self.sample = margin, res, step, sample
        self.buf, self.area = stop_buffer, area
        self.xy_bin, self.yaw_bin = xy_bin, math.radians(yaw_bin_deg)
        self.rev_w, self.sw_w, self.st_w = reverse_w, switch_w, steer_w
        self.max_expand, self.budget = max_expand, time_budget_s
        self.verify_margin = verify_margin
        self.h_w = h_weight                               # weighted A*: >1 = faster, slightly longer plans
        self.partial_gain = math.radians(partial_min_gain_deg)   # 0 or less = no partial plans
        self.last_why = ""
        # owner-aware reversing (a2r_merge): the K-turn used to back 0.8-2.0 m, up to the owner
        self.max_rev = max_reverse_m          # total reverse travel allowed in one plan (<= 0 = no cap)
        self.max_rev_leg = max_rev_leg_m      # ... and in any ONE reverse leg (<= 0 = no cap)
        self.min_leg = min_leg_m              # no direction change before a leg is this long: every
                                              # change costs ~1 s on the car (pre-steer, ESC brake/neutral)
        self.owner_keep = owner_keep_m        # base_link never closer to the owner than this
        self.owner_w = owner_w                # cost per metre of keep-zone intrusion ... (soft, 1.5x keep)
        self.away_w = away_w                  # cost per metre a REVERSE step moves away from the owner
        ms = car.max_steer
        steers = [-ms, 0.0, ms] if not half_steer else [-ms, -0.5 * ms, 0.0, 0.5 * ms, ms]
        self.prim_defs = [(d, s) for d in (1, -1) for s in steers]
        # relative samples of each primitive: the move itself (exactly `step`, ending on its last
        # sample, which is the child pose) and then the stop-buffer extension on the same arc
        rel = []
        n_move = max(1, int(round(step / sample)))
        n_ext = int(round(stop_buffer / sample)) if stop_buffer > 1e-9 else 0
        for d, s in self.prim_defs:
            _, x, y, yaw = car.poses(s, d * step, step / n_move)
            mv = np.stack([x[1:], y[1:], yaw[1:]], axis=1)
            if n_ext:
                _, xe, ye, ye_w = car.poses(s, d * stop_buffer, stop_buffer / n_ext,
                                            x0=float(x[-1]), y0=float(y[-1]), yaw0=float(yaw[-1]))
                mv = np.concatenate([mv, np.stack([xe[1:], ye[1:], ye_w[1:]], axis=1)])
            rel.append(mv)
        self.rel = np.stack(rel)                          # (P, K, 3)
        self.k_step = n_move                              # rel[:, k_step-1] = the child pose
        self.r_c = car.radii()[1]                         # base_link turning radius at full lock
        self.fp_r = math.hypot(max(car.f, -car.r), car.w) + margin
        self.reach = step + stop_buffer + self.fp_r + 0.05

    # ── obstacles: exact footprint test (same rule as Ackermann.first_hit) ──
    def _build(self, pts: np.ndarray, start=(0.0, 0.0, 0.0)):
        car, m = self.car, self.margin
        P = np.zeros((0, 2)) if pts is None or len(pts) == 0 else np.asarray(pts, dtype=float)
        P = P[(np.abs(P[:, 0] - start[0]) < self.area) & (np.abs(P[:, 1] - start[1]) < self.area)]
        # thin to one point per 1 cm square (bounds the cost; the search margin is 1 cm wider
        # than the verify margin, so thinning can never let a too-close plan through)
        if len(P):
            q = np.floor(P / 0.01).astype(np.int64)
            _, keep = np.unique(q[:, 0] * 1000003 + q[:, 1], return_index=True)
            P = P[np.sort(keep)]
        c, s = math.cos(start[2]), math.sin(start[2])
        dx, dy = P[:, 0] - start[0], P[:, 1] - start[1]
        bx, by = c * dx + s * dy, -s * dx + c * dy
        in_bare = (bx < car.f) & (bx > car.r) & (np.abs(by) < car.w)
        in_band = (bx < car.f + m) & (bx > car.r - m) & (np.abs(by) < car.w + m)
        # inside the bare body at the start = the car itself / a ghost: never an obstacle
        self.ghost = int(in_bare.sum())
        P, in_band = P[~in_bare], in_band[~in_bare]
        o = np.argsort(P[:, 0], kind="stable")             # a2r_fix_freeze: sorted by x -> _near slices
        P, in_band = P[o], in_band[o]
        self.P = P
        self.Px = np.ascontiguousarray(P[:, 0])
        # already inside the margin band at the start: counts only against the bare body
        self.Pm = np.where(in_band, 0.0, m)
        self.start = start
        # a2r_fix_freeze: homogeneous points (3, N) + per-point bounds for the matrix test
        self.Pt = np.vstack([P[:, 0], P[:, 1], np.ones(len(P))]) if len(P) else np.zeros((3, 0))
        self.Bf = car.f + self.Pm
        self.Br = car.r - self.Pm
        self.Bw = car.w + self.Pm

    def _near(self, x, y, reach=None):
        r = self.reach if reach is None else reach
        if len(self.P) == 0:
            return self.Pt, self.Bf, self.Br, self.Bw
        i0, i1 = np.searchsorted(self.Px, (x - r, x + r))
        k = np.abs(self.P[i0:i1, 1] - y) < r
        return self.Pt[:, i0:i1][:, k], self.Bf[i0:i1][k], self.Br[i0:i1][k], self.Bw[i0:i1][k]

    def collide(self, X, Y, TH, near=None) -> np.ndarray:
        """Boolean per pose: the (margin-grown) footprint holds an obstacle point, or the pose
        leaves the known area (unknown space is not free space)."""
        X, Y, TH = np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(TH)
        lim = self.area - self.fp_r
        out = (np.abs(X - self.start[0]) > lim) | (np.abs(Y - self.start[1]) > lim)
        if near is None:
            cx, cy = float(np.mean(X)), float(np.mean(Y))
            half = float(max(np.max(np.abs(X - cx)), np.max(np.abs(Y - cy))))
            near = self._near(cx, cy, half + self.fp_r + 0.02)
        Qt, Bf, Br, Bw = near
        if Qt.shape[1] == 0:
            return out
        # body frame of every pose for every point in ONE matrix product:
        #   bx =  c*px + s*py - (c*x + s*y),   by = -s*px + c*py + (s*x - c*y)
        n = len(X)
        c, s = np.cos(TH), np.sin(TH)
        A = np.empty((2 * n, 3))
        A[:n, 0], A[:n, 1], A[:n, 2] = c, s, -(c * X + s * Y)
        A[n:, 0], A[n:, 1], A[n:, 2] = -s, c, s * X - c * Y
        B = A @ Qt
        bx = B[:n]
        hit = bx < Bf
        hit &= bx > Br
        hit &= np.abs(B[n:]) < Bw
        return out | hit.any(axis=1)

    # ── goal ────────────────────────────────────────────────────────────
    def _err(self, goal: ManeuverGoal, x, y, th):
        return _wrap(math.atan2(goal.gy - y, goal.gx - x) - th)

    def _h(self, goal: ManeuverGoal, x, y, th):
        e = abs(self._err(goal, x, y, th))
        h = self.r_c * max(0.0, e - goal.ang_tol)      # heading change needs >= R * angle of travel
        if goal.mode == "drive":
            h += 0.2 * max(0.0, math.hypot(goal.gx - x, goal.gy - y) - 1.0)
        return h

    def _forward_free(self, goal: ManeuverGoal, x, y, th) -> bool:
        d = math.hypot(goal.gx - x, goal.gy - y)
        e = self._err(goal, x, y, th)
        L = self.car.L
        steer = math.atan(L * 2.0 * math.sin(e) / max(d, 0.6))
        steer = max(-self.car.max_steer, min(self.car.max_steer, steer))
        run = min(d, goal.free_m) + self.buf
        _, X, Y, TH = self.car.poses(steer, run, self.sample, x0=x, y0=y, yaw0=th)
        return not self.collide(X[1:], Y[1:], TH[1:]).any()

    def is_goal(self, goal: ManeuverGoal, x, y, th) -> bool:
        if abs(self._err(goal, x, y, th)) > goal.ang_tol:
            return False
        if goal.mode == "face":
            return True
        return self._forward_free(goal, x, y, th)

    # ── search ──────────────────────────────────────────────────────────
    def plan(self, pts: np.ndarray, goal: ManeuverGoal, start=(0.0, 0.0, 0.0),
             verify_pts: Optional[np.ndarray] = None) -> Optional[Plan]:
        t0 = time.perf_counter()
        self._build(pts, start)
        sx, sy, sth = start
        if self.is_goal(goal, sx, sy, sth):
            return Plan([], np.array([[sx, sy, sth]]), 0, 1e3 * (time.perf_counter() - t0))
        nodes = [(sx, sy, sth, 0, 0.0, -1, None, 0.0, 0.0)]  # x, y, th, dir, steer, parent, prim,
                                                             # reverse m in total, length of this leg
        has_o = goal.ox is not None and goal.oy is not None
        ox, oy = (goal.ox, goal.oy) if has_o else (0.0, 0.0)
        # never plan INTO the owner's keep zone, but a car that already stands inside it may leave it
        keep = min(self.owner_keep, math.hypot(ox - sx, oy - sy) - 0.05) if has_o else 0.0
        soft = 1.5 * self.owner_keep
        # a2r_step3: reverse PROPORTIONAL to the turn still needed. A full-lock turn removes heading
        # at 1/R per metre, so the arc still to drive is R * (heading error - tolerance). The fixed
        # 0.5 m / 1.2 m caps cut a U-turn into 4-6 short shuffles; now one reverse may be as long as
        # the turn needs (U-turn ~1.3 m), a 100 deg turn still only ~0.6 m.
        need = self.r_c * max(0.0, abs(self._err(goal, sx, sy, sth)) - goal.ang_tol)
        rev_cap = max(self.max_rev, self.rev_total_k * need) if self.max_rev > 0.0 else math.inf
        leg_cap = max(self.max_rev_leg, self.rev_leg_k * need) if self.max_rev_leg > 0.0 else math.inf
        self.last_leg_cap = leg_cap
        key0 = self._key(sx, sy, sth, 0)
        best_g = {key0: 0.0}
        heap = [(self.h_w * self._h(goal, sx, sy, sth), 0.0, 0)]
        self.last_why = ""
        # best effort: the node that turned the car furthest toward the goal
        tol = goal.ang_tol
        e0 = max(0.0, abs(self._err(goal, sx, sy, sth)) - tol)
        bst_e, bst_g, bst_i = e0, 0.0, 0
        P, K = self.rel.shape[0], self.rel.shape[1]
        rel = self.rel
        expanded, found = 0, None
        while heap:
            f, gcost, idx = heapq.heappop(heap)
            x, y, th, d0, s0, _, _, rv0, ll0 = nodes[idx]
            if gcost > best_g.get(self._key(x, y, th, d0), math.inf) + 1e-9:
                continue
            expanded += 1
            if expanded > self.max_expand:
                self.last_why = f"node cap ({self.max_expand}) reached"
                break
            if expanded % 16 == 0 and time.perf_counter() - t0 > self.budget:
                self.last_why = f"no route within {self.budget:.1f} s ({expanded} nodes)"
                break
            c, s = math.cos(th), math.sin(th)
            X = x + c * rel[:, :, 0] - s * rel[:, :, 1]
            Y = y + s * rel[:, :, 0] + c * rel[:, :, 1]
            TH = th + rel[:, :, 2]
            hit = self.collide(X.ravel(), Y.ravel(), TH.ravel(), self._near(x, y)).reshape(P, K)
            free = ~hit.any(axis=1)                      # move + stop buffer clear
            for p in np.flatnonzero(free):
                d, st = self.prim_defs[p]
                k = self.k_step - 1
                nx, ny, nth = float(X[p, k]), float(Y[p, k]), float(TH[p, k])
                cost = self.step * (1.0 if d > 0 else self.rev_w)
                rv = rv0 + (self.step if d < 0 else 0.0)
                if d0 != 0 and d != d0 and ll0 < self.min_leg - 1e-9:
                    continue                             # a leg shorter than min_leg: not worth a stop
                ll = (ll0 + self.step) if d == d0 else self.step
                if rv > rev_cap + 1e-9 or (d < 0 and ll > leg_cap + 1e-9):
                    continue                             # reverse budget spent: turn, do not back away
                if has_o:
                    do = math.hypot(ox - nx, oy - ny)
                    if do < keep:
                        continue                         # into the owner's keep zone
                    if do < soft:
                        cost += self.owner_w * self.step * (soft - do) / soft
                    if d < 0:                            # a reverse that moves AWAY from the owner costs
                        cost += self.away_w * max(0.0, do - math.hypot(ox - x, oy - y))
                if d0 != 0 and d != d0:
                    cost += self.sw_w
                if d0 != 0 and abs(st - s0) > 1e-9:
                    cost += self.st_w
                ng = gcost + cost
                key = self._key(nx, ny, nth, d)
                if ng >= best_g.get(key, math.inf) - 1e-9:
                    continue
                best_g[key] = ng
                nodes.append((nx, ny, nth, d, st, idx, (d, st, self.step), rv, ll))
                ni = len(nodes) - 1
                if self.is_goal(goal, nx, ny, nth):
                    found = ni
                    break
                e = max(0.0, abs(self._err(goal, nx, ny, nth)) - tol)
                # 5 deg bins: among poses about as well turned, the cheapest path wins
                if (round(e / 0.087), ng) < (round(bst_e / 0.087), bst_g):
                    bst_e, bst_g, bst_i = e, ng, ni
                heapq.heappush(heap, (ng + self.h_w * self._h(goal, nx, ny, nth), ng, ni))
            if found is not None:
                break
        ms = 1e3 * (time.perf_counter() - t0)
        partial = False
        if found is None:
            if not self.last_why:
                self.last_why = f"no route exists ({expanded} nodes searched)"
            if self.partial_gain <= 0.0 or bst_i == 0 or e0 - bst_e < self.partial_gain:
                return None
            found, partial = bst_i, True
        prims, poses, i = [], [], found
        while i > 0:
            n = nodes[i]
            prims.append(n[6])
            poses.append(n[:3])
            i = n[5]
        prims.reverse()
        poses.append((sx, sy, sth))
        poses.reverse()
        prims = merge_prims(prims)
        plan = Plan(prims, np.array(poses), expanded, ms, split_legs(prims),
                    partial, math.degrees(e0 - bst_e) if partial else 0.0)
        if verify_pts is not None and not self.verify(plan, verify_pts, start):
            self.last_why = "plan failed the exact footprint check"
            return None
        return plan

    def _key(self, x, y, th, d):
        return (int(round(x / self.xy_bin)), int(round(y / self.xy_bin)),
                int(round(_wrap(th) / self.yaw_bin)) % int(round(2 * math.pi / self.yaw_bin)), d)

    def verify(self, plan: Plan, pts: np.ndarray, start=(0.0, 0.0, 0.0)) -> bool:
        """Exact check (Ackermann.first_hit, monitor margin) of every segment + stop buffer."""
        x, y, th = start
        for d, st, ln in plan.prims:
            s, X, Y, TH = self.car.poses(st, d * (ln + self.buf), 0.02, x0=x, y0=y, yaw0=th)
            if math.isfinite(self.car.first_hit(pts, s, X, Y, TH, self.verify_margin)):
                return False
            s, X, Y, TH = self.car.poses(st, d * ln, 0.02, x0=x, y0=y, yaw0=th)
            x, y, th = float(X[-1]), float(Y[-1]), float(TH[-1])
        return True


def segment_poses(car, prims, start=(0.0, 0.0, 0.0), step=0.02):
    """Poses along a primitive list from `start` (for display / remaining-path checks)."""
    x, y, th = start
    out = [(x, y, th)]
    for d, st, ln in prims:
        _, X, Y, TH = car.poses(st, d * ln, step, x0=x, y0=y, yaw0=th)
        out.extend(zip(X[1:].tolist(), Y[1:].tolist(), TH[1:].tolist()))
        x, y, th = float(X[-1]), float(Y[-1]), float(TH[-1])
    return np.array(out)


# ══════════════════════════════════════════════════════════════════════
#  v2.8 helpers (pure, testable without ROS)
# ══════════════════════════════════════════════════════════════════════
def voxel_thin(pts: np.ndarray, cell: float) -> np.ndarray:
    """One point per `cell`-sized square, the first one met (an actual
    measurement, never a cell centre). With the lidar at ~0.16 deg a wall
    1 m away returns a point every 3 mm; the footprint sweep only needs one
    every couple of centimetres, and its cost grows with the point count.
    A 2-3 cm chair leg still keeps its point."""
    if pts is None or len(pts) == 0 or cell <= 0.0:
        return pts
    k = np.floor(pts / cell).astype(np.int64)
    key = k[:, 0] * 1_000_003 + k[:, 1]
    _, first = np.unique(key, return_index=True)
    return pts[np.sort(first)]


def detect_trap_core(ang, rng, real, group_dist=0.40, span_min=math.radians(160.0),
                     concave_frac=0.70, max_range=2.0, far_range=1.4):
    """VFH+T internal-trap test (Neamah, Elek & Korondi 2024, sec. 4.3 + eq. 51).

    ang, rng : every beam, base_link polar, in scan order
    real     : True where the beam is a real return (not "no return", not the body)

    v2.8 fixes. The old loop grouped ALL beams, and "no return" beams had been
    set to scan_max, so open space became a ring of fake points 3.5 m out,
    6 cm apart: one huge "obstacle" round the car. In an ordinary room the
    largest group spanned ~360 deg, passed the concavity test, and a "trap"
    was reported nearly every cycle - which threw away the Jetson-first
    avoidance, handed the car to the planner with a 2.5x pull toward the
    "exit", and fed the stall / reverse logic. Now:
      * only real returns within max_range form groups (a pocket is near);
      * consecutive POINTS (drop-outs skipped) link if < group_dist apart;
      * the span is unwrapped (a group across the +-180 deg seam used to
        measure ~0 or ~360 deg);
      * vectorised - it ran a Python loop over every beam, now ~2500 of them.
    Returns the exit bearing (rad, base_link) or None."""
    ang = np.asarray(ang, dtype=np.float64)
    rng = np.asarray(rng, dtype=np.float64)
    real = np.asarray(real, dtype=bool)
    n = len(ang)
    if n < 20:
        return None
    obs = np.flatnonzero(real & (rng < max_range))
    if len(obs) < 5:
        return None
    x, y = rng[obs] * np.cos(ang[obs]), rng[obs] * np.sin(ang[obs])
    link = np.hypot(np.diff(x), np.diff(y)) <= group_dist
    cuts = np.flatnonzero(~link) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts - 1, len(obs) - 1]
    groups = [np.arange(s, e + 1) for s, e in zip(starts, ends)]
    # 360-degree seam: the last group continues into the first
    if len(groups) > 1 and math.hypot(x[0] - x[-1], y[0] - y[-1]) <= group_dist:
        groups[0] = np.r_[groups[-1], groups[0]]
        groups.pop()
    elif len(groups) == 1 and math.hypot(x[0] - x[-1], y[0] - y[-1]) <= group_dist:
        return None          # one closed ring round the car: boxed in, not a pocket with an exit
    best, best_span = None, 0.0
    for g in groups:
        if len(g) < 5:
            continue
        a = ang[obs[g]]
        span = float(np.sum(np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)))
        if span > best_span:
            best, best_span = g, span
    if best is None or best_span < span_min:
        return None

    # ── concaveness: is the car INSIDE the hollow? ──
    # eq. 51 measures the group against the chord between its two ends, ray
    # by ray. With the car inside a pocket that chord is BEHIND it, the rays
    # ahead never cross it, and the test only "passed" before because the
    # fake far ring sat at the mouth. The geometric meaning is kept: close
    # the group with its chord and ask whether the car is inside that
    # outline (even-odd rule), and whether the outline is deep enough round
    # the car (concave_frac of its points beyond the car, seen from the chord).
    gx, gy = x[best], y[best]
    px, py = np.r_[gx, gx[0]], np.r_[gy, gy[0]]          # closed by the chord
    x0, y0, x1, y1 = px[:-1], py[:-1], px[1:], py[1:]
    crosses = ((y0 > 0.0) != (y1 > 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (0.0 - y0) * (x1 - x0) / np.where(crosses, y1 - y0, 1.0)
    inside = int(np.sum(crosses & (xc > 0.0))) % 2 == 1
    if not inside:
        return None
    ex, ey = gx[-1] - gx[0], gy[-1] - gy[0]
    el = math.hypot(ex, ey)
    if el > 1e-6:
        nx, ny = -ey / el, ex / el
        s0 = -(gx[0] * nx + gy[0] * ny)                       # the car, from the chord
        if s0 < 0.0:
            nx, ny, s0 = -nx, -ny, -s0
        deeper = ((gx - gx[0]) * nx + (gy - gy[0]) * ny) > s0 + 0.10
        if float(np.mean(deeper)) < 0.5 * concave_frac:
            return None

    # ── exit: open bearings outside the group's angular extent ──
    in_grp = np.zeros(n, dtype=bool)
    i0, i1 = int(obs[best[0]]), int(obs[best[-1]])
    if i0 <= i1:
        in_grp[i0:i1 + 1] = True
    else:                                     # wraps the seam
        in_grp[i0:] = True
        in_grp[:i1 + 1] = True
    far = ~in_grp & (~real | (rng > far_range))
    if not far.any():
        far = ~in_grp
    if not far.any():
        return None
    return float(math.atan2(float(np.mean(np.sin(ang[far]))), float(np.mean(np.cos(ang[far])))))


class OutputSmoother:
    """v2.8: 50 Hz rate limiter between the 5 Hz planner and the motors.

    The planner changes (v, steer) in steps every 200 ms - 0 -> 0.168 -> 0.24
    m/s, or 18 deg of steering at once - and the send thread held each step
    for ten ticks. That staircase is the jerk you feel. Here speed ramps at
    accel/decel limits and the steering slews, every 20 ms.
      * Starting from rest jumps straight to `start_frac` of the stall floor:
        ramping through the ESC dead band only adds delay, then a lurch.
      * Below the stall floor on the way down -> 0 at once (no creeping).
      * A reversal of direction decelerates through zero first.
    Units are the normalised drive command (fractions)."""

    def __init__(self, accel=0.8 / 1.2, decel=1.5 / 1.2, steer_rate=2.5,
                 floor=0.14, start_frac=0.8):
        self.accel, self.decel, self.steer_rate = accel, decel, steer_rate
        self.floor, self.start_frac = floor, start_frac
        self.v = self.s = 0.0
        self.t = None

    def step(self, v_t: float, s_t: float, now: float):
        dt = 0.02 if self.t is None else max(0.0, min(0.1, now - self.t))
        self.t = now
        v = self.v
        if v != 0.0 and (v_t == 0.0 or math.copysign(1.0, v_t) != math.copysign(1.0, v)):
            v = math.copysign(max(0.0, abs(v) - self.decel * dt), v)     # slow down / through zero
            if abs(v) < self.floor:
                v = 0.0
        elif abs(v_t) > abs(v):
            if v == 0.0:
                v = math.copysign(min(abs(v_t), self.start_frac * self.floor), v_t)
            v = math.copysign(min(abs(v_t), abs(v) + self.accel * dt), v_t)
        elif abs(v_t) < abs(v):
            v = math.copysign(max(abs(v_t), abs(v) - self.decel * dt), v_t)
        self.v = v
        ds = s_t - self.s
        lim = self.steer_rate * dt
        self.s += max(-lim, min(lim, ds))
        return self.v, self.s


class LocalPlannerBridgeNode(Node):

    # ══════════════════════════════════════════════════════════════════
    #  INIT
    # ══════════════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__("local_planner_bridge")

        # ── Serial / firmware ─────────────────────────────────────────
        self.declare_parameter("port", "/dev/ttyARDUINO")   # "auto" → VID/PID scan
        self.declare_parameter("baud", 921600)
        self.declare_parameter("dry_run", False)

        # ── Ackermann kinematics (measured: 31 cm wide, 28.0 cm wheelbase) ─
        self.declare_parameter("max_speed_mps",       1.2)
        self.declare_parameter("max_speed_cmd",       0.7)    # firmware throttle cap
        self.declare_parameter("wheelbase_m",         WHEELBASE_M)
        self.declare_parameter("max_steer_deg",       30.0)
        self.declare_parameter("min_speed_for_steer", 0.15)

        # ── Speed management (anti-stall) ─────────────────────────────
        self.declare_parameter("min_drive_speed",  0.28)  # m/s floor while a path exists
        self.declare_parameter("clearance_gain",   0.6)   # v = des·(0.4 + 0.6·clear)
        self.declare_parameter("stall_time_s",     1.5)   # stalled this long → escape
        self.declare_parameter("reverse_speed",    0.35)  # m/s during escape
        self.declare_parameter("reverse_time_s",   1.4)   # escape manoeuvre duration
        self.declare_parameter("reverse_cooldown_s", 2.0) # min gap between escapes
        self.declare_parameter("esc_reverse_delay_s", 0.7)  # fw brake 350 ms + neutral 350 ms

        # ── Rates / watchdog ──────────────────────────────────────────
        self.declare_parameter("cmd_rate_hz",   20.0)
        self.declare_parameter("cmd_timeout_s", 0.6)
        self.declare_parameter("plan_rate_hz",  10.0)

        # ── Topics ────────────────────────────────────────────────────
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic",    "/scan")

        # ── Costmap + inflation (footprint-based) ─────────────────────
        # Footprint: 0.31 m wide → inscribed (lethal) radius = half-width
        # + 4 cm safety = 0.195 m. Inflation extends a modest 0.25 m past
        # that so corridors ≥ ~0.9 m stay traversable — "reasonable", not
        # paralysing.
        self.declare_parameter("costmap_size_m",       4.0)
        self.declare_parameter("costmap_res_m",        0.05)
        self.declare_parameter("robot_radius_m",       0.195)
        self.declare_parameter("inflation_radius_m",   0.45)
        self.declare_parameter("inflation_cost_scale", 5.0)
        self.declare_parameter("costmap_frame",        "base_link")
        self.declare_parameter("lidar_offset_x",       0.0)
        self.declare_parameter("lidar_offset_y",       0.0)
        self.declare_parameter("lidar_yaw_offset",     0.0)   # A2M12 on this mount: 0.0 (set in params.yaml)
        self.declare_parameter("scan_max_range",       3.5)
        self.declare_parameter("scan_min_range",       0.15)
        # Footprint (base_link, metres). The rollout sweeps THIS rectangle along
        # the exact bicycle path, the same way the collision_monitor does, so
        # the planner no longer picks arcs the monitor will then veto.
        self.declare_parameter("footprint_front_m",    FOOT_FRONT_M)
        self.declare_parameter("footprint_rear_m",     FOOT_REAR_M)
        self.declare_parameter("footprint_half_width_m", FOOT_HALF_W_M)
        self.declare_parameter("footprint_margin_m",   0.04)   # > monitor's 0.03
        # A footprint hit closer than max(hard_min_m, hard_time_s * v) along an
        # arc makes it LETHAL. A hit further away only shortens clear_dist, so
        # the planner keeps creeping toward the gap instead of freezing 1.3 m
        # from the first wall it sees.
        self.declare_parameter("hard_min_m",           0.35)
        self.declare_parameter("hard_time_s",          0.5)
        self.declare_parameter("self_filter_pad_m",    0.01)   # own-body points
        self.declare_parameter("cm_block_s",           0.8)    # monitor STOP this long -> escape
        # Reject sweeps returning far fewer points than usual: a partial or
        # corrupted scan makes the world look empty for one cycle, and the
        # planner would act on it immediately.
        self.declare_parameter("scan_min_valid_frac",  0.45)
        self.declare_parameter("self_filter_x",        0.24)   # body half-length
        self.declare_parameter("self_filter_y",        0.18)   # body half-width
        self.declare_parameter("scan_near_keep_m",     0.80)   # never drop a sweep
                                                               # that sees this close
        # v2.8: the high-resolution lidar returns ~2500 points per turn; the
        # footprint sweep keeps one per scan_voxel_m square (0 = all points).
        self.declare_parameter("scan_voxel_m",         0.02)
        # v2.8: 50 Hz output smoothing (see OutputSmoother)
        self.declare_parameter("out_smoothing",        True)
        self.declare_parameter("out_accel_mps2",       0.8)
        self.declare_parameter("out_decel_mps2",       1.5)
        self.declare_parameter("out_steer_slew_dps",   150.0)
        # v2.8: stand-off hysteresis - once holding at the stand-off, start
        # again only when the goal is this much further than goal_tol_m.
        # Without it an owner standing ~1.3 m away made the car stop-go-stop.
        self.declare_parameter("goal_restart_m",       0.20)
        self.declare_parameter("trap_range_m",         2.0)    # a pocket is NEAR the car

        # ── Shared planner settings ───────────────────────────────────
        self.declare_parameter("horizon_m",          1.3)  # path length examined
        self.declare_parameter("rollout_dt",         0.1)
        self.declare_parameter("lethal_cost_thresh", 90)

        # DWA
        self.declare_parameter("dwa_v_samples",     4)
        self.declare_parameter("dwa_steer_samples", 15)
        self.declare_parameter("dwa_w_cmd",         1.0)
        self.declare_parameter("dwa_w_clearance",   0.7)
        self.declare_parameter("dwa_w_speed",       0.35)
        self.declare_parameter("dwa_w_prox",        1.2)   # early-turn strength
        self.declare_parameter("approach_slow_m",   1.10)  # start easing off here
        self.declare_parameter("approach_min_frac", 0.45)  # floor as frac of desired

        # ── Heading memory / re-centring ──────────────────────────────
        # ── Shared skill-layer tunables (used by ALL eight planners) ──
        self.declare_parameter("commit_penalty",   0.45)  # cost of flipping sides
        self.declare_parameter("commit_bonus",     0.15)  # reward for staying
        self.declare_parameter("fade_floor",       0.15)  # min cmd weight when blocked
        self.declare_parameter("steer_slew_dps",   90.0)  # servo slew limit, deg/s

        # ── Command arbitration: who is in charge ───────────────────────
        # "jetson_first" : the Jetson's /cmd_vel is the authority. It is
        #                  driven exactly as sent while its arc is clear; the
        #                  planner only bends it (the smallest safe deviation)
        #                  when an obstacle is on it, and hands control back as
        #                  soon as the arc is clear again. If the owner is
        #                  lost, the robot keeps going on the Jetson's last
        #                  command (slower, turn fading out) while avoiding
        #                  obstacles, for lost_continue_s, then waits.
        # "planner"      : the original behaviour - /cmd_vel is only a wish
        #                  and the selected planner re-plans every cycle.
        self.declare_parameter("arbitration_mode",   "jetson_first")
        self.declare_parameter("pass_cost_max",      60)    # arc cost below this = drive the Jetson cmd untouched
        self.declare_parameter("safety_lookahead_s", 1.0)   # check the arc this many seconds ahead (+0.35 m)
        self.declare_parameter("safety_lookahead_min_m", 0.5)  # ...but never less than this
        self.declare_parameter("avoid_cost_max",     75)    # an arc is "safe" in AVOID only below this cost (margin, not just not-hitting)
        self.declare_parameter("avoid_steer_samples", 21)
        self.declare_parameter("avoid_w_cost",       3.0)   # margin vs. smallest deviation (tuned in sim: 13 cm worst-case body clearance)
        self.declare_parameter("avoid_w_flip",       0.35)  # penalty for dodging to the other side mid-dodge
        self.declare_parameter("clear_hold_s",       0.4)   # Jetson arc must stay clear this long to hand back
        self.declare_parameter("lost_continue_s",    3.0)   # keep going this long after the owner is lost (0 = stop at once)
        self.declare_parameter("lost_speed_scale",   0.6)   # speed while continuing without the owner
        self.declare_parameter("lost_turn_decay_s",  1.2)   # the remembered turn fades with this time constant
        self.declare_parameter("pet_state_topic",    "/pet/state")
        self.declare_parameter("lost_states",        "FIND,SEARCH,LOST")   # Jetson FSM: owner not visible
        self.declare_parameter("stop_states",        "IDLE,COMPLETED")     # Jetson FSM: never continue, stop

        # ── Follow pipeline (follow_manager -> goal -> this node) ───────
        # A fresh goal from follow_manager_node takes precedence over the
        # Jetson's /cmd_vel. The goal is turned into a desired (v, w) and
        # then goes through exactly the same obstacle logic as before.
        self.declare_parameter("goal_topic",        "/follow/goal")
        self.declare_parameter("goal_timeout_s",    0.5)
        self.declare_parameter("goal_tol_m",        0.10)   # closer than this to the goal = hold
        self.declare_parameter("goal_face_enabled",  True)   # face the owner when they stand beside the car
        self.declare_parameter("goal_face_deg",      30.0)   # owner further to the side than this -> face
        self.declare_parameter("goal_face_done_deg", 12.0)   # ...until they are this close to straight ahead
        self.declare_parameter("goal_face_zone_m",   0.40)   # only near the stand-off (goal closer than this)
        self.declare_parameter("goal_face_speed",    0.17)   # m/s reverse (above the ESC stall speed)
        self.declare_parameter("goal_face_max_s",    4.0)    # longest single facing manoeuvre
        self.declare_parameter("goal_face_max_m",    0.45)   # a2r_merge: ... and never more reverse than this
        self.declare_parameter("goal_face_cooldown_s", 3.0)
        self.declare_parameter("goal_k_lin",        0.8)    # v = k * distance (then capped)
        self.declare_parameter("goal_k_ang",        1.0)    # v2.2: scale on the pure-pursuit curvature
        self.declare_parameter("goal_pp_lookahead_m", 1.2)  # pure pursuit: never aim closer than this
        self.declare_parameter("goal_bearing_deadband_deg", 3.0)  # tracking jitter below this: straight
        self.declare_parameter("goal_max_speed",    0.6)    # m/s while following a goal
        self.declare_parameter("goal_turn_bearing_deg", 75.0)  # owner further round than this ->
        self.declare_parameter("goal_turn_speed",   0.30)   #   drive a full-lock arc to turn toward them
        # After a goal drops out, keep coasting on it (owner-lost continuation)
        # for this long instead of switching to the Jetson's direct /cmd_vel.
        # Switching between the two controllers every time the owner position
        # skips a frame made the car weave and lose the owner.
        self.declare_parameter("goal_hold_s",       2.0)
        # FollowMe's own /cmd_vel as a fallback when there is no follow goal.
        # OFF by default: with the follow pipeline the goal is the only
        # authority - a missing goal means coast briefly, then wait. On this
        # robot FollowMe's steering reached the planner with the opposite
        # sign, so every fallback steered away from the owner.
        self.declare_parameter("use_jetson_cmd_fallback", False)
        # The costmap treats the car as a 0.195 m circle around base_link, but
        # the bumper is 0.225 m ahead of it. Every path is ALSO checked at a
        # point this far ahead (forward) / behind (reverse) of the centre, so
        # the planner sees an obstacle in front of the nose early enough to
        # steer around it instead of driving up to it.
        self.declare_parameter("footprint_nose_m", 0.20)
        self.declare_parameter("footprint_tail_m", 0.20)
        # The collision monitor has held the car this long while the planner
        # wanted to drive forward -> the way ahead is blocked: reverse escape.
        self.declare_parameter("monitor_escape_s", 1.5)
        # Owner lost (e.g. out of the camera's 70 deg view after an escape):
        # drive a slow arc toward the side the owner was last seen on, with
        # obstacle avoidance, for up to this long - so the camera finds them
        # again instead of the car waiting forever. 0 = just wait.
        self.declare_parameter("owner_search_s", 10.0)
        self.declare_parameter("jetson_angular_sign", 1.0)   # -1 flips FollowMe's steering
        self.declare_parameter("escape_topic",      "/cmd_vel_escape")  # escape manoeuvre -> twist_mux
        # a2r_pi_update2: multi-point turns (maneuver.py)
        self.declare_parameter("maneuver_enabled",        True)
        self.declare_parameter("maneuver_scan_topic",     "/scan/merged")       # what the monitor sees
        self.declare_parameter("maneuver_odom_topic",     "/odometry/filtered")
        self.declare_parameter("maneuver_crawl_topic",    "/collision_monitor/crawl")
        self.declare_parameter("maneuver_speed",          0.18)    # m/s, just above the stall floor
        self.declare_parameter("maneuver_stop_buffer_m",  0.20)    # >= monitor crawl stop distance
        self.declare_parameter("maneuver_margin_m",       0.04)    # search margin (1 cm > verify)
        self.declare_parameter("maneuver_verify_margin_m", 0.03)   # = collision_monitor margin_m
        self.declare_parameter("maneuver_step_m",         0.10)
        self.declare_parameter("maneuver_time_budget_s",  2.0)
        self.declare_parameter("maneuver_max_expand",     6000)
        self.declare_parameter("maneuver_face_deg",       75.0)    # owner further round -> K-turn
        self.declare_parameter("maneuver_face_tol_deg",   18.0)
        self.declare_parameter("maneuver_drive_tol_deg",  30.0)
        self.declare_parameter("maneuver_free_m",         0.6)
        self.declare_parameter("maneuver_owner_dist_m",   1.2)
        self.declare_parameter("maneuver_max_leg_m",      1.0)
        self.declare_parameter("maneuver_max_legs",       12)
        self.declare_parameter("maneuver_max_s",          45.0)
        self.declare_parameter("maneuver_cooldown_s",     3.0)
        self.declare_parameter("maneuver_leg_pause_s",    0.3)
        self.declare_parameter("maneuver_scan_min_m",     0.10)    # = collision_monitor scan_min_range
        self.declare_parameter("maneuver_area_m",         2.2)
        # a2r_fix_freeze: no full route -> drive the plan that turns the car at least this much
        # further toward the goal, then re-plan (0 = off, the old all-or-nothing search)
        self.declare_parameter("maneuver_partial_min_gain_deg", 20.0)
        # a2r_merge: owner-aware reversing + a faster search (see MULTI-POINT TURNS)
        self.declare_parameter("maneuver_max_reverse_m",   1.2)     # all reverse legs of one plan
        self.declare_parameter("maneuver_max_rev_leg_m",   0.5)     # one reverse leg
        self.declare_parameter("maneuver_owner_keep_m",    0.75)    # never closer to the owner
        self.declare_parameter("maneuver_owner_w",         2.0)
        self.declare_parameter("maneuver_away_w",          0.8)     # reverse AWAY from the owner costs
        self.declare_parameter("maneuver_reverse_w",       1.8)
        self.declare_parameter("maneuver_xy_bin_m",        0.10)
        self.declare_parameter("maneuver_yaw_bin_deg",     15.0)
        self.declare_parameter("maneuver_h_weight",        2.0)
        self.declare_parameter("maneuver_min_leg_m",       0.2)     # no back-and-forth stubs
        self.declare_parameter("maneuver_switch_w",        2.5)     # a2r_step3: cost of one direction change (m)
        self.declare_parameter("maneuver_rev_leg_k",       1.0)     # a2r_step3: reverse leg <= k * arc still needed
        self.declare_parameter("maneuver_rev_total_k",     1.5)     # a2r_step3: all reverse <= k * arc still needed
        self.declare_parameter("maneuver_rev_cut_min_m",   0.15)    # reverse at least this before the
                                                                     # live "forward turn free" cut
        self.declare_parameter("owner_topic",              "/follow/owner")   # lidar-leg owner track
        self.declare_parameter("owner_fresh_s",            0.6)
        # a2r_merge: owner lost with a wall in front -> reverse a little and look again
        self.declare_parameter("lost_backoff_enabled",     True)
        self.declare_parameter("lost_backoff_front_m",     0.45)    # forward room below this = at the wall
        self.declare_parameter("lost_backoff_m",           0.35)
        self.declare_parameter("lost_backoff_wait_s",      0.6)     # owner gone this long first
        self.declare_parameter("lost_backoff_cooldown_s",  4.0)
        self.declare_parameter("escape_force_after_s",     0.6)     # was 1.5 hard-coded: the stare at walls

        self.declare_parameter("heading_memory",      False)
        self.declare_parameter("heading_w",           0.55)   # DWA weight
        self.declare_parameter("heading_norm_deg",    60.0)   # normalising scale
        self.declare_parameter("heading_decay_s",     8.0)    # drift bleed-off
        self.declare_parameter("heading_deadband_deg", 4.0)   # ignore tiny errors
        self.declare_parameter("use_imu_yaw",         True)   # auto-falls back

        # ── Driver-style reverse ──────────────────────────────────────
        self.declare_parameter("escape_presteer_s",   0.45)   # servo swing time
        self.declare_parameter("escape_lock_frac",    0.90)   # of max_steer
        self.declare_parameter("escape_post_bias_s",  2.6)    # drive out, do not re-enter
        self.declare_parameter("escape_post_bias_w",  0.8)
        self.declare_parameter("escape_alternate",    True)   # flip side if repeating
        self.declare_parameter("escape_lookahead",    True)   # simulate before committing
        self.declare_parameter("escape_dists",  [0.30, 0.45, 0.60])  # a2r_merge: was up to 1.2 m - backed far too far
        self.declare_parameter("escape_steer_samples", 7)
        self.declare_parameter("escape_min_yaw_deg",  55.0)  # a decisive nose swing
        self.declare_parameter("escape_min_gain_m",   0.35)   # extra room worth backing for
        self.declare_parameter("escape_eval_m",       2.5)    # lookahead depth (> horizon_m)
        self.declare_parameter("escape_rotation_w",   0.9)    # value of swinging the nose
        self.declare_parameter("escape_reset_heading", True)  # escape = intended new course

        # ── VFH+T trap detection (Neamah, Elek, Korondi 2024) ─────────
        self.declare_parameter("trap_group_dist_m",  0.40)   # 2·rr adjacency threshold
        self.declare_parameter("trap_span_deg",      160.0)  # min azimuth for internal trap
        self.declare_parameter("trap_concave_frac",  0.70)   # fraction passing concavity test
        self.declare_parameter("trap_exit_w",        2.5)    # DWA bias toward exit
        self.declare_parameter("orbit_detect_s",     8.0)    # sustained turn → orbit trap
        self.declare_parameter("orbit_steer_deg",    12.0)   # threshold for "turning hard"

        # ── Planner selection ─────────────────────────────────────────
        # The original file selected a planner by commenting out six of
        # seven dispatch lines. That works, but it means changing planner
        # requires editing source on the robot, and it makes A/B testing
        # in the field impossible. It is now a parameter:
        #
        #   ros2 param set /smart_planner_node planner vfh
        #   ros2 launch a2r_bot_follow follow.launch.py planner:=rrt
        #
        # All eight implementations are still present and unchanged; only
        # the selection mechanism moved.
        self.declare_parameter("planner", "pure_pursuit")
        self.declare_parameter("planner_fallback", "dwa")

        # ── Output routing ────────────────────────────────────────────
        # "topic"  : publish a Twist on cmd_out_topic and let
        #            a2r_bot_arduino own the serial port. DEFAULT.
        # "serial" : open the port here, exactly as the original did.
        #            Useful for running the planner standalone with no
        #            other nodes, and for bisecting problems.
        # Two processes must never open one tty. The bridge is the owner.
        self.declare_parameter("output_mode", "topic")
        self.declare_parameter("cmd_out_topic", "/local_planner/cmd_out")
        self.declare_parameter("imu_topic", "/imu/data")

        # ── 8) RRT ────────────────────────────────────────────────────
        # These defaults were measured, not guessed. scripts/test_rrt_offline.py
        # runs this planner against four scenarios — an open field, a wall
        # with a gap, a two-wall S-curve that no single-curvature planner can
        # solve, and a pocket that can only be left in reverse — over 25
        # random seeds each, reporting solve rate and plan time. Sweeping:
        #
        #   iters  bias  greedy  nodes  step | S-curve | plan time
        #   ------------------------------------------------------
        #    600   0.15    20     900   0.22 |   56%   |   25 ms
        #    900   0.15    20     900   0.22 |   60%   |   29 ms
        #    900   0.15    30    1400   0.22 |   96%   |   29 ms  <-- here
        #    900   0.15    30    1400   0.16 |   56%   |   48 ms
        #    600   0.15    40    1400   0.12 |   20%   |   33 ms
        #
        # The binding constraint turned out to be the node cap, not the
        # iteration count: raising max_nodes from 900 to 1400 took the solve
        # rate from 60% to 96% for no extra time, because the loop was hitting
        # the cap and stopping early. Shrinking the step made things markedly
        # worse — a finer step turns less per node, so the tree needs many
        # more nodes to fan out, and it hits the cap sooner.
        #
        # Timings are from a development machine; a Pi 4 is several times
        # slower, so budget roughly 100 ms there against the 10 Hz plan loop.
        # If it overruns, lower rrt_iterations before anything else — it is
        # the parameter with the gentlest quality gradient.
        self.declare_parameter("rrt_iterations",     900)
        self.declare_parameter("rrt_step_m",         0.22)
        self.declare_parameter("rrt_goal_bias",      0.15)
        self.declare_parameter("rrt_goal_tol_m",     0.28)
        self.declare_parameter("rrt_reuse_frac",     0.5)
        self.declare_parameter("rrt_smooth_passes",  40)
        self.declare_parameter("rrt_lookahead_m",    0.55)
        self.declare_parameter("rrt_sample_span_deg", 220.0)
        self.declare_parameter("rrt_greedy_steps",   30)
        self.declare_parameter("rrt_max_nodes",      1400)

        g = lambda n: self.get_parameter(n).value
        self._baud       = int(g("baud"));            self._dry = bool(g("dry_run"))
        self._max_mps    = float(g("max_speed_mps")); self._max_cmd = float(g("max_speed_cmd"))
        self._wheelbase  = float(g("wheelbase_m"))
        self._max_steer  = math.radians(float(g("max_steer_deg")))
        self._v_floor    = float(g("min_speed_for_steer"))
        self._min_drive  = float(g("min_drive_speed"))
        self._clear_gain = float(g("clearance_gain"))
        self._stall_time = float(g("stall_time_s"))
        self._rev_speed  = float(g("reverse_speed"))
        self._rev_time   = float(g("reverse_time_s"))
        self._rev_cool   = float(g("reverse_cooldown_s"))
        self._esc_rev_delay = float(g("esc_reverse_delay_s"))
        self._rate       = float(g("cmd_rate_hz"));   self._timeout = float(g("cmd_timeout_s"))
        self._plan_rate  = float(g("plan_rate_hz"))
        self._map_size   = float(g("costmap_size_m")); self._res = float(g("costmap_res_m"))
        self._robot_r    = float(g("robot_radius_m"))
        self._infl_r     = float(g("inflation_radius_m"))
        self._infl_k     = float(g("inflation_cost_scale"))
        self._frame      = str(g("costmap_frame"))
        self._lx, self._ly = float(g("lidar_offset_x")), float(g("lidar_offset_y"))
        self._lyaw       = float(g("lidar_yaw_offset"))
        self._scan_max   = float(g("scan_max_range"))
        self._scan_min   = float(g("scan_min_range"))
        self._car = Ackermann(self._wheelbase, self._max_steer,
                              float(g("footprint_front_m")), float(g("footprint_rear_m")),
                              float(g("footprint_half_width_m")))
        self._fp_margin  = float(g("footprint_margin_m"))
        self._hard_min   = float(g("hard_min_m"))
        self._hard_t     = float(g("hard_time_s"))
        self._self_pad   = float(g("self_filter_pad_m"))
        self._cm_block   = float(g("cm_block_s"))
        self._scan_pts   = np.zeros((0, 2), dtype=np.float32)
        self._cm_since   = None          # collision_monitor: STOP of a real command since
        self._cm_t       = -1e9          # last /collision_monitor/state received
        self._scan_frac  = float(g("scan_min_valid_frac"))
        self._body_x    = float(g("self_filter_x"))
        self._body_y    = float(g("self_filter_y"))
        self._near_keep = float(g("scan_near_keep_m"))
        self._voxel     = float(g("scan_voxel_m"))
        self._goal_restart = float(g("goal_restart_m"))
        self._at_standoff = False
        self._trap_range = float(g("trap_range_m"))
        self._smooth_on = bool(g("out_smoothing"))
        self._scan_hist  = []
        self._scan_drop  = 0
        self._horizon_m  = float(g("horizon_m"));     self._dt = float(g("rollout_dt"))
        self._lethal     = int(g("lethal_cost_thresh"))
        self._n_v        = int(g("dwa_v_samples"));   self._n_s = int(g("dwa_steer_samples"))
        self._w_cmd      = float(g("dwa_w_cmd"))
        self._w_clear    = float(g("dwa_w_clearance"))
        self._w_speed    = float(g("dwa_w_speed"))
        self._w_prox     = float(g("dwa_w_prox"))
        self._appr_m     = float(g("approach_slow_m"))
        self._appr_min   = float(g("approach_min_frac"))
        self._commit_pen = float(g("commit_penalty"))
        self._commit_bon = float(g("commit_bonus"))
        self._fade_floor = float(g("fade_floor"))
        self._slew       = math.radians(float(g("steer_slew_dps")))
        self._arb_mode   = str(g("arbitration_mode")).strip().lower()
        self._pass_cost  = int(g("pass_cost_max"))
        self._safety_t   = float(g("safety_lookahead_s"))
        self._look_min   = float(g("safety_lookahead_min_m"))
        self._avoid_cmax = int(g("avoid_cost_max"))
        self._avoid_n    = max(5, int(g("avoid_steer_samples")))
        self._avoid_wc   = float(g("avoid_w_cost"))
        self._avoid_wf   = float(g("avoid_w_flip"))
        self._clear_hold = float(g("clear_hold_s"))
        self._lost_s     = float(g("lost_continue_s"))
        self._lost_scale = float(g("lost_speed_scale"))
        self._lost_decay = max(0.05, float(g("lost_turn_decay_s")))
        _split = lambda v: {x.strip().upper() for x in str(v).split(",") if x.strip()}
        self._lost_states = _split(g("lost_states"))
        self._stop_states = _split(g("stop_states"))
        self._goal_to     = float(g("goal_timeout_s"))
        self._goal_tol    = float(g("goal_tol_m"))
        self._face_on     = bool(g("goal_face_enabled"))
        self._face_b      = math.radians(float(g("goal_face_deg")))
        self._face_done   = math.radians(float(g("goal_face_done_deg")))
        self._face_zone   = float(g("goal_face_zone_m"))
        self._face_v      = float(g("goal_face_speed"))
        self._face_max_s  = float(g("goal_face_max_s"))
        self._face_max_m  = float(g("goal_face_max_m"))
        self._face_cool   = float(g("goal_face_cooldown_s"))
        self._facing, self._face_since, self._face_pause_until = False, 0.0, 0.0
        self._goal_klin   = float(g("goal_k_lin"))
        self._goal_kang   = float(g("goal_k_ang"))
        self._goal_ld     = float(g("goal_pp_lookahead_m"))
        self._goal_db     = math.radians(float(g("goal_bearing_deadband_deg")))
        self._goal_vmax   = float(g("goal_max_speed"))
        self._goal_turn_b = math.radians(float(g("goal_turn_bearing_deg")))
        self._goal_turn_v = float(g("goal_turn_speed"))
        self._goal_hold   = float(g("goal_hold_s"))
        self._jetson_fallback = bool(g("use_jetson_cmd_fallback"))
        self._nose_m = max(0.0, float(g("footprint_nose_m")))
        self._tail_m = max(0.0, float(g("footprint_tail_m")))
        self._mon_esc_s = float(g("monitor_escape_s"))
        self._mon_block_since = None
        self._search_s = float(g("owner_search_s"))
        self._last_goal_b = 0.0
        self._last_goal_d = 0.0
        self._jetson_wsign = float(g("jetson_angular_sign"))
        self._last_goal_t = -1e9
        self._goal = None          # (x, y, receive time) in base_link
        self._goal_mode = False
        self._pet_tokens = set()
        self._esc_active = False
        self._head_on    = bool(g("heading_memory"))
        self._head_w     = float(g("heading_w"))
        self._head_norm  = math.radians(float(g("heading_norm_deg")))
        self._head_decay = float(g("heading_decay_s"))
        self._head_db    = math.radians(float(g("heading_deadband_deg")))
        self._use_imu    = bool(g("use_imu_yaw"))
        self._pre_s      = float(g("escape_presteer_s"))
        self._lock_frac  = float(g("escape_lock_frac"))
        self._post_s     = float(g("escape_post_bias_s"))
        self._post_w     = float(g("escape_post_bias_w"))
        self._alternate  = bool(g("escape_alternate"))
        self._esc_look   = bool(g("escape_lookahead"))
        self._esc_dists  = [float(d) for d in g("escape_dists")]
        self._esc_n      = int(g("escape_steer_samples"))
        self._esc_gain   = float(g("escape_min_gain_m"))
        self._esc_eval   = float(g("escape_eval_m"))
        self._esc_rot_w  = float(g("escape_rotation_w"))
        self._esc_min_yaw = math.radians(float(g("escape_min_yaw_deg")))
        self._esc_reset  = bool(g("escape_reset_heading"))
        # a2r_pi_update2: multi-point turns
        self._mnv_on      = bool(g("maneuver_enabled"))
        self._mnv_speed   = max(float(g("maneuver_speed")), float(g("min_drive_speed")))
        self._mnv_buf     = float(g("maneuver_stop_buffer_m"))
        self._mnv_vmargin = float(g("maneuver_verify_margin_m"))
        self._mnv_face_b  = math.radians(float(g("maneuver_face_deg")))
        self._mnv_face_tol = math.radians(float(g("maneuver_face_tol_deg")))
        self._mnv_drive_tol = math.radians(float(g("maneuver_drive_tol_deg")))
        self._mnv_free    = float(g("maneuver_free_m"))
        self._mnv_owner_d = float(g("maneuver_owner_dist_m"))
        self._mnv_max_leg = float(g("maneuver_max_leg_m"))
        self._mnv_max_legs = int(g("maneuver_max_legs"))
        self._mnv_max_s   = float(g("maneuver_max_s"))
        self._mnv_cool    = float(g("maneuver_cooldown_s"))
        self._mnv_pause   = float(g("maneuver_leg_pause_s"))
        self._mnv_rmin    = float(g("maneuver_scan_min_m"))
        self._mnv_area    = float(g("maneuver_area_m"))
        self._mnv_slew    = math.radians(max(30.0, float(g("out_steer_slew_dps"))))   # smoother servo rate
        self._mnv_planner = ManeuverPlanner(
            self._car, margin=float(g("maneuver_margin_m")), step=float(g("maneuver_step_m")),
            stop_buffer=self._mnv_buf, area=self._mnv_area, max_expand=int(g("maneuver_max_expand")),
            time_budget_s=float(g("maneuver_time_budget_s")), verify_margin=self._mnv_vmargin,
            partial_min_gain_deg=float(g("maneuver_partial_min_gain_deg")),
            max_reverse_m=float(g("maneuver_max_reverse_m")), max_rev_leg_m=float(g("maneuver_max_rev_leg_m")),
            owner_keep_m=float(g("maneuver_owner_keep_m")), owner_w=float(g("maneuver_owner_w")),
            away_w=float(g("maneuver_away_w")), reverse_w=float(g("maneuver_reverse_w")),
            xy_bin=float(g("maneuver_xy_bin_m")), yaw_bin_deg=float(g("maneuver_yaw_bin_deg")),
            h_weight=float(g("maneuver_h_weight")), min_leg_m=float(g("maneuver_min_leg_m")),
            switch_w=float(g("maneuver_switch_w")),                      # a2r_step3
            rev_leg_k=float(g("maneuver_rev_leg_k")), rev_total_k=float(g("maneuver_rev_total_k")))
        self._mnv_rev_cut = float(g("maneuver_rev_cut_min_m"))
        self._mnv, self._mnv_until = None, 0.0
        self._mnv_scan, self._mnv_odom = None, None
        self._mnv_pts_cache = (None, None)          # a2r_merge: (scan stamp, points) - built once per scan
        self._trig = {}                              # a2r_merge: beam cos/sin tables per scan layout
        # a2r_merge: the owner's lidar-leg track (owner_fusion relays the Jetson keeper), base_link
        self._owner_pt, self._owner_fresh = None, float(g("owner_fresh_s"))
        self.create_subscription(PointStamped, str(g("owner_topic")), self._owner_cb, 1)
        # a2r_merge: lost-owner back-off at a wall
        self._bo_on = bool(g("lost_backoff_enabled"))
        self._bo_front, self._bo_m = float(g("lost_backoff_front_m")), float(g("lost_backoff_m"))
        self._bo_wait, self._bo_cool = float(g("lost_backoff_wait_s")), float(g("lost_backoff_cooldown_s"))
        self._bo, self._bo_next, self._lost_since = None, 0.0, None
        self._esc_force_s = float(g("escape_force_after_s"))
        # a2r_merge: IMU yaw, integrated here (manoeuvre / escape progress on steered arcs)
        self._imu_yaw, self._imu_t = 0.0, None
        self.create_subscription(LaserScan, str(g("maneuver_scan_topic")), self._mnv_scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g("maneuver_odom_topic")), self._mnv_odom_cb, 10)
        self._crawl_pub = self.create_publisher(Bool, str(g("maneuver_crawl_topic")), 10)
        self._mnv_path_pub = self.create_publisher(Path, "/local_planner/maneuver_path", 1)

        # VFH+T params
        self._trap_group_dist = float(g("trap_group_dist_m"))
        self._trap_span  = math.radians(float(g("trap_span_deg")))
        self._trap_frac  = float(g("trap_concave_frac"))
        self._trap_w     = float(g("trap_exit_w"))
        self._vfht_pub   = self.create_publisher(Bool, "/local_planner/vfht_trap", 10)
        self._orbit_t    = float(g("orbit_detect_s"))
        self._orbit_th   = math.radians(float(g("orbit_steer_deg")))

        # ── RRT ───────────────────────────────────────────────────────
        self._rrt_iters   = int(g("rrt_iterations"))
        self._rrt_step    = float(g("rrt_step_m"))
        self._rrt_bias    = float(g("rrt_goal_bias"))
        self._rrt_tol     = float(g("rrt_goal_tol_m"))
        self._rrt_reuse   = float(g("rrt_reuse_frac"))
        self._rrt_smooth  = int(g("rrt_smooth_passes"))
        self._rrt_look    = float(g("rrt_lookahead_m"))
        self._rrt_span    = math.radians(
            float(g("rrt_sample_span_deg")))
        self._rrt_greedy  = int(g("rrt_greedy_steps"))
        self._rrt_max_nodes = int(g("rrt_max_nodes"))
        self._rrt_tree    = None      # cached tree, reused between cycles
        self._smoother = OutputSmoother(
            accel=float(g("out_accel_mps2")) / self._max_mps,
            decel=float(g("out_decel_mps2")) / self._max_mps,
            steer_rate=math.radians(float(g("out_steer_slew_dps"))) / self._max_steer,
            floor=self._min_drive / self._max_mps)
        self._rrt_path    = None      # last accepted waypoint list

        # ── Output routing ────────────────────────────────────────────
        self._out_mode = str(g("output_mode")).strip().lower()
        if self._out_mode not in ("topic", "serial"):
            self.get_logger().warn(
                f"output_mode '{self._out_mode}' unknown — using topic")
            self._out_mode = "topic"

        # ── Planner dispatch table ────────────────────────────────────
        # Built once here rather than looked up per cycle. The bound
        # methods are the same ones the comment-swap used to call.
        self._planners = {
            "dwa":          self._plan_dwa,
            "apf":          self._plan_apf,
            "teb":          self._plan_teb,
            "vfh":          self._plan_vfh,
            "pure_pursuit": self._plan_pure_pursuit,
            "fgm":          self._plan_fgm,
            "lattice":      self._plan_lattice,
            "rrt":          self._plan_rrt,
        }
        # Friendly aliases, because people type what they remember.
        self._planner_aliases = {
            "pp": "pure_pursuit", "purepursuit": "pure_pursuit",
            "potential": "apf", "apf_potential": "apf",
            "vfh+": "vfh", "vfh_plus": "vfh",
            "gap": "fgm", "follow_the_gap": "fgm",
            "state_lattice": "lattice", "arc_lattice": "lattice",
            "rrt_star": "rrt", "rrt*": "rrt",
        }
        self._planner_name = self._resolve_planner(str(g("planner")))
        self._planner_fallback = self._resolve_planner(
            str(g("planner_fallback")))
        self._planner_fn = self._planners[self._planner_name]
        self._planner_faults = 0
        self.add_on_set_parameters_callback(self._on_param_set)

        # ── Costmap grid ──────────────────────────────────────────────
        self._cells = int(round(self._map_size / self._res))
        self._half  = self._map_size / 2.0
        self._grid  = np.zeros((self._cells, self._cells), dtype=np.int8)
        self._grid_lock = threading.Lock()
        self._build_inflation_offsets()

        # ── Latest scan in base frame (for FGM) ───────────────────────
        self._scan_pts: Optional[np.ndarray] = None   # (N,2) x,y
        self._scan_ang: Optional[np.ndarray] = None   # (N,)  bearings

        # ── Command / output state ────────────────────────────────────
        self._des_v = 0.0
        self._des_w = 0.0
        self._last_cmd_t = 0.0

        # ── Arbitration state (jetson_first) ──────────────────────────
        self._pet_state = ""
        self._pet_t = 0.0
        self._mem_v = 0.0          # last moving Jetson command (owner visible)
        self._mem_w = 0.0
        self._mem_t = 0.0          # 0 = no valid memory
        self._arb = "PASS"         # PASS (Jetson drives) | AVOID (bending it)
        self._arb_clear_since = 0.0
        self._avoid_side = 0.0     # which way the current dodge goes
        self._arb_last_steer = 0.0
        self._out_v = 0.0          # normalised [-1..1] for firmware
        self._out_s = 0.0
        self._stopped = True
        self._lock = threading.Lock()

        # ── Stall / escape state ──────────────────────────────────────
        self._state        = ST_FOLLOW
        self._stall_since  = None      # monotonic time when stall began
        self._rev_until    = 0.0       # escape ends at this time
        self._rev_steer    = 0.0
        self._last_rev_end = 0.0

        # ── Heading memory ────────────────────────────────────────────
        self._head_dev  = 0.0     # rad, accumulated drift from commanded bearing
        self._head_t    = 0.0     # last integration timestamp
        self._exec_v    = 0.0     # what we last actually commanded (m/s)
        # Side commitment: once a turn direction is chosen, stick with it
        # for a few cycles so the car doesn't wiggle left-right-left.
        self._commit_sign = 0.0    # +1 = committed right, -1 = committed left
        self._commit_age  = 0      # how many cycles since we committed
        self._exec_st   = 0.0     # what we last actually commanded (rad)
        self._imu_wz    = 0.0     # latest gyro z
        self._imu_bias  = 0.0     # gyro zero-rate offset, learned while stopped
        self._imu_n     = 0       # messages seen
        self._imu_live  = 0       # messages with a non-zero rate
        self._imu_ok    = False   # decided after the first ~40 messages
        self._imu_judged = False

        # ── Driver-style reverse ──────────────────────────────────────
        self._presteer_until = 0.0
        self._post_until     = 0.0
        self._post_dir       = 0.0
        self._post_steer     = 0.0    # forward steer the lookahead proved works
        self._esc_count      = 0
        self._last_esc_sign  = 0.0      # +1/-1: which way the last escape steered
        self._good_since     = 0.0
        self._hold_since     = 0.0

        # ── VFH+T trap state ──────────────────────────────────────────
        self._trap_exit      = None     # exit direction (rad), or None
        self._trap_exit_t    = 0.0      # when trap was last detected
        self._orbit_dir      = 0        # +1/-1 sustained turn direction
        self._orbit_since    = 0.0      # when sustained turn started
        self._planner_pinned = False   # set by _finish(), read by _plan_loop
        self._last_steer     = None    # steering slew-limit memory

        # ── Serial ────────────────────────────────────────────────────
        self._ser: Optional["serial.Serial"] = None
        self._rx_buf = ""
        if self._out_mode == "serial":
            if not self._dry:
                self._open_serial(str(g("port")))
            else:
                self.get_logger().warn("DRY RUN — frames logged only")
        else:
            # a2r_bot_arduino owns the device. We publish a Twist and it
            # does the talking. This is the only arrangement in which the
            # IMU, the sonars and the battery can also reach ROS, because
            # a serial port has exactly one reader.
            self._dry = True
            self.get_logger().info(
                "output_mode=topic — a2r_bot_arduino owns the serial port")

        # ── QoS ───────────────────────────────────────────────────────
        cmd_qos  = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        scan_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,   # a2r_fix_freeze: was 5
                              history=HistoryPolicy.KEEP_LAST)
        map_qos  = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST)

        # ── Subs / pubs ───────────────────────────────────────────────
        self.create_subscription(Twist, str(g("cmd_vel_topic")), self._cmd_cb, cmd_qos)
        self.create_subscription(LaserScan, str(g("scan_topic")), self._scan_cb, scan_qos)
        self.create_subscription(String, "/bridge/command", self._special_cb, 10)
        self.create_subscription(String, str(g("pet_state_topic")), self._pet_cb, 10)
        self.create_subscription(PoseStamped, str(g("goal_topic")), self._goal_cb, 1)  # newest only
        # Listen to our own republished IMU so heading memory can use the gyro
        # when the firmware actually provides one.
        # The fused /imu/data carries an orientation and a bias-corrected
        # gyro; /imu/data_raw is the unfiltered fallback when the fusion
        # stack is not running.
        self.create_subscription(Imu, str(g("imu_topic")),
                                 self._gyro_cb, qos_profile_sensor_data)

        self._map_pub   = self.create_publisher(OccupancyGrid, "/local_costmap", map_qos)
        self._cloud_pub = self.create_publisher(PointCloud2, "/local_costmap/cloud", 1)
        self._scan_topic_name = str(g("scan_topic"))
        self._last_map_pub = time.monotonic()
        self.create_timer(1.0, self._costmap_keepalive)
        self._path_pub  = self.create_publisher(Path, "/local_planner/path", 1)
        self._trajs_pub = self.create_publisher(MarkerArray, "/local_planner/trajectories", 1)
        self._best_pub  = self.create_publisher(MarkerArray, "/local_planner/best_traj", 1)
        self._wish_pub  = self.create_publisher(Marker, "/local_planner/desired_cmd", 1)
        self._state_pub = self.create_publisher(Marker, "/local_planner/state", 1)
        self._status_pub = self.create_publisher(String, "/local_planner/status", 10)
        self._imu_pub   = self.create_publisher(Imu,          "/imu/data_raw",  10)
        self._batt_pub  = self.create_publisher(BatteryState, "/battery_state", 10)
        self._enc_pub   = self.create_publisher(Float32,      "/wheel_encoder", 10)
        self._estop_pub = self.create_publisher(Bool,         "/e_stop",        10)
        self._head_pub  = self.create_publisher(Float32, "/local_planner/heading_dev", 10)
        self._cmd_out_pub = self.create_publisher(
            Twist, str(g("cmd_out_topic")), 10)
        self._escape_pub = self.create_publisher(
            Twist, str(g("escape_topic")), 10)
        # The monitor can zero our command; without hearing it, a STOP was
        # invisible here and the car waited in front of the obstacle forever.
        # a2r_merge: ONE subscription (was two, each json-parsing every 50 Hz state message)
        self.create_subscription(String, "/collision_monitor/state", self._cm_state_cb, 10)
        self._planner_pub = self.create_publisher(
            String, "/local_planner/active_planner", 10)
        self._rrt_pub = self.create_publisher(
            MarkerArray, "/local_planner/rrt_tree", 1)

        # ── Timers ────────────────────────────────────────────────────
        # a2r_v5 sync: light check at 2x plan rate; plans only on a NEW scan (or after 1.5 periods)
        self._last_plan_t, self._scan_seq, self._planned_seq = 0.0, 0, -1
        self.create_timer(0.5 / self._plan_rate, self._plan_timer)
        # Send loop runs in its own Python thread — NOT a ROS timer.
        # The GIL blocks ROS timer callbacks while _scan_cb's inflation
        # loop runs (~50-100ms per scan), starving the old timer-based
        # sender. A raw thread with time.sleep() gets scheduled by the
        # OS between GIL switches and keeps a steady 50 Hz stream.
        self._send_thread = threading.Thread(target=self._send_thread_fn, daemon=True)
        self._send_thread.start()
        if not self._dry:      # v2.5: topic mode has no serial port - this 100 Hz
            self.create_timer(0.01,                  self._read_loop)  # callback only ever returned

        self.get_logger().info(
            f"SmartPlanner ready | planner={self._planner_name} | out={self._out_mode} | heading_memory="
            f"{'on' if self._head_on else 'off'} | costmap {self._map_size} m @ "
            f"{self._res} m | lethal r={self._robot_r} m infl={self._infl_r} m | "
            f"v∈[{self._min_drive},{self._max_mps}] m/s steer=±"
            f"{math.degrees(self._max_steer):.0f}° {'[DRY RUN]' if self._dry else ''}")

    # ══════════════════════════════════════════════════════════════════
    #  SERIAL  (JSON protocol of a2r_bot_drive.ino)
    # ══════════════════════════════════════════════════════════════════
    def _open_serial(self, port_param: str):
        if not HAS_SERIAL:
            self.get_logger().error("pyserial missing (pip install pyserial) — DRY RUN")
            self._dry = True
            return
        port = port_param if port_param != "auto" else self._autodetect()
        if port is None:
            self.get_logger().error("No Arduino port found — DRY RUN")
            self._dry = True
            return
        try:
            self._ser = serial.Serial(port, self._baud, timeout=0.0)
            time.sleep(2.0)                       # board auto-resets on open
            self._ser.reset_input_buffer()
            self.get_logger().info(f"Opened serial: {port} @ {self._baud}")
        except Exception as e:
            self.get_logger().error(f"Serial open failed ({port}): {e} — DRY RUN")
            self._dry = True

    def _autodetect(self) -> Optional[str]:
        candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        if not candidates:
            return None
        try:
            import serial.tools.list_ports as lp
            for p in lp.comports():
                vid = f"{p.vid:04x}" if p.vid else ""
                pid = f"{p.pid:04x}" if p.pid else ""
                if (vid, pid) in _ESP32_HINTS:
                    return p.device
        except Exception:
            pass
        return candidates[0]

    def _write(self, obj: dict):
        line = json.dumps(obj) + "\n"
        if self._dry or self._ser is None:
            self.get_logger().info(f"[DRY] → {line.strip()}", throttle_duration_sec=0.5)
            return
        try:
            payload = line.encode("ascii")
            n = self._ser.write(payload)
            # DIAGNOSTIC: prove what actually left the Pi, not just what the
            # planner decided. If n < len(payload), the OS serial buffer is
            # rejecting bytes (often a sign of a wedged/brownout-reset
            # Arduino not draining its RX buffer) — this is exactly the kind
            # of "log looks right, robot doesn't move" gap being chased.
            if n is not None and n < len(payload):
                self.get_logger().warn(
                    f"Serial short write: sent {n}/{len(payload)} bytes — "
                    f"frame may not have reached the Arduino: {line.strip()}")
            if obj.get("cmd") == "DRIVE":
                self.get_logger().info(
                    f"[SENT] → {line.strip()}", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")

    def _special_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd in ("RESET_ENC", "CAL_IMU"):
            self._write({"cmd": cmd})

    def _read_loop(self):
        if self._dry or self._ser is None:
            return
        try:
            n = self._ser.in_waiting
            if n:
                self._rx_buf += self._ser.read(n).decode(errors="ignore")
        except Exception:
            return
        while "\n" in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split("\n", 1)
            line = line.strip()
            if line.startswith("{"):
                self._parse_telemetry(line)

    def _parse_telemetry(self, line: str):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return
        if "status" in d:
            self.get_logger().info(f"Arduino: {d['status']}")
            return
        ts = self.get_clock().now().to_msg()
        if "ax" in d:
            imu = Imu()
            imu.header.stamp = ts
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = float(d.get("ax", 0.0))
            imu.linear_acceleration.y = float(d.get("ay", 0.0))
            imu.linear_acceleration.z = float(d.get("az", 9.80665))
            imu.angular_velocity.x = float(d.get("gx", 0.0))
            imu.angular_velocity.y = float(d.get("gy", 0.0))
            imu.angular_velocity.z = float(d.get("gz", 0.0))
            imu.orientation_covariance[0] = -1.0
            self._imu_pub.publish(imu)
        if "bat_v" in d:
            b = BatteryState()
            b.header.stamp = ts
            b.voltage = float(d["bat_v"])
            b.present = True
            self._batt_pub.publish(b)
        if "enc" in d:
            self._enc_pub.publish(Float32(data=float(d["enc"])))
        if "e_stop" in d:
            self._estop_pub.publish(Bool(data=bool(d["e_stop"])))

    # ══════════════════════════════════════════════════════════════════
    #  DESIRED COMMAND from Jetson tracker
    # ══════════════════════════════════════════════════════════════════
    def _cmd_cb(self, msg: Twist):
        with self._lock:
            self._des_v = msg.linear.x
            self._des_w = msg.angular.z * getattr(self, "_jetson_wsign", 1.0)
            self._last_cmd_t = time.monotonic()

    def _cm_state_cb(self, msg: String):
        """a2r_merge: parse the monitor state once, feed both trackers."""
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._cm_cb(msg, d)
        self._mon_cb(msg, d)

    def _cm_cb(self, msg: String, d=None):
        """Track how long the collision_monitor has been zeroing a real command."""
        if d is None:
            try:
                d = json.loads(msg.data)
            except (ValueError, TypeError):
                return
        now = time.monotonic()
        self._cm_t = now
        stopped = str(d.get("state", "")).startswith("STOP")
        if stopped and abs(float(d.get("in_v", 0.0) or 0.0)) > 0.01:
            self._cm_last = now
            if self._cm_since is None:
                self._cm_since = now
        elif now - getattr(self, "_cm_last", -1e9) > 0.5:
            # a2r_fix_freeze: only a real 0.5 s break resets it - single glitch frames (goal blink,
            # LAST_KNOWN/FOLLOW flapping) restarted the clock and the car stared at the bend
            self._cm_since = None

    def _cm_blocked(self, now: float, hold: Optional[float] = None) -> bool:
        """The monitor has held the car still for `hold` seconds (default cm_block_s)."""
        hold = self._cm_block if hold is None else hold
        return (self._cm_since is not None and (now - self._cm_t) < 0.5
                and (now - self._cm_since) >= hold)

    def _pet_cb(self, msg: String):
        """Jetson FSM state. Used only to tell 'owner lost' (keep going
        carefully) apart from 'owner reached / task done' (stop)."""
        with self._lock:
            self._pet_state = str(msg.data).strip().upper()
            # FollowMe's /followme/state is "STATE|mode|OWNER"; a plain
            # single word (older trackers) works the same way.
            self._pet_tokens = {t.strip() for t in self._pet_state.split("|") if t.strip()}
            self._pet_t = time.monotonic()

    def _mon_cb(self, msg: String, d=None):
        """Collision monitor feedback: how long has it been holding the car
        while we asked to drive FORWARD? (Reverse holds and 'no scan' stops
        do not count.)"""
        if d is None:
            try:
                d = json.loads(msg.data)
            except (ValueError, TypeError):
                return
        st = str(d.get("state", ""))
        try:
            in_v = float(d.get("in_v") or 0.0)
        except (TypeError, ValueError):
            in_v = 0.0
        now = time.monotonic()
        self._mon_t = now
        if st.startswith("STOP") and "scan" not in st and in_v > 0.01:
            self._mon_last = now
            if self._mon_block_since is None:
                self._mon_block_since = now
        elif now - getattr(self, "_mon_last", -1e9) > 0.5:
            self._mon_block_since = None      # a2r_fix_freeze: 0.5 s grace, as _cm_cb

    def _goal_cb(self, msg: PoseStamped):
        with self._lock:
            self._goal = (float(msg.pose.position.x), float(msg.pose.position.y),
                          time.monotonic(),
                          2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w))  # owner bearing

    def _face_step(self, now: float) -> bool:
        """Face the owner (a2r_pi_face_owner.sh). At the stand-off with the owner well to the side
        (outside the camera's view), an Ackermann car cannot turn on the spot: reverse slowly with
        the wheels locked AWAY from the owner, which swings the nose TOWARD them (and adds room).
        Ends when the owner is near straight ahead, the goal moves away, the collision monitor
        blocks the reverse, or after face_max_s (then a pause of face_cooldown_s)."""
        if not self._face_on or self._state != ST_FOLLOW or now < self._face_pause_until:
            self._facing = False
            return False
        with self._lock:
            goal = self._goal
        if goal is None or len(goal) < 4 or now - goal[2] > self._goal_to:
            self._facing = False
            return False
        d, oyaw = math.hypot(goal[0], goal[1]), goal[3]
        o = self._owner_pt                          # a2r_merge: the live lidar-leg bearing if fresh
        if o is not None and now - o[2] <= self._owner_fresh:
            oyaw = math.atan2(o[1], o[0])
        start = abs(oyaw) > self._face_b and d < self._face_zone
        keep = self._facing and abs(oyaw) > self._face_done and d < self._face_zone + 0.3
        if not (start or keep) or self._cm_blocked(now, 0.4):
            if self._facing:
                self._face_pause_until = now + 0.5
            self._facing = False
            return False
        if not self._facing:
            self._face_since = now
            self._face_yaw0 = self._imu_yaw
            od = self._mnv_odom
            self._face_odom0 = od if (od is not None and now - od[3] < 0.3) else None
            self.get_logger().info(f"face owner: owner at {math.degrees(oyaw):+.0f} deg beside the car - "
                                   f"reversing to bring them back in front")
        elif now - self._face_since > self._face_max_s or self._face_travel(now) > self._face_max_m:
            self._facing = False
            self._face_pause_until = now + self._face_cool
            return False
        self._facing = True
        # reversing with LEFT lock swings the heading RIGHT (ackermann.py): lock away from the owner
        self._set_out(-self._face_v, -math.copysign(self._max_steer, oyaw))
        self._publish_state("FACE-OWNER")
        return True

    def _face_travel(self, now) -> float:
        """a2r_merge: how far the facing reverse has gone (odometry, or IMU heading on the lock)."""
        od, o0 = self._mnv_odom, getattr(self, "_face_odom0", None)
        d = 0.0
        if od is not None and o0 is not None and now - od[3] < 0.3:
            d = math.hypot(od[0] - o0[0], od[1] - o0[1])
        s_imu = self._imu_progress([(-1, self._max_steer, 9.9)], self._imu_yaw - getattr(self, "_face_yaw0", 0.0))
        return max(d, s_imu or 0.0)

    def _goal_cmd(self, now: float):
        """Fresh follow goal (base_link) -> desired (v, w), or None."""
        with self._lock:
            goal = self._goal
        if goal is None or now - goal[2] > self._goal_to:
            return None
        gx, gy = goal[0], goal[1]
        d = math.hypot(gx, gy)
        tol = self._goal_tol + (self._goal_restart if self._at_standoff else 0.0)
        if d < tol:
            self._at_standoff = True
            return 0.0, 0.0                               # at the stand-off: hold
        self._at_standoff = False
        b = math.atan2(gy, gx)
        if abs(b) > self._goal_turn_b:
            # owner well off to the side / behind: an Ackermann car cannot
            # turn on the spot, so turn toward them on a full-lock arc
            v = self._goal_turn_v
            return v, math.copysign(v * math.tan(self._max_steer) / self._wheelbase, b)
        v = min(self._goal_vmax, self._goal_klin * d) * max(0.0, math.cos(b))
        if v < 1e-3:
            return 0.0, 0.0
        # v2.2: pure-pursuit curvature k = 2 sin(b) / Ld, not w = k*bearing.
        # On an Ackermann car w = 1.6*b gave ~18 deg of steer for 10 deg of
        # bearing at 0.24 m/s: every bit of tracking jitter flipped the
        # wheels (sim: 120 reversals in 32 s). Pure pursuit gives ~4 deg.
        if abs(b) < self._goal_db:
            return v, 0.0
        return v, self._goal_kang * v * 2.0 * math.sin(b) / max(d, self._goal_ld)

    # ══════════════════════════════════════════════════════════════════
    #  COSTMAP + INFLATION LAYER
    # ══════════════════════════════════════════════════════════════════
    def _build_inflation_offsets(self):
        """Vectorised inflation: for each kernel offset (di,dj) with cost c,
        the whole hit-mask is shifted and max-combined. cost(d):
            d ≤ robot_radius        → 100 (lethal — footprint would collide)
            robot_r < d ≤ infl_r    → 99·exp(-k·(d − robot_r))   (decay)
        """
        rc = int(math.ceil(self._infl_r / self._res))
        offs = []
        for di in range(-rc, rc + 1):
            for dj in range(-rc, rc + 1):
                d = math.hypot(di, dj) * self._res
                if d > self._infl_r:
                    continue
                if d <= self._robot_r:
                    c = COST_LETHAL
                else:
                    c = int(round(COST_INSCRIBED
                                  * math.exp(-self._infl_k * (d - self._robot_r))))
                    c = max(1, min(COST_INSCRIBED - 1, c))
                offs.append((di, dj, c))
        self._infl_offs = offs

    def _scan_cb(self, msg: LaserScan):
        """a2r_v5 sync: build the costmap (full resolution), mark the scan as new."""
        self._scan_cb_raw(msg)
        self._scan_seq += 1

    def _plan_timer(self):
        now = time.monotonic()
        if self._scan_seq != self._planned_seq or now - self._last_plan_t >= 1.5 / max(self._plan_rate, 1e-3):
            self._planned_seq = self._scan_seq
            self._plan_loop()

    def _scan_cb_raw(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        # a2r_merge: beam angles and their cos/sin are the same every scan - computed once
        cA, sA, ab_full = self._beam_trig(n, float(msg.angle_min), float(msg.angle_increment))
        # Full scan for FGM: out-of-range / inf rays count as FREE space,
        # clipped to scan_max (a long ray is an open gap, not an obstacle)
        min_r = max(msg.range_min, self._scan_min)
        # Self filter: drop only returns that land INSIDE the robot's own
        # outline (spoiler, display, Jetson - all behind the lidar). The old
        # 0.40 m range cutoff hid those, but it also hid every obstacle within
        # 40 cm in FRONT of the bumper, and made follow-the-gap treat them as
        # open space. A real obstacle can never be inside the chassis.
        with np.errstate(invalid='ignore', over='ignore'):
            _xb = self._lx + r * cA
            _yb = self._ly + r * sA
            body = (np.abs(_xb) < self._body_x) & (np.abs(_yb) < self._body_y)
        r_full = np.where(np.isfinite(r) & (r > min_r) & ~body, r, self._scan_max)
        r_full = np.minimum(r_full, self._scan_max)
        xf = self._lx + r_full * cA
        yf = self._ly + r_full * sA
        # Own-body returns count as open space for FGM / trap detection too,
        # otherwise the Jetson and battery look like a pocket round the car.
        own = self._car.body_mask(np.stack([xf, yf], axis=1), self._self_pad)
        if own.any():
            xf = np.where(own, self._lx + self._scan_max * cA, xf)
            yf = np.where(own, self._ly + self._scan_max * sA, yf)
        self._scan_ang = np.arctan2(yf, xf)
        self._scan_rng = np.hypot(xf, yf)
        # v2.8: which beams are REAL returns (the rest were set to scan_max)
        with np.errstate(invalid='ignore'):
            self._scan_hit = (np.isfinite(r) & (r > min_r) & (r < self._scan_max)
                              & ~body & ~own)
        # Costmap only marks REAL hits (inside scan_max)
        valid = np.isfinite(r) & (r > min_r) & \
                (r < min(msg.range_max, self._scan_max)) & ~body
        r = r[valid]
        cV, sV = cA[valid], sA[valid]

        # ---- reject implausibly sparse sweeps -------------------------
        n_valid = int(r.size)
        self._scan_hist.append(n_valid)
        if len(self._scan_hist) > 20:
            self._scan_hist.pop(0)
        if len(self._scan_hist) >= 8:
            ref = sorted(self._scan_hist)[len(self._scan_hist) // 2]
            if ref > 20 and n_valid < self._scan_frac * ref and not (r.size and float(r.min()) < self._near_keep):
                self._scan_drop += 1
                self.get_logger().warn(
                    f"Sparse scan ignored ({n_valid} pts vs median {ref}) — "
                    f"{self._scan_drop} dropped so far",
                    throttle_duration_sec=5.0)
                return            # keep the previous costmap

        # lidar frame → base_link (fixed mounting offset, no TF lookup needed)
        x = self._lx + r * cV
        y = self._ly + r * sV
        # The robot's own deck, Jetson and battery show up in the scan. They
        # used to be hidden by scan_min_range 0.40, which also hid every real
        # obstacle within ~25 cm of the bumper. Drop them by footprint instead.
        pts = np.stack([x, y], axis=1)
        keep = ~self._car.body_mask(pts, self._self_pad)
        x, y = x[keep], y[keep]
        # v2.8: thin for the footprint sweep only; the costmap below still
        # marks every hit
        self._scan_pts = voxel_thin(pts[keep], self._voxel)

        # Hit mask
        i = ((x + self._half) / self._res).astype(np.int32)
        j = ((y + self._half) / self._res).astype(np.int32)
        ok = (i >= 0) & (i < self._cells) & (j >= 0) & (j < self._cells)
        hits = np.zeros((self._cells, self._cells), dtype=np.int16)
        hits[i[ok], j[ok]] = 1

        # Vectorised inflation via shifted-max
        grid = np.zeros_like(hits)
        C = self._cells
        # a2r_v7 edt: distance to the nearest hit, then cost(d) - same values as the shifted-max
        if _a2r_edt is not None and hits.any():
            d = _a2r_edt(hits == 0) * self._res
            le = d <= self._robot_r
            dec = (~le) & (d <= self._infl_r)
            grid[dec] = np.clip(np.rint(COST_INSCRIBED * np.exp(-self._infl_k * (d[dec] - self._robot_r))),
                                1, COST_INSCRIBED - 1).astype(grid.dtype)
            grid[le] = COST_LETHAL
        for di, dj, c in (self._infl_offs if (_a2r_edt is None or not hits.any()) else ()):
            si0, si1 = max(0, di),  C + min(0, di)
            ti0, ti1 = max(0, -di), C + min(0, -di)
            sj0, sj1 = max(0, dj),  C + min(0, dj)
            tj0, tj1 = max(0, -dj), C + min(0, -dj)
            src = hits[ti0:ti1, tj0:tj1]
            tgt = grid[si0:si1, sj0:sj1]
            np.maximum(tgt, src * c, out=tgt)

        with self._grid_lock:
            self._grid = np.clip(grid, 0, 100).astype(np.int8)
        self._publish_costmap(msg.header.stamp)

    def _publish_costmap(self, stamp):
        # a2r_fix_freeze: the grid is fresh either way; build and send the messages only when
        # someone (RViz, the Jetson) subscribes - 2 messages per scan for nobody was pure CPU
        self._last_map_pub = time.monotonic()
        if self._cloud_pub.get_subscription_count() > 0:
            self._publish_costmap_cloud(stamp)
        if self._map_pub.get_subscription_count() == 0:
            return
        m = OccupancyGrid()
        m.header.stamp = stamp
        m.header.frame_id = self._frame
        m.info.resolution = self._res
        m.info.width = m.info.height = self._cells
        m.info.origin.position.x = -self._half
        m.info.origin.position.y = -self._half
        m.info.origin.orientation.w = 1.0
        with self._grid_lock:
            m.data = array.array("b", np.ascontiguousarray(self._grid.T, dtype=np.int8).tobytes())
        self._map_pub.publish(m)

    def _costmap_keepalive(self):
        """Republish the current grid if no scan has produced one lately.

        The costmap used to be published ONLY from the scan callback, so
        no scans (lidar down, fusion not running, sweeps rejected as sparse)
        meant RViz sat on "No map received" with no hint why. Now a map is
        always there once the node is up, and a warning names the cause."""
        age = time.monotonic() - getattr(self, '_last_map_pub', 0.0)
        if age > 1.0:
            self._publish_costmap(self.get_clock().now().to_msg())
            self.get_logger().warn(
                f"No usable scan on '{self._scan_topic_name}' for {age:.0f} s - "
                "republishing the last costmap. Check: ros2 topic hz "
                f"{self._scan_topic_name}",
                throttle_duration_sec=10.0)

    def _publish_costmap_cloud(self, stamp):
        """The costmap as a PointCloud2 of its non-free cells (intensity = cost).

        RViz's Map display cannot draw on the Raspberry Pi's GPU driver: its
        shader fails to link ("active samplers with a different type refer to
        the same texture image unit"), so /local_costmap arrives - Width 80,
        Height 80 - but is never drawn. Point clouds use the same shader as the
        laser scan, which works, so this is the costmap to look at on the Pi.
        """
        with self._grid_lock:
            g = self._grid.copy()
        ii, jj = np.nonzero(g > 0)
        n = int(ii.size)
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, 0] = -self._half + (ii + 0.5) * self._res
        pts[:, 1] = -self._half + (jj + 0.5) * self._res
        pts[:, 3] = g[ii, jj]
        c = PointCloud2()
        c.header.stamp = stamp
        c.header.frame_id = self._frame
        c.height, c.width = 1, n
        c.fields = [PointField(name=nm, offset=4 * k,
                               datatype=PointField.FLOAT32, count=1)
                    for k, nm in enumerate(('x', 'y', 'z', 'intensity'))]
        c.is_bigendian = False
        c.point_step, c.row_step = 16, 16 * n
        c.is_dense = True
        c.data = array.array('B', pts.tobytes())
        self._cloud_pub.publish(c)

    def _cost_at(self, x: float, y: float) -> int:
        i = int(math.floor((x + self._half) / self._res))
        j = int(math.floor((y + self._half) / self._res))
        if 0 <= i < self._cells and 0 <= j < self._cells:
            return int(self._grid[i, j])
        return COST_FREE

    # ══════════════════════════════════════════════════════════════════
    #  SHARED PLANNER HELPERS
    # ══════════════════════════════════════════════════════════════════
    def _desired_steer(self, des_v: float, des_w: float) -> float:
        v_eff = max(abs(des_v), self._v_floor)
        return max(-self._max_steer, min(self._max_steer,
                   math.atan(self._wheelbase * des_w / v_eff)))

    def _costs_at(self, xs: np.ndarray, ys: np.ndarray):
        """Vectorised costmap lookup -> (cost int32 array, inside-map mask)."""
        grid = self._grid
        i = np.floor((xs + self._half) / self._res).astype(np.int64)
        j = np.floor((ys + self._half) / self._res).astype(np.int64)
        inside = (i >= 0) & (i < self._cells) & (j >= 0) & (j < self._cells)
        c = np.zeros(len(xs), dtype=np.int32)
        if inside.any():
            c[inside] = grid[i[inside], j[inside]]
        return c, inside

    def _hard_dist(self, v: float) -> float:
        """A footprint hit closer than this along the arc makes it lethal."""
        return max(self._hard_min, self._hard_t * abs(v))

    def _rollout(self, v: float, steer: float,
                 dist: Optional[float] = None, hard: Optional[float] = None,
                 ramp: bool = True) -> Tuple[List[Tuple[float, float]], int, float]:
        """Ackermann rollout: exact kinematic bicycle + swept footprint.

        Three things changed, all of them reasons the car stopped in front
        of obstacles instead of going round them:

          1. KINEMATICS. The old loop integrated the REAR-AXLE arc as if it
             were base_link's (base_link is mid-wheelbase), used the wrong
             wheelbase (0.257 m, measured 0.280 m) and assumed the wheels
             snap to the new angle instantly. It believed the car turns
             ~10 % tighter and a lot sooner than it can. Now: ackermann.py,
             with the steering ramping at the servo slew rate.
          2. FOOTPRINT. Only base_link was checked, against a 0.195 m circle.
             The nose is 0.215 m out and the outer front corner swings 0.73 m
             from the turn centre, so the planner happily chose arcs that
             put the bumper into things; the collision_monitor then zeroed
             them and the car sat there. The real rectangle (+ margin) is
             now swept along the path, as the monitor does.
          3. HARD vs SOFT. A footprint hit within _hard_dist(v) is lethal.
             Further out it only shortens clear_dist, so approach easing,
             early turning and the pinned-at-lock stall signal (all of which
             were dead code, because every accepted arc reported its full
             length as clear) work again, and the car keeps creeping toward
             the opening instead of freezing 1.3 m from the first wall.

        The costmap is now purely a SOFT clearance cost for scoring; the
        footprint alone decides collisions. Sampled by arc length, so cost is
        a property of the path, not of the intended speed. v < 0 = reverse.
        Returns (points, max_cost, clear_dist)."""
        if dist is None:
            dist = self._horizon_m
        step = min(0.5 * self._res, max(abs(v), 0.05) * self._dt)
        sgn = 1.0 if v >= 0.0 else -1.0
        slew = 0.0
        if ramp and self._slew > 0.0:
            slew = self._slew / max(abs(v), 0.1)            # rad per metre
        s, x, y, yaw = self._car.poses(
            steer, sgn * dist, step,
            steer0=float(getattr(self, "_exec_st", 0.0)) if slew > 0.0 else None,
            slew_per_m=slew)
        hit = self._car.first_hit(self._scan_pts, s, x, y, yaw, self._fp_margin)
        hard = self._hard_dist(v) if hard is None else hard
        sd = np.abs(s[1:])
        xs, ys = x[1:], y[1:]
        cost, _ = self._costs_at(xs, ys)
        if hit <= hard:
            k = int(np.searchsorted(sd, hit)) + 1
            self._roll_costs = cost[:k].tolist()
            return (list(zip(xs[:k].tolist(), ys[:k].tolist())),
                    COST_LETHAL, float(hit))
        if math.isfinite(hit):
            keep = sd < hit
            xs, ys, cost, sd = xs[keep], ys[keep], cost[keep], sd[keep]
        # soft cost only: the footprint said this part of the path is free
        max_c = int(min(int(cost.max()) if len(cost) else 0, self._lethal - 1))
        clear_dist = float(min(hit, dist))
        haz = np.flatnonzero(cost >= COST_INSCRIBED)
        if len(haz):
            clear_dist = min(clear_dist, float(sd[haz[0]]))
        self._roll_costs = cost.tolist()
        return list(zip(xs.tolist(), ys.tolist())), max_c, clear_dist

    def _speed_law(self, des_v: float, max_cost: int) -> float:
        """Anti-stall speed shaping. Clearance scales speed LINEARLY with a
        hard floor: v = des·(1−gain + gain·clear), then clamped ≥ min_drive.
        With gain=0.6 even max_cost=89 keeps ≥40 % of desired speed."""
        clear = 1.0 - max_cost / 100.0
        v = abs(des_v) * ((1.0 - self._clear_gain) + self._clear_gain * clear)
        v = min(v, self._max_mps)
        return max(v, self._min_drive)

    def _side_openness(self) -> float:
        """+1 = left much freer, −1 = right much freer (front semicircle)."""
        with self._grid_lock:
            grid = self._grid.astype(np.float32)
        C, h = self._cells, self._cells // 2
        front = grid[h:, :]                      # x > 0 rows
        left  = float(front[:, h:].sum())        # y > 0
        right = float(front[:, :h].sum())        # y < 0
        tot = left + right + 1e-6
        return (right - left) / tot              # right blocked → positive → go left

    # ══════════════════════════════════════════════════════════════════
    #  PLANNER LOOP + STALL / REVERSE-ESCAPE MANAGER
    # ══════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════
    #  HEADING MEMORY
    # ══════════════════════════════════════════════════════════════════
    def _gyro_cb(self, msg: Imu):
        """Watch the IMU and decide, once, whether it is worth trusting.
        A firmware that publishes all-zero gyro data is common; rather than
        silently integrating zeros we detect it and fall back to the model."""
        wz = float(msg.angular_velocity.z)
        self._imu_wz = wz
        t = time.monotonic()                          # a2r_merge: integrated yaw (bias-corrected)
        if self._imu_t is not None and 0.0 < t - self._imu_t < 0.2:
            self._imu_yaw += (wz - self._imu_bias) * (t - self._imu_t)
        self._imu_t = t
        self._imu_n += 1
        if abs(wz) > 1e-4:
            self._imu_live += 1
        # Learn the zero-rate offset while the car is not driving
        if abs(self._exec_v) < 1e-3:
            self._imu_bias += 0.02 * (wz - self._imu_bias)
        if (not self._imu_judged) and self._imu_n >= 40:
            self._imu_judged = True
            self._imu_ok = self._use_imu and (self._imu_live >= 4)
            if self._imu_ok:
                self.get_logger().info("Heading source: IMU gyro")
            else:
                self.get_logger().warn(
                    "IMU gyro reads all zeros — heading memory will use the "
                    "steering model instead (fix the firmware to improve it)")

    def _yaw_rate(self) -> float:
        """Best available estimate of the car's current yaw rate."""
        if self._imu_ok:
            return self._imu_wz - self._imu_bias
        # Bicycle model on what we actually commanded
        return (self._exec_v / self._wheelbase) * math.tan(self._exec_st)

    def _update_heading(self, now: float, des_w: float):
        """Integrate (actual - requested) yaw rate into an accumulated drift.
        Zero drift means 'pointing where the Jetson asked'."""
        dt = now - self._head_t if self._head_t > 0.0 else 0.0
        self._head_t = now
        if dt <= 0.0 or dt > 0.5:
            return
        self._head_dev += (self._yaw_rate() - des_w) * dt
        # Bleed off slowly: without absolute yaw this is dead reckoning, and
        # we would rather forget an old error than chase an accumulated lie.
        if self._head_decay > 0.0:
            self._head_dev *= math.exp(-dt / self._head_decay)
        lim = math.radians(150.0)
        self._head_dev = max(-lim, min(lim, self._head_dev))
        self._head_pub.publish(Float32(data=float(math.degrees(self._head_dev))))

    def _heading_score(self, v: float, s: float, des_w: float) -> float:
        """0..1 — how well this arc unwinds the accumulated drift.
        Returns a flat 0 inside the deadband so straight running is untouched."""
        if (not self._head_on) or abs(self._head_dev) < self._head_db:
            return 0.0
        t_h = self._horizon_m / max(abs(v), 0.05)
        w_arc = (v / self._wheelbase) * math.tan(s)
        dev_pred = self._head_dev + (w_arc - des_w) * t_h
        return 1.0 - min(abs(dev_pred) / self._head_norm, 1.0)

    # ══════════════════════════════════════════════════════════════════
    #  SHARED DRIVER-SKILL LAYER
    # ══════════════════════════════════════════════════════════════════
    #  Everything below used to live inside _plan_dwa. That meant switching
    #  the PLANNER DISPATCH line silently dropped the cmd fade, the early-turn
    #  proximity term, the commitment filter, heading memory, the post-escape
    #  exit, the approach easing, the pinned-at-lock stall signal AND the
    #  VFH+T trap-exit bias — the other six ran the original 2024 logic.
    #
    #  Now the skills live here and each planner plugs into them THROUGH ITS
    #  OWN COMPUTATION. The seven still disagree about *how* to decide; they
    #  no longer disagree about how much they are allowed to know.
    # ══════════════════════════════════════════════════════════════════
    def _situation(self, des_v: float, des_w: float) -> Tuple[float, float, float]:
        """How blocked is the direction the Jetson actually asked for?

        Returns (des_steer, des_clear, fade). `fade` is 1.0 when the commanded
        heading is open and falls toward `fade_floor` as it closes up. Every
        planner uses it to decide how literally to take the command — they
        differ only in where they apply it."""
        des_steer = self._desired_steer(des_v, des_w)
        _, _, des_clear = self._rollout(max(abs(des_v), self._min_drive), des_steer)
        frac = des_clear / (self._horizon_m * 0.7)
        fade = max(self._fade_floor, min(1.0, frac))
        return des_steer, des_clear, fade

    def _bias(self, steer: float, clear01: float, v: float,
              des_w: float, now_t: float) -> float:
        """Side commitment + heading memory + post-escape exit + VFH+T trap
        exit, as one number that any planner can add to its own preference.

        The first two terms are multiplied by clearance, so a bias can break a
        tie between two safe arcs but can never argue the car into hugging an
        obstacle. The escape and trap terms deliberately are NOT gated that
        way — the way out of a pocket runs between its walls and therefore
        always looks expensive, which is exactly why the car kept choosing the
        roomy-looking interior and orbiting."""
        b = 0.0

        # ── side commitment: do not wiggle left-right-left ──
        if self._commit_sign != 0.0 and self._commit_age < 15:
            if steer * self._commit_sign < -0.01:
                b -= self._commit_pen * clear01
            else:
                b += self._commit_bon * clear01

        # ── heading memory: unwind drift once there is room for it ──
        b += self._head_w * clear01 * self._heading_score(v, steer, des_w)

        # ── second half of the three-point turn: the proven exit ──
        if now_t < self._post_until and self._post_dir != 0.0:
            align = 1.0 - abs(steer - self._post_steer) / (2 * self._max_steer)
            if self._esc_count >= 2:
                # After two escapes the car keeps re-entering the same pocket.
                # Ungated and at 3x so it beats w_prox, which is what rewards
                # arcs pointing into the deceptively clear pocket interior.
                b += self._post_w * 3.0 * align
            else:
                b += self._post_w * clear01 * align

        # ── VFH+T trap exit (Neamah, Elek & Korondi 2024) ──
        # _detect_trap() runs once per cycle in _plan_loop and is therefore
        # already shared; only this bias was DWA-only.
        if (self._trap_exit is not None
                and (now_t - self._trap_exit_t) < 10.0
                and abs(self._trap_exit) < math.pi * 0.6):
            clamped = max(-self._max_steer, min(self._max_steer, self._trap_exit))
            b += self._trap_w * (1.0 - abs(steer - clamped) / (2 * self._max_steer))
        return b

    def _commit_update(self, steer: float):
        """Lock in a turn direction once it is decisive; let it fade when
        running straight."""
        if abs(math.degrees(steer)) > 8.0:
            new_sign = 1.0 if steer > 0 else -1.0
            if new_sign == self._commit_sign:
                self._commit_age = max(0, self._commit_age - 1)   # reinforce
            else:
                self._commit_sign, self._commit_age = new_sign, 0  # fresh
        else:
            self._commit_age += 1

    def _finish(self, des_v: float, steer: float, max_cost: int,
                clear_dist: float, pts) -> Tuple[float, float]:
        """Shared output shaping — speed law, approach easing, stiction floor,
        steering slew limit, and the pinned-at-lock signal the stall manager
        needs. _plan_dwa used to be the only planner that eased off on
        approach or that could report being cornered."""
        v = self._speed_law(des_v, max_cost)
        if clear_dist < self._appr_m:
            frac = self._appr_min + (1.0 - self._appr_min) * \
                   (clear_dist / max(self._appr_m, 1e-3))
            v *= max(self._appr_min, min(1.0, frac))
        v = max(v, self._min_drive)

        # ── steering slew limit ──
        # The servo has a physical slew rate; a 55 deg flip inside one 200 ms
        # cycle just makes it buzz. The limited arc is RE-VALIDATED, because
        # validating an arc you then do not drive is how planners quietly
        # collide. Safety wins over smoothness: if the limited steer is
        # lethal, the full one goes through.
        if self._slew > 0.0:
            prev = getattr(self, "_last_steer", None)
            if prev is not None:
                lim = self._slew / max(self._plan_rate, 1e-3)
                if abs(steer - prev) > lim:
                    trial = prev + math.copysign(lim, steer - prev)
                    _, tc, tcd = self._rollout(max(v, self._min_drive), trial)
                    if tc < self._lethal:
                        steer, clear_dist = trial, tcd
            self._last_steer = steer

        # Consumed by _plan_loop's stall detector. Replaces _dwa_pinned.
        self._planner_pinned = (abs(steer) >= 0.9 * self._max_steer
                                and clear_dist < 0.45)
        self._commit_update(steer)
        self._publish_best(pts)
        return v, steer

    def _score_arc(self, v: float, steer: float, des_steer: float,
                   des_v_c: float, fade: float, des_w: float, now_t: float):
        """The common currency: evaluate one arc. Returns
        (score, pts, max_cost, clear_dist) or None if lethal.

        Planners that rank arcs (DWA, Lattice) use it directly; planners whose
        method produces a single answer (APF, FGM) use _rollout to validate and
        _bias to break ties, which keeps their character intact."""
        pts, max_c, clear_dist = self._rollout(v, steer)
        if max_c >= self._lethal:
            return None
        clear01 = 1.0 - max_c / 100.0
        sim = 1.0 - abs(steer - des_steer) / (2 * self._max_steer) \
                  - 0.5 * abs(v - des_v_c) / self._max_mps
        prox = min(clear_dist / self._horizon_m, 1.0)
        score = (self._w_cmd * fade * sim
                 + self._w_clear * clear01
                 + self._w_prox * prox
                 + self._w_speed * abs(v) / self._max_mps
                 + self._bias(steer, clear01, v, des_w, now_t))
        return score, pts, max_c, clear_dist

    # ══════════════════════════════════════════════════════════════════
    def _plan_loop(self):
        self._last_plan_t = time.monotonic()   # a2r_v5
        self._plan_loop_raw()

    def _plan_loop_raw(self):
        now = time.monotonic()
        with self._lock:
            des_v, des_w = self._des_v, self._des_w
            fresh = (now - self._last_cmd_t) <= self._timeout
            pet_tok = self._pet_tokens if (now - self._pet_t) < 2.0 else set()
        jetson_first = (self._arb_mode == "jetson_first")
        lost_mode = False
        # 1st choice: the follow goal (follow_manager -> /follow/goal)
        gcmd = self._goal_cmd(now) if jetson_first else None
        # a2r_pi_update2: a planned multi-point turn owns the output while it runs
        if jetson_first and self._mnv_on:
            if self._state == ST_MANEUVER:
                if self._mnv_step(now):
                    return
            elif self._state == ST_FOLLOW:
                _why = self._mnv_need(now)
                if _why and self._mnv_begin(now, _why):
                    return
        if jetson_first and self._face_step(now):
            return                                   # facing the owner this cycle
        self._goal_mode = gcmd is not None
        if self._goal_mode:
            self._lost_since = None                   # a2r_merge
            des_v, des_w = gcmd
            fresh = True
            self._last_goal_t = now
            with self._lock:
                _g = self._goal
            if _g is not None:
                self._last_goal_d = math.hypot(_g[0], _g[1])
                if self._last_goal_d > 0.05:
                    self._last_goal_b = math.atan2(_g[1], _g[0])
        elif jetson_first and now - self._last_goal_t < self._goal_hold:
            # The goal skipped a beat. Do NOT hand the car to the Jetson's
            # direct /cmd_vel (a different steering law): coast on the goal's
            # memory through the owner-lost continuation below instead.
            des_v, des_w, fresh = 0.0, 0.0, False
        elif jetson_first and not self._jetson_fallback:
            # Goal is the only authority: FollowMe's /cmd_vel is ignored.
            # No goal -> owner-lost continuation (memory), then stop and wait.
            des_v, des_w, fresh = 0.0, 0.0, False
        if jetson_first:
            # An Ackermann car cannot turn on the spot, so a command with no
            # forward speed is a stop (FollowMe's hard stop sends (0, w)).
            moving = des_v >= 1e-3
        else:
            moving = abs(des_v) >= 1e-3 or abs(des_w) >= 1e-3
        stop_fsm = bool(pet_tok & self._stop_states)
        lost_fsm = bool(pet_tok & self._lost_states)

        # ── ESCAPE IN PROGRESS: finish it ────────────────────────────────
        # This used to run AFTER the stop logic, so the escape was cancelled
        # the moment the Jetson paused. Search patterns do exactly that - zero
        # commands between arcs, and "NO_OWNER" in the state string, which is
        # a stop_state - which is why search mode never managed to reverse.
        # A three-point turn is a short, collision-monitored manoeuvre; only
        # an explicit stop, or the owner standing at the stand-off, ends it.
        if self._state in (ST_PRESTEER, ST_REVERSE):
            owner_here = self._goal_mode and abs(des_v) < 1e-3
            explicit_stop = stop_fsm and not lost_fsm
            if owner_here or explicit_stop:
                self._state = ST_FOLLOW
                self._last_rev_end = now
            elif self._run_escape(now):
                return

        if jetson_first:
            # ── WHO IS IN CHARGE THIS CYCLE ──────────────────────────────
            #  TRACK         fresh, moving Jetson command -> it is the
            #                authority; remember it in case the owner is lost
            #  LOST-CONTINUE owner lost (Jetson silent, or its FSM says
            #                FIND/SEARCH/LOST) -> keep going on the last
            #                command, slower, turn fading out, obstacles
            #                still avoided, for at most lost_continue_s
            #  STOP          owner reached (Jetson sends zeros while
            #                following), FSM says IDLE/COMPLETED, or the
            #                continuation ran out -> wait for the Jetson
            if fresh and moving:
                # The Jetson only sends motion when it wants motion (search
                # arcs, voice "circular"...), so a moving command is obeyed
                # whatever the FSM word is - with obstacle avoidance.
                self._mem_v, self._mem_w, self._mem_t = des_v, des_w, now
            else:
                # goal fresh but zero = standing at the stand-off: reached
                owner_lost = (not self._goal_mode) and ((not fresh) or lost_fsm)
                age = now - self._mem_t
                if (owner_lost and not stop_fsm
                        and self._mem_t > 0.0 and age < self._lost_s):
                    lost_mode = True
                    fade = math.exp(-age / self._lost_decay)
                    des_v = self._mem_v * self._lost_scale
                    # scale w with v so the remembered turn keeps its radius,
                    # then fade it so the car straightens instead of circling
                    des_w = self._mem_w * self._lost_scale * fade
                elif (owner_lost and not stop_fsm and self._search_s > 0.0
                        and self._last_goal_t > 0.0
                        and self._last_goal_d > 0.3      # owner was NOT right at the stand-off
                        and (now - self._last_goal_t) < self._lost_s + self._search_s):
                    # SEARCH: slow arc toward the side the owner was last seen
                    lost_mode = True
                    turn = 1.0 if self._last_goal_b >= 0.0 else -1.0
                    des_v = self._goal_turn_v
                    des_w = turn * des_v * math.tan(0.8 * self._max_steer) / self._wheelbase
                else:
                    # a2r_merge: owner lost with a wall right in front -> back off and look again
                    # (the lidar legs / camera / phone get a view; follow_manager's search then
                    # sends the car toward the last-seen position). It used to just stand there.
                    if owner_lost and not stop_fsm and self._lost_backoff(now):
                        return
                    if not owner_lost or stop_fsm:
                        self._lost_since = None
                        self._mem_t = 0.0          # reached / told to stop: forget
                        label = "IDLE"
                    elif self._mem_t > 0.0:
                        label = "OWNER LOST - waiting for Jetson"
                    else:
                        label = "IDLE (no Jetson command)"
                    self._set_out(0.0, 0.0)
                    self._state = ST_FOLLOW
                    self._stall_since = None
                    self._arb, self._avoid_side = "PASS", 0.0
                    self._publish_wish(0.0, 0.0)
                    self._publish_state(label)
                    return

        # Owner reached (Jetson sends zero) or tracker silent → STOP
        elif (not fresh) or (not moving):
            self._set_out(0.0, 0.0)
            self._state = ST_FOLLOW
            self._stall_since = None
            self._publish_wish(0.0, 0.0)
            self._publish_state("IDLE")
            return

        self._publish_wish(des_v, des_w)
        _, _, fwd_clear = self._rollout(max(abs(des_v), 0.1),
                                        self._desired_steer(des_v, des_w),
                                        dist=self._horizon_m)
        if fwd_clear >= self._horizon_m * 0.65:
            self._update_heading(now, des_w)
        else:
            self._head_t = now

        # ── BLOCKED BY THE COLLISION MONITOR -> REVERSE ESCAPE ─────────
        # The monitor sees the real footprint on raw lidar; if it keeps the
        # car stopped while we want to go forward, the way is blocked and
        # waiting will not help - back out and turn, then carry on.
        if (jetson_first and self._mon_block_since is not None
                and (now - getattr(self, "_mon_t", -1e9)) < 0.5      # a2r_fix_freeze: monitor alive
                and (now - self._mon_block_since) >= self._mon_esc_s
                and (now - self._last_rev_end) >= self._rev_cool):
            self._mon_block_since = None
            self.get_logger().warn(
                f"collision monitor held the car {self._mon_esc_s:.1f} s - "
                "way ahead blocked, starting reverse escape")
            self._escape_or_mnv(now)              # a2r_pi_update2: planned turn first
            return

        # ── JETSON-FIRST SAFETY FILTER ─────────────────────────────────
        # Drives the Jetson's command itself while it is safe, or the
        # smallest safe bend of it. Only when NOTHING near the Jetson's wish
        # is safe does control fall through to the full planner below, with
        # its stall / trap / escape logic. That is also the only place the
        # orbit detector runs now - a Jetson that is following an owner
        # walking in a circle is not an "orbit trap".
        if jetson_first:
            # v2.2: VFH+T every cycle, not only once stuck - a pocket ahead
            # hands over to the trap-aware planner before the car is in it
            if now - getattr(self, "_trap_chk_t", -1.0) >= 0.4:   # v2.5: a pocket does not
                self._trap_chk_t, self._trap_chk = now, self._detect_trap()  # change in 0.1 s
            _tx = self._trap_chk
            if _tx is not None:
                self._trap_exit, self._trap_exit_t = _tx, now
            res = self._jetson_first(des_v, des_w, now, lost_mode)   # harmony: owner first
            if _tx is not None and (res is None or "(clear)" not in res[2]):
                res = None       # way to the owner blocked: the trap logic takes over
            # A "clear" answer is not believed while the monitor is holding
            # the car still: hand over to the planner and its escape logic.
            if res is not None and not self._cm_blocked(now):
                v, steer, label = res
                self._stall_since = None
                self._orbit_dir = 0
                self._orbit_since = now
                self._set_out(v, steer)
                self._publish_state(label)
                return

        # ─────────────────────────────────────────────────────────────
        #  PLANNER DISPATCH
        #
        #  All eight planners live below and share the driver-skill
        #  layer. Which one runs is the `planner` parameter, settable at
        #  launch or at runtime:
        #
        #    dwa | apf | teb | vfh | pure_pursuit | fgm | lattice | rrt
        #
        #  A planner that raises is not allowed to take the robot with
        #  it: we fall back rather than stop publishing, because a car
        #  that stops receiving commands coasts, and a coasting car with
        #  a stale steering angle is worse than a badly-planned one.
        # ─────────────────────────────────────────────────────────────
        try:
            v, steer = self._planner_fn(des_v, des_w)
        except Exception as exc:                      # noqa: BLE001
            self._planner_faults += 1
            self.get_logger().error(
                f"planner '{self._planner_name}' raised "
                f"({type(exc).__name__}: {exc}) — falling back to "
                f"'{self._planner_fallback}'")
            if self._planner_faults >= 3:
                self._planner_name = self._planner_fallback
                self._planner_fn = self._planners[self._planner_fallback]
                self._planner_faults = 0
                self.get_logger().error(
                    "switched permanently to the fallback planner")
            try:
                v, steer = self._planners[self._planner_fallback](
                    des_v, des_w)
            except Exception:                         # noqa: BLE001
                v, steer = 0.0, 0.0

        _pm = String()
        _pm.data = self._planner_name
        self._planner_pub.publish(_pm)

        # ── VFH+T trap detection ──────────────────────────────────────
        # Run concavity analysis on the current scan.
        if now - getattr(self, "_trap_chk_t", -1.0) >= 0.4:      # v2.8: cached, as above
            self._trap_chk_t, self._trap_chk = now, self._detect_trap()
        trap_exit = self._trap_chk
        if trap_exit is not None:
            self._trap_exit = trap_exit
            self._trap_exit_t = now

        # ── Orbit / circular-trap detector ────────────────────────────
        # A car circling inside a pocket at high steer never triggers the
        # stall detector because DWA keeps finding "valid" arcs. Detect
        # sustained same-direction turning and force an escape.
        if abs(steer) > self._orbit_th:
            s_dir = 1 if steer > 0 else -1
            if s_dir != self._orbit_dir:
                self._orbit_dir = s_dir
                self._orbit_since = now
            elif (now - self._orbit_since) >= self._orbit_t:
                self.get_logger().warn(
                    f"Orbit trap — sustained {math.degrees(steer):+.0f}deg "
                    f"for {self._orbit_t:.0f}s, forcing escape")
                if self._stall_since is None:
                    self._stall_since = now - self._stall_time  # pre-age so escape fires NOW
                self._orbit_since = now  # reset to avoid re-triggering instantly
        else:
            self._orbit_dir = 0
            self._orbit_since = now

        # ── Stall detection → trigger reverse escape ──────────────────
        # NOTE: _speed_law() floors v at min_drive_speed whenever a path
        # exists at all, so v is essentially NEVER ~0 while DWA is active.
        # That made the old `abs(v) < 1e-3` check dead code: the car could
        # sit pinned at max steer, re-fighting the same obstacle every
        # cycle, indefinitely — which is what reads as "indecisive" wag on
        # the servo (candidate winner hopping by one steer bucket cycle to
        # cycle while never actually opening up). Treat "pinned at lock
        # with tight clearance" as blocked too.
        cm_block = self._cm_blocked(now)
        blocked = (abs(v) < 1e-3) or getattr(self, "_planner_pinned", False) or cm_block
        if blocked:
            if self._stall_since is None:
                self._stall_since = now
            # A hard front block halves the patience, but NEVER bypasses it.
            # Letting _front_blocked() trigger instantly meant a single
            # corrupted scan could launch an entire manoeuvre.
            need = 0.0 if cm_block else \
                self._stall_time * (0.5 if self._front_blocked() else 1.0)
            stalled = (now - self._stall_since) >= need
            can_escape = (now - self._last_rev_end) >= self._rev_cool
            self._good_since = 0.0
            if stalled and can_escape:
                self._escape_or_mnv(now)          # a2r_pi_update2: planned turn first
                return
        else:
            self._stall_since = None
            if self._good_since == 0.0:
                self._good_since = now
            elif now - self._good_since > 12.0 and self._esc_count:
                self._esc_count = 0      # driving cleanly again — forget the streak
                self._last_esc_sign = 0.0
                self._trap_exit = None   # no longer trapped

        self._set_out(v, steer)
        deg = math.degrees(self._head_dev)
        if abs(self._head_dev) >= self._head_db:
            self._publish_state(f"FOLLOW  drift {deg:+.0f}deg")
        else:
            self._publish_state("FOLLOW")

    def _lost_backoff(self, now: float) -> bool:
        """a2r_merge: owner lost and the car is nose-to-wall: reverse lost_backoff_m on the lock
        that swings the nose toward where the owner was last seen. True if it started one."""
        if not self._bo_on:
            return False
        if self._lost_since is None:
            self._lost_since = now
        if now - self._lost_since < self._bo_wait or now < self._bo_next:
            return False
        if self._scan_pts is None or len(self._scan_pts) == 0:
            return False
        room, _ = self._forward_freedom(0.0, 0.0, 0.0, 0.08)
        if room >= self._bo_front:
            return False                             # not at a wall: wait for the search goal
        self._bo_next = now + self._bo_cool
        side = 1.0 if self._last_goal_b >= 0.0 else -1.0
        lock = self._lock_frac * self._max_steer
        for st in (-side * lock, 0.0, side * lock):  # reverse with lock AWAY = nose TOWARD the owner
            sr, xr, yr, yawr = self._car.poses(st, -self._bo_m, 0.04)
            if math.isfinite(self._car.first_hit(self._scan_pts, sr, xr, yr, yawr, self._fp_margin)):
                continue
            self._state = ST_PRESTEER
            self._rev_steer, self._rev_dist = float(st), self._bo_m
            self._post_steer = self._esc_exit = 0.0
            self._presteer_until = now + max(self._pre_s, self._esc_rev_delay)
            self._last_rev_end = now
            self.get_logger().info(
                f"owner lost, wall {room:.2f} m ahead - backing off {self._bo_m:.2f} m at "
                f"{math.degrees(st):+.0f} deg to look for them (last seen "
                f"{'left' if side > 0 else 'right'})")
            self._publish_state("LOST-BACKOFF")
            return True
        return False

    def _run_escape(self, now: float) -> bool:
        """PRE-STEER -> REVERSE -> exit. True while the manoeuvre owns the output."""
        # ── PRE-STEER: swing the servo to lock BEFORE moving ────────────
        # Motor idle, wheels turning - what a driver does instinctively.
        if self._state == ST_PRESTEER:
            if now < self._presteer_until:
                self._set_out(0.0, self._rev_steer)
                self._publish_state("PRE-STEER")
                return True
            self._state = ST_REVERSE
            # Duration from the distance the lookahead chose (a time BACKSTOP since a2r_merge:
            # the reverse now ends on measured distance - odometry / IMU heading)
            dist = getattr(self, "_rev_dist", 0.6)
            self._rev_until = now + max(0.5, dist / max(self._rev_speed, 0.05)) + self._esc_rev_delay
            self._cm_since = None
            od = self._mnv_odom
            self._rev_odom0 = od if (od is not None and now - od[3] < 0.3) else None
            self._rev_yaw0 = self._imu_yaw

        # ── REVERSE until the distance is covered ────────────────────────
        if self._state == ST_REVERSE:
            # The monitor also guards the reverse. If it has been zeroing it
            # (something behind the car the lidar did see), stop reversing
            # and take the exit now rather than "reversing" in place for the
            # rest of the timer.
            blocked = self._cm_blocked(now, 0.4)
            done = self._rev_travel(now)
            cut = ""
            if done >= getattr(self, "_rev_dist", 0.6):
                cut = "distance reached"
            elif done >= self._mnv_rev_cut:
                cut = self._mnv_rev_cut_why(now)     # a2r_merge: owner/goal reachable forward now
            if now < self._rev_until and not blocked and not cut:
                self._set_out(-self._rev_speed, self._rev_steer)
                self._publish_state("ESCAPE-REVERSE")
                return True
            if cut and cut != "distance reached":
                self.get_logger().info(f"reverse ended after {done:.2f} m: {cut}")
            if blocked:
                self.get_logger().warn("Reverse blocked behind the car - taking the exit early")
            self._state = ST_FOLLOW
            self._stall_since = None
            self._last_rev_end = now
            # Leave on the exit the simulation PROVED clear (it used to be
            # overwritten here with a blind "opposite lock"), falling back to
            # opposite lock when the simulation had no opinion.
            exit_s = getattr(self, "_esc_exit", 0.0)
            if abs(exit_s) < math.radians(3.0):
                exit_s = -self._rev_steer
            self._post_steer = exit_s
            self._commit_sign = 1.0 if exit_s > 0 else -1.0
            self._commit_age = 0
            self._post_dir = 1.0
            self._post_until = now + self._post_s + 1.5
            if self._esc_reset:
                self._head_dev = 0.0
        return False

    def _rev_travel(self, now) -> float:
        """a2r_merge: metres reversed in this escape (odometry, or IMU heading on the lock)."""
        od, o0 = self._mnv_odom, getattr(self, "_rev_odom0", None)
        d = 0.0
        if od is not None and o0 is not None and now - od[3] < 0.3:
            d = math.hypot(od[0] - o0[0], od[1] - o0[1])
        st = getattr(self, "_rev_steer", 0.0)
        s_imu = self._imu_progress([(-1, st, 9.9)], self._imu_yaw - getattr(self, "_rev_yaw0", 0.0))
        return max(d, s_imu or 0.0)

    # ══════════════════════════════════════════════════════════════════
    #  JETSON-FIRST ARBITRATION
    # ══════════════════════════════════════════════════════════════════
    def _slew_to(self, target: float) -> float:
        """Servo slew limit, same rate the planners use."""
        cur = float(getattr(self, "_exec_st", 0.0))
        if self._slew <= 0.0:
            return target
        lim = self._slew / max(self._plan_rate, 1e-3)
        return cur + max(-lim, min(lim, target - cur))

    def _ahead_cost(self, pts, max_c: int, c0: int) -> int:
        """Cost that counts for choosing a path: only where the path gets
        CLOSER to obstacles than the robot already is (cost above the cost at
        the robot's own position c0), at full value. A wall behind or an
        object beside the car raises c0 but not the cost of a path that
        moves away from it, so it no longer pushes the car off the path to
        the owner. Anything the car would hit (lethal) always counts."""
        if max_c >= self._lethal:
            return max_c
        worst = 0
        costs = getattr(self, "_roll_costs", None)
        if costs is None or len(costs) != len(pts):
            costs = [self._cost_at(x, y) for (x, y) in pts]
        for c in costs:
            if c > c0 and c > worst:
                worst = c
        return worst

    def _jetson_first(self, des_v: float, des_w: float, now: float,
                      lost: bool) -> Optional[Tuple[float, float, str]]:
        """Shared control with the Jetson as the authority.

        1. PASS  - roll out the Jetson's own arc over a speed-dependent
                   lookahead. Clear (no inscribed cell, cost < pass_cost_max)
                   -> drive it unchanged. No drift correction, no bias.
        2. AVOID - otherwise pick, among arcs that are safe over the
                   lookahead, the one CLOSEST to the Jetson's steering
                   (a little extra weight on lower cost, and a penalty for
                   switching the dodge side mid-dodge so it does not wag).
                   The Jetson's own arc is always a candidate, so the moment
                   it is safe again the robot is back on it; the PASS label
                   returns once it has stayed clear for clear_hold_s.
        3. None  - no safe arc near the wish: the caller hands over to the
                   full planner and its escape logic.

        Forward motion only. A Jetson asking for v <= 0 (reverse / turn on
        the spot, which an Ackermann car cannot do) goes to the planner."""
        if des_v <= 0.0:
            return None
        tag = ("LOST-CONTINUE" if lost else
               "FOLLOW-GOAL" if self._goal_mode else "FOLLOW")
        v_des = min(des_v, self._max_mps)
        steer_j = self._desired_steer(v_des, des_w)
        look = max(self._look_min, min(self._horizon_m, 0.35 + v_des * self._safety_t))
        # cost where the robot already is: obstacles it is merely NEXT to
        # (behind / beside) must not steer it away from the owner
        c0 = self._cost_at(0.0, 0.0)
        pts_j, c_j_raw, clr_j = self._rollout(v_des, steer_j, dist=look)
        c_j = self._ahead_cost(pts_j, c_j_raw, c0)
        arc_ok = (c_j < self._pass_cost) and (clr_j >= look - 1e-6)

        if arc_ok:
            if self._arb == "AVOID":
                if self._arb_clear_since == 0.0:
                    self._arb_clear_since = now
                if now - self._arb_clear_since >= self._clear_hold:
                    self._arb, self._avoid_side = "PASS", 0.0
            if self._arb == "PASS":
                v = self._speed_law(v_des, c_j)
                self._publish_best(pts_j)
                return v, self._slew_to(steer_j), f"{tag}  (clear)"
        else:
            self._arb = "AVOID"
            self._arb_clear_since = 0.0

        look_min = max(0.45, 0.6 * look)
        cands = [steer_j] + [-self._max_steer + 2.0 * self._max_steer * k /
                             (self._avoid_n - 1) for k in range(self._avoid_n)]
        best = None
        for s in cands:
            pts, c_raw, clr = self._rollout(v_des, s, dist=look)
            c = self._ahead_cost(pts, c_raw, c0)
            if c >= min(self._avoid_cmax, self._lethal) or clr < look_min:
                continue
            dev = (s - steer_j) / self._max_steer
            side = 0.0 if abs(dev) < 1e-6 else (1.0 if dev > 0 else -1.0)
            score = abs(dev) + self._avoid_wc * c / 100.0
            if self._avoid_side and side and side != self._avoid_side:
                score += self._avoid_wf
            if best is None or score < best[0]:
                best = (score, s, c, clr, pts, side)
        if best is None:
            return None

        _, s, c, clr, pts, side = best
        if side:
            self._avoid_side = side
        v = self._speed_law(v_des, c) * max(0.5, min(1.0, clr / look))
        v = max(v, self._min_drive)
        self._publish_best(pts)
        dev_deg = math.degrees(s - steer_j)
        return v, self._slew_to(s), f"{tag}  AVOID {dev_deg:+.0f}deg"

    def _front_blocked(self) -> bool:
        """Immediate front arc lethal at min speed → no forward candidate exists."""
        _, c, _ = self._rollout(self._min_drive, 0.0, dist=0.45, hard=0.45, ramp=False)
        return c >= self._lethal

    def _forward_freedom(self, x0, y0, yaw0, ds=0.08, max_m=None):
        """From a simulated pose, how far can the car drive forward, and on
        which steering angle? Returns (metres_of_clear_travel, best_steer).

        Swept FOOTPRINT, not the centre point, and scoring stops where the
        path leaves the local costmap: unknown space is not free space.
        Counting it as free is what made the escape prefer the longest (and
        blindest) reverse, because the far end of a 2 m reverse looks onto
        cells nobody has seen.

        max_m defaults to the escape evaluation depth rather than horizon_m,
        so a timid reverse and a decisive one do not tie at the same number."""
        if max_m is None:
            max_m = self._esc_eval
        best_d, best_s = -1.0, 0.0
        for fs in np.linspace(-self._max_steer, self._max_steer, self._esc_n):
            sa, xa, ya, ya_w = self._car.poses(float(fs), max_m, ds,
                                              x0=float(x0), y0=float(y0), yaw0=float(yaw0))
            hit = self._car.first_hit(self._scan_pts, sa, xa, ya, ya_w, self._fp_margin)
            sd = np.abs(sa[1:])
            c, inside = self._costs_at(xa[1:], ya[1:])
            stop = len(sd)
            if math.isfinite(hit):
                stop = min(stop, int(np.searchsorted(sd, hit)))
            off = np.flatnonzero(~inside)
            if len(off):
                stop = min(stop, int(off[0]))
            # Near-field weighting: clear space in the first metre or so is
            # what makes an escape usable NOW, so two reverses that both reach
            # open ground are still ranked by how open their IMMEDIATE path is.
            score = 0.0
            if stop > 0:
                score = float(np.sum(ds * (1.0 - c[:stop] / 100.0)
                                     * np.exp(-sd[:stop] / 0.7)))
            if score > best_d:
                best_d, best_s = score, float(fs)
        # metres-equivalent: a fully clear 0.7 m path scores ~0.7
        return max(best_d, 0.0), best_s

    # ══════════════════════════════════════════════════════════════════
    #  VFH+T-INSPIRED TRAP DETECTION  (Neamah, Elek & Korondi 2024)
    # ══════════════════════════════════════════════════════════════════
    def _detect_trap(self):
        """a2r_test: VFH+T (always on) + logging around the original detector."""
        ex = self._detect_trap_raw()
        try:
            self._vfht_pub.publish(Bool(data=ex is not None))
        except Exception:
            pass
        return ex

    def _detect_trap_raw(self):
        """Is the car inside a concave pocket? (VFH+T, Neamah, Elek & Korondi
        2024). See detect_trap_core for the v2.8 fixes. Returns the exit
        bearing in base_link radians, steered toward the owner when a goal is
        fresh, or None."""
        ang = getattr(self, "_scan_ang", None)
        rng = getattr(self, "_scan_rng", None)
        hit = getattr(self, "_scan_hit", None)
        if ang is None or rng is None or hit is None or len(hit) != len(ang):
            return None
        exit_ang = detect_trap_core(ang, rng, hit, self._trap_group_dist, self._trap_span,
                                    self._trap_frac, self._trap_range,
                                    far_range=self._scan_max * 0.4)
        if exit_ang is None:
            return None
        _g = self._goal                       # harmony: escape toward the owner
        if _g is not None and time.monotonic() - _g[2] < 2.0:
            far = ~hit | (rng > self._scan_max * 0.4)
            exit_ang = goal_aware_exit(ang, far, math.atan2(_g[1], _g[0]), exit_ang)
        return exit_ang

    # ══════════════════════════════════════════════════════════════════
    #  DRIVER-STYLE REVERSE-ESCAPE
    # ══════════════════════════════════════════════════════════════════
    def _begin_escape(self, now: float):
        """Expert-driver escape.

        Rather than backing up on the first arc that happens to be clear, every
        candidate (reverse angle, reverse distance) pair is simulated, and from
        each resulting pose we measure how far the car could then drive FORWARD.
        The reverse that opens the most forward room wins. That is what a driver
        does when they glance over their shoulder, pick one decisive lock, and
        come out of the space in a single movement — instead of shuffling back
        and forth discovering the same dead end repeatedly.
        """
        ds = 0.08

        # How much forward room do we have right now? Anything we choose must
        # beat this by escape_min_gain_m or the manoeuvre was pointless.
        room_now, _ = self._forward_freedom(0.0, 0.0, 0.0, ds)
        # a2r_merge: weigh each reverse by where the OWNER is (lidar legs / phone / goal): prefer the
        # lock that swings the nose toward them, never back into their keep zone
        own = self._owner_now(now)
        if own is None and self._last_goal_t > 0.0 and now - self._last_goal_t < 5.0:
            own = (1.5 * math.cos(self._last_goal_b), 1.5 * math.sin(self._last_goal_b))   # last seen side
        own_e0 = abs(math.atan2(own[1], own[0])) if own is not None else 0.0

        best = None
        if self._esc_look:
            for rev_dist in self._esc_dists:
                for rs in np.linspace(-self._max_steer, self._max_steer, self._esc_n):
                    rs = float(rs)
                    # Reverse on the pre-steered lock: exact kinematics, whole
                    # footprint. The old check only asked whether the CENTRE
                    # landed on a lethal cell, so it could not see the tail,
                    # and a car resting near a wall found every reverse
                    # "blocked" at its own starting cell.
                    sr, xr, yr, yawr = self._car.poses(rs, -rev_dist, 0.04)
                    if math.isfinite(self._car.first_hit(
                            self._scan_pts, sr, xr, yr, yawr, self._fp_margin)):
                        continue
                    _, inside = self._costs_at(xr, yr)
                    if not inside.all():
                        continue                     # never back into the unknown
                    x, y, yaw = float(xr[-1]), float(yr[-1]), float(yawr[-1])
                    own_gain = 0.0
                    if own is not None:
                        if math.hypot(own[0] - x, own[1] - y) < 0.6:
                            continue                 # would back into the owner
                        own_gain = own_e0 - abs(_wrap(math.atan2(own[1] - y, own[0] - x) - yaw))
                    room, fwd_s = self._forward_freedom(x, y, yaw, ds)
                    # What actually gets an Ackermann car out of a pocket is
                    # rotation, so score the nose swing alongside the room.
                    # Rounding the room lets rotation and commitment break
                    # near-ties. Expert-driver rule: the SHORTEST reverse that
                    # opens the most room AND swings the nose decisively.
                    decisive = abs(yaw) >= self._esc_min_yaw
                    # ── U-pocket trap breaker ──────────────────────────
                    # 2+ escapes the same way and still stuck: the geometry is
                    # funnelling us back in, so penalise the same side.
                    room_adj = room + 0.25 * own_gain        # ~0.4 m of room per 90 deg toward the owner
                    if (self._alternate and self._esc_count >= 2
                            and self._last_esc_sign != 0.0):
                        same = math.copysign(1.0, rs) == self._last_esc_sign
                        if self._esc_count >= 3:
                            room_adj = room * (0.15 if same else 1.3)
                        else:
                            room_adj = room * (0.4 if same else 1.2)
                    ndigits = 2 if room_now < 0.5 else 1
                    key = (round(room_adj, ndigits), decisive, -rev_dist, abs(rs))
                    if best is None or key > best[0]:
                        best = (key, rs, rev_dist, fwd_s, room, abs(yaw))

        if best is None:
            # Lookahead disabled or every reverse blocked — fall back to the
            # simple "largest lock towards the open side" rule.
            open_side = self._side_openness()
            lock = self._lock_frac * self._max_steer
            pref = -math.copysign(lock, open_side if abs(open_side) > 1e-3 else 1.0)
            if self._alternate and (self._esc_count % 2 == 1):
                pref = -pref
            for cand in (pref, -pref, 0.5 * pref, -0.5 * pref, 0.0):
                cand = max(-self._max_steer, min(self._max_steer, float(cand)))
                _, c, clr = self._rollout(-self._rev_speed, cand, dist=0.6,
                                          hard=0.25, ramp=False)
                if c < self._lethal:
                    back = max(0.25, min(0.6, clr - 0.05))
                    best = ((0.0, False, -back, abs(cand)), cand, back, 0.0, 0.0, 0.0)
                    break

        if best is None:
            self._set_out(0.0, 0.0)
            self._publish_state("BOXED-IN (STOP)")
            self.get_logger().warn("Boxed in: forward and reverse blocked — stopping",
                                   throttle_duration_sec=2.0)
            return

        _, rev_steer, rev_dist, fwd_steer, room_after, yaw_gain = best

        # How long have we been stuck here? The longer it is, the less extra
        # room we insist on — and after force_after_s we take the best reverse
        # no matter what, because sitting still solves nothing.
        if self._hold_since == 0.0:
            self._hold_since = now
        held = now - self._hold_since
        # Monitor already holding us still: reverse now, waiting only loses time.
        force_after_s = self._esc_force_s          # a2r_merge: was 1.5 s of staring at the wall
        req_gain = self._esc_gain * max(0.0, 1.0 - held / force_after_s)
        forced = self._cm_blocked(now)

        if (self._esc_look and not forced and room_after < room_now + req_gain
                and held < force_after_s):
            self._set_out(0.0, 0.0)
            self._publish_state("BOXED-IN (STOP)")
            self.get_logger().warn(
                f"Reverse gains little (room {room_now:.2f}m -> "
                f"{room_after:.2f}m, held {held:.1f}s) — waiting",
                throttle_duration_sec=3.0)
            return

        self._esc_count += 1
        self._last_esc_sign = 1.0 if rev_steer > 0 else -1.0
        self._hold_since = 0.0
        self._state = ST_PRESTEER
        self._rev_steer = rev_steer
        self._rev_dist = rev_dist
        self._post_steer = fwd_steer          # the exit the simulation proved
        self._esc_exit = fwd_steer
        self._presteer_until = now + self._pre_s
        self.get_logger().info(
            f"Stall -> PRE-STEER {math.degrees(rev_steer):+.0f}deg, back "
            f"{rev_dist:.2f}m, turns {math.degrees(yaw_gain):.0f}deg, opens "
            f"{room_now:.2f}m -> {room_after:.2f}m, exit "
            f"{math.degrees(fwd_steer):+.0f}deg (attempt {self._esc_count})")

    # ══════════════════════════════════════════════════════════════════
    #  a2r_pi_update2: MULTI-POINT TURN  (maneuver.py)
    #  Pure pursuit only drives forward and the stall escape is ONE reverse
    #  arc, so in a corridor narrower than a full-lock U-turn (1.46 m) the car
    #  could never turn round. This plans the K-turn a driver would make, on
    #  the scan the collision monitor sees (/scan/merged, owner KEPT), and
    #  drives it one leg at a time on /cmd_vel_escape, re-checking every
    #  cycle. collision_monitor still guards every metre of it.
    # ══════════════════════════════════════════════════════════════════
    def _owner_cb(self, msg):
        """a2r_merge: the owner's lidar-leg track (base_link), for owner-aware turning."""
        if msg.header.frame_id and msg.header.frame_id != self._frame:
            return                                    # only base_link points are usable as they are
        self._owner_pt = (float(msg.point.x), float(msg.point.y), time.monotonic())

    def _owner_now(self, now, goal=None):
        """Where the owner is now (base_link): the lidar-leg track if fresh, else what the follow
        goal says (goal yaw = owner bearing; the goal sits follow_distance short of them)."""
        o = self._owner_pt
        if o is not None and now - o[2] <= self._owner_fresh:
            return o[0], o[1]
        g = self._goal if goal is None else goal
        if g is None or len(g) < 4 or now - g[2] > self._goal_to:
            return None
        d = math.hypot(g[0], g[1]) + self._mnv_owner_d
        return d * math.cos(g[3]), d * math.sin(g[3])

    def _fwd_turn_free(self, b, pts, full=False) -> bool:
        """Could a full-lock FORWARD turn toward bearing b (base_link) get past the collision
        monitor right now? (the test _mnv_need uses to decide that a K-turn is needed)
        a2r_step3 full=True (ending a reverse early): the arc must be free for the WHOLE turn plus
        0.15 m - the old test (turn minus the 30 deg tolerance) ended reverses after 0.2 m, the
        forward turn then hit the wall again and the next K-turn started (5+ shuffles)."""
        if pts is None:
            return False
        tol = 0.0 if full else self._mnv_drive_tol
        need = max(0.2, (abs(b) - tol) * self._car.radii()[1] + self._mnv_buf + (0.15 if full else 0.0))
        s, x, y, yaw = self._car.poses(math.copysign(self._max_steer, b), need, 0.02)
        return not math.isfinite(self._car.first_hit(pts, s, x, y, yaw, self._mnv_vmargin))

    def _imu_progress(self, prims, dyaw):
        """a2r_merge: arc length along `prims` that explains a measured heading change dyaw
        (IMU). None if the IMU is not trusted or the leg has straight / mixed-lock parts."""
        if not self._imu_ok or not prims:
            return None
        sg = math.copysign(1.0, prims[0][1])
        if any(abs(st) < math.radians(5.0) or math.copysign(1.0, st) != sg for _, st, _ in prims):
            return None
        left, s_out = abs(dyaw), 0.0
        for _, st, ln in prims:
            k = abs(self._car.curvature(st))
            if left >= k * ln:
                s_out += ln
                left -= k * ln
            else:
                return s_out + left / k
        return s_out

    def _mnv_odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._mnv_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, time.monotonic())

    def _mnv_scan_cb(self, msg):
        self._mnv_scan = (np.asarray(msg.ranges, dtype=np.float32), float(msg.angle_min),
                          float(msg.angle_increment), float(msg.range_min), float(msg.range_max),
                          time.monotonic())

    def _beam_trig(self, n, amin, ainc):
        """a2r_merge: cos/sin of every beam (+ lidar yaw), computed once per scan layout instead
        of ~2300 cos + sin per scan in every callback."""
        key = (n, round(amin, 7), round(ainc, 9))
        t = self._trig.get(key)
        if t is None:
            ab = amin + ainc * np.arange(n, dtype=np.float64) + self._lyaw
            t = (np.cos(ab), np.sin(ab), ab)
            if len(self._trig) > 4:
                self._trig.clear()
            self._trig[key] = t
        return t

    def _mnv_points(self):
        """Obstacles (base_link) as the collision monitor sees them, or None if that scan is stale
        (the monitor stops the car without a fresh scan anyway). a2r_merge: built once per scan."""
        sc = self._mnv_scan
        if sc is None or time.monotonic() - sc[5] > 0.5:
            return None
        if self._mnv_pts_cache[0] == sc[5]:
            return self._mnv_pts_cache[1]
        r, amin, ainc, rmin, rmax, _ = sc
        c, sn, _ = self._beam_trig(len(r), amin, ainc)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(r) & (r > max(rmin, self._mnv_rmin)) & (r < min(rmax, self._mnv_area + 0.6))
        rr = r[ok].astype(np.float64)
        pts = np.stack([self._lx + rr * c[ok], self._ly + rr * sn[ok]], axis=1)
        pts = pts[~self._car.body_mask(pts, 0.0)]
        self._mnv_pts_cache = (sc[5], pts)
        return pts

    def _mnv_goal(self, now):
        g = self._goal
        if g is None or len(g) < 4 or now - g[2] > self._goal_to:
            return None
        o = self._owner_now(now, g)                           # a2r_merge: plan keeps clear of them
        ox, oy = (o if o is not None else (None, None))
        if math.hypot(g[0], g[1]) < self._face_zone:          # at the stand-off: face the owner
            fx, fy = (o if o is not None else (self._mnv_owner_d * math.cos(g[3]),
                                               self._mnv_owner_d * math.sin(g[3])))
            return ManeuverGoal("face", fx, fy, self._mnv_face_tol, ox=ox, oy=oy)
        return ManeuverGoal("drive", g[0], g[1], self._mnv_drive_tol, self._mnv_free, ox=ox, oy=oy)

    def _mnv_need(self, now) -> str:
        """Reason to start a multi-point turn now, or ''."""
        if now < self._mnv_until:
            return ""
        g = self._goal
        if g is None or len(g) < 4 or now - g[2] > self._goal_to:
            return ""
        if math.hypot(g[0], g[1]) < self._face_zone:
            o = self._owner_now(now, g)                       # a2r_merge: live lidar-leg bearing
            ob = math.atan2(o[1], o[0]) if o is not None else g[3]
            if self._face_on and abs(ob) >= self._mnv_face_b:
                return f"owner {math.degrees(ob):+.0f} deg - behind the car"
            return ""
        b = math.atan2(g[1], g[0])
        if abs(b) <= self._goal_turn_b:
            return ""
        pts = self._mnv_points()
        if pts is None:
            return ""
        # a2r_merge: the owner walking round the car is not a wall. Their own legs used to make
        # the full-lock U-turn look "blocked" and start a K-turn (stand still, plan, shuffle):
        # the lag when the owner went behind the car. The monitor still stops for them.
        o = self._owner_pt
        if o is not None and now - o[2] <= self._owner_fresh and len(pts):
            pts = pts[np.hypot(pts[:, 0] - o[0], pts[:, 1] - o[1]) > 0.45]
        # would the full-lock forward turn toward the goal get past the collision monitor?
        if not self._fwd_turn_free(b, pts):
            return f"goal {math.degrees(b):+.0f} deg - full-lock turn blocked"
        return ""

    def _escape_or_mnv(self, now):
        """Stuck: plan a multi-point turn toward the goal first; the old one-arc escape only
        when there is no goal or no plan."""
        if (self._mnv_on and self._arb_mode == "jetson_first" and now >= self._mnv_until
                and self._mnv_begin(now, "stuck")):
            return
        self._begin_escape(now)

    def _mnv_begin(self, now, why) -> bool:
        goal, pts = self._mnv_goal(now), self._mnv_points()
        if goal is None or pts is None:
            return False
        self._mnv = {"why": why, "t0": now, "legs": 0, "plans": 0, "phase": "plan", "job": None,
                     "prims": [], "dir": 0, "fails": 0}
        self._state = ST_MANEUVER
        self._stall_since = None
        self._mnv_plan(now, goal, pts)
        self._set_out(0.0, float(getattr(self, "_exec_st", 0.0)))
        self._publish_state("MANOEUVRE planning")
        return True

    def _mnv_plan(self, now, goal, pts):
        job = {"goal": goal, "pts": pts, "res": None, "done": False, "t": now}

        def work():
            try:
                job["res"] = self._mnv_planner.plan(pts, goal, verify_pts=pts)
                job["why"] = getattr(self._mnv_planner, "last_why", "")   # a2r_fix_freeze
            except Exception as exc:                        # noqa: BLE001
                job["err"] = f"{type(exc).__name__}: {exc}"
            job["done"] = True
        self._mnv["job"], self._mnv["phase"] = job, "plan"
        self._mnv["plans"] += 1
        threading.Thread(target=work, daemon=True).start()

    def _mnv_end(self, now, ok, text):
        m = self._mnv
        self._mnv = None
        self._state = ST_FOLLOW
        self._stall_since = None
        self._last_rev_end = now                  # no old escape straight after
        self._mnv_until = now + (0.5 if ok else self._mnv_cool)
        self._set_out(0.0, float(getattr(self, "_exec_st", 0.0)))
        self._crawl_pub.publish(Bool(data=False))
        dur = now - m["t0"] if m else 0.0
        # a2r_fix_freeze: ONE call site per severity - rclpy raises "Logger severity cannot be
        # changed between calls" otherwise, which killed the planner mid-manoeuvre (frozen car)
        _msg = f"manoeuvre {'done' if ok else 'ended'}: {text} ({m['legs'] if m else 0} legs, {dur:.1f} s)"
        if ok:
            self.get_logger().info(_msg)
        else:
            self.get_logger().warn(_msg)

    def _mnv_publish_path(self, plan):
        try:
            poses = segment_poses(self._car, plan.prims, step=0.05)
            p = Path()
            p.header.frame_id = self._frame
            p.header.stamp = self.get_clock().now().to_msg()
            for x, y, th in poses:
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
                p.poses.append(ps)
            self._mnv_path_pub.publish(p)
        except Exception:                                   # noqa: BLE001 - display only
            pass

    def _mnv_start_leg(self, now):
        m = self._mnv
        leg, rest, d0 = [], list(m["prims"]), m["prims"][0][0]
        # a2r_merge: only REVERSE legs are chopped (the rear is lidar-blind beyond ~1 m); a forward
        # leg runs in one go - it is re-checked against every scan anyway, a stop costs ~1 s
        # a2r_step3: a reverse leg is driven as planned (up to the planner's proportional cap): every
        # chop was another stop + pre-steer + ESC handshake. The rest of the leg is still re-checked
        # against every scan while driving ("way changed").
        left = max(self._mnv_max_leg, float(getattr(self._mnv_planner, "last_leg_cap", 0.0))) if d0 < 0 else math.inf
        while rest and rest[0][0] == d0 and left > 1e-3:
            d, st, ln = rest.pop(0)
            if ln > left + 1e-6:
                rest.insert(0, (d, st, ln - left))
                ln = left
            leg.append((d, st, ln))
            left -= ln
        m["leg"], m["prims"], m["s"] = leg, rest, 0.0
        m["leg_len"] = sum(p[2] for p in leg)
        m["odom0"] = self._mnv_odom if (self._mnv_odom and now - self._mnv_odom[3] < 0.3) else None
        m["odom_prev"] = None
        m["yaw0"] = self._imu_yaw                    # a2r_merge: IMU heading at the leg start
        m["t_leg"], m["t_last"] = now, now
        # wheels at the leg's lock BEFORE moving: the output smoother slews the servo (150 deg/s),
        # a lock-to-lock swing takes ~0.4 s; plus the ESC brake/neutral handshake on a reversal
        swing = abs(leg[0][1] - float(getattr(self, "_exec_st", 0.0))) / self._mnv_slew
        pause = max(self._pre_s, swing + 0.1, self._esc_rev_delay if d0 != m["dir"] else 0.0)
        m["phase"], m["until"], m["dir"] = "presteer", now + pause, d0
        m["legs"] += 1

    def _mnv_rev_cut_why(self, now) -> str:
        """a2r_merge: reason to stop a reverse leg early (live owner / goal), or ''."""
        g = self._goal
        if g is None or len(g) < 4 or now - g[2] > self._goal_to:
            return ""
        o = self._owner_now(now, g)
        if math.hypot(g[0], g[1]) < self._face_zone:                       # face mode
            b = math.atan2(o[1], o[0]) if o is not None else g[3]
            return "owner now ahead" if abs(b) <= self._mnv_face_tol else ""
        b = math.atan2(g[1], g[0])
        if abs(b) <= self._mnv_drive_tol:
            return "goal now ahead"
        pts = self._mnv_points()
        if pts is not None and o is not None and len(pts):
            pts = pts[np.hypot(pts[:, 0] - o[0], pts[:, 1] - o[1]) > 0.45]
        if self._fwd_turn_free(b, pts, full=True):                         # a2r_step3
            return f"forward turn to the goal ({math.degrees(b):+.0f} deg) is free - no more reversing"
        return ""

    def _mnv_step(self, now) -> bool:
        """Run the manoeuvre. True while it owns the output."""
        m = self._mnv
        if m is None:
            self._state = ST_FOLLOW
            return False
        g = self._goal
        if g is None or now - g[2] > self._goal_hold:
            self._mnv_end(now, False, "no follow goal")
            return False
        if now - m["t0"] > self._mnv_max_s or m["legs"] > self._mnv_max_legs:
            self._mnv_end(now, False, "taking too long")
            return False
        # owner arrived right in front: the follow logic has it
        if math.hypot(g[0], g[1]) < self._face_zone and abs(g[3]) < self._mnv_face_tol:
            self._mnv_end(now, True, "owner ahead")
            return False
        ph = m["phase"]
        if ph == "plan":
            job = m["job"]
            if not job["done"]:
                self._set_out(0.0, float(getattr(self, "_exec_st", 0.0)))
                self._publish_state("MANOEUVRE planning")
                return True
            res = job["res"]
            if res is None:
                why = job.get("err") or job.get("why") or f"no route within {self._mnv_planner.budget:.1f} s"
                self._mnv_end(now, False, f"no plan ({m['why']}): {why}")
                if m["why"] == "stuck":
                    self._begin_escape(now)                # the old one-arc escape as the last resort
                    return self._state in (ST_PRESTEER, ST_REVERSE)
                return False
            if not res.prims:
                self._mnv_end(now, True, "facing the way to go")
                return False
            self.get_logger().info(
                f"manoeuvre ({m['why']}): {len(res.legs)} legs {res.summary()} - plan {res.ms:.0f} ms, "
                f"{res.expanded} nodes" + ("" if m["plans"] == 1 else f" (re-plan {m['plans'] - 1})")
                + (f" - PARTIAL: turns {res.gain_deg:.0f} deg toward the goal, re-plans after"
                   if getattr(res, "partial", False) else ""))
            self._mnv_publish_path(res)
            m["prims"] = list(res.prims)
            m["partial"] = bool(getattr(res, "partial", False))
            self._mnv_start_leg(now)
            ph = m["phase"]
        leg_tag = f"MANOEUVRE {'FWD' if m['dir'] > 0 else 'REV'} {m['legs']}"
        if ph == "presteer":
            self._crawl_pub.publish(Bool(data=True))
            self._set_out(0.0, m["leg"][0][1])
            self._publish_state(leg_tag + " pre-steer")
            if now >= m["until"]:
                m["phase"], m["t_last"] = "drive", now
            return True
        if ph == "drive":
            self._crawl_pub.publish(Bool(data=True))
            od = self._mnv_odom
            if od is not None and now - od[3] < 0.3 and m["odom0"] is not None:
                prev = m.get("odom_prev") or m["odom0"]
                m["s"] += math.hypot(od[0] - prev[0], od[1] - prev[1])
                m["odom_prev"] = od
            else:
                m["s"] += self._mnv_speed * max(0.0, now - m["t_last"])   # no odometry: dead reckoning
            m["t_last"] = now
            # a2r_merge: on a steered leg the IMU heading change is the most direct measure of how
            # far round the arc the car has got (odometry from commanded speed over-runs when the
            # wheels slip or the ESC lags). Take whichever says "further": never over-reverse.
            s_imu = self._imu_progress(m["leg"], self._imu_yaw - m["yaw0"])
            if s_imu is not None and s_imu > m["s"]:
                m["s"] = s_imu
            # remaining part of the leg from where the car is now
            s_left, rem = m["s"], []
            for d, st, ln in m["leg"]:
                if s_left >= ln:
                    s_left -= ln
                    continue
                rem.append((d, st, ln - s_left))
                s_left = 0.0
            end, why = not rem, "leg done"
            if not end and self._cm_blocked(now, 0.4):
                end, why = True, "monitor stop"
            # a2r_merge: LIVE OWNER CHECK while reversing. The plan was made from where the owner
            # was at the start; the owner moves (lidar legs / phone / goal keep updating). Stop
            # backing up as soon as reversing is no longer what gets the car to the owner:
            #   face : the owner is already (nearly) straight ahead
            #   drive: the forward full-lock turn toward the live goal is now free
            if not end and m["dir"] < 0 and m["s"] >= self._mnv_rev_cut:
                cut = self._mnv_rev_cut_why(now)
                if cut:
                    self._mnv_end(now, True, cut)
                    self._face_pause_until = now + 1.5     # do not reverse again straight away
                    return False
            if not end and now - m["t_leg"] > m["leg_len"] / self._mnv_speed * 2.5 + 2.5:
                end, why = True, "leg timeout"
            if not end:
                pts = self._mnv_points()
                if pts is None:
                    end, why = True, "no fresh scan"
                else:
                    x = y = th = 0.0
                    for d, st, ln in rem:
                        s, X, Y, TH = self._car.poses(st, d * ln, 0.02, x0=x, y0=y, yaw0=th)
                        if math.isfinite(self._car.first_hit(pts, s, X, Y, TH, self._mnv_vmargin)):
                            end, why = True, "way changed"
                            break
                        x, y, th = float(X[-1]), float(Y[-1]), float(TH[-1])
            if end:
                m["phase"], m["until"], m["end_why"] = "settle", now + self._mnv_pause, why
                if why != "leg done":
                    m["prims"] = []                         # the rest of that plan is void: re-plan
                self._set_out(0.0, rem[0][1] if rem else m["leg"][-1][1])
                self._publish_state(leg_tag + " stop")
                return True
            self._set_out(m["dir"] * self._mnv_speed, rem[0][1])
            self._publish_state(leg_tag)
            return True
        if ph == "settle":
            self._set_out(0.0, float(getattr(self, "_exec_st", 0.0)))
            if now < m["until"]:
                return True
            goal, pts = self._mnv_goal(now), self._mnv_points()
            if goal is None or pts is None:
                self._mnv_end(now, False, "goal or scan lost")
                return False
            if (m["prims"] and not m.get("partial") and m["legs"] >= 1
                    and len({p[0] for p in m["prims"]}) == 1
                    and sum(p[2] for p in m["prims"]) < self._mnv_planner.min_leg - 1e-6):
                # a2r_merge: only a stub of a last leg left (the search's stop-buffer step): a stop,
                # pre-steer and ESC handshake (~1 s) for 0.1 m is not worth it - follow takes over
                self._mnv_end(now, True, "last leg is a stub - handing back to follow")
                return False
            if m["prims"]:
                # carry on with the plan if it is still clear from where the car really is
                x = y = th = 0.0
                ok = True
                for d, st, ln in m["prims"]:
                    s, X, Y, TH = self._car.poses(st, d * (ln + self._mnv_buf), 0.02, x0=x, y0=y, yaw0=th)
                    if math.isfinite(self._car.first_hit(pts, s, X, Y, TH, self._mnv_vmargin)):
                        ok = False
                        break
                    s, X, Y, TH = self._car.poses(st, d * ln, 0.02, x0=x, y0=y, yaw0=th)
                    x, y, th = float(X[-1]), float(Y[-1]), float(TH[-1])
                if ok:
                    self._mnv_start_leg(now)
                    return True
            self._mnv_plan(now, goal, pts)                  # re-plan from here (also checks "done")
            self._publish_state("MANOEUVRE planning")
            return True
        self._mnv_end(now, False, "unknown phase")
        return False

    def _set_out(self, v: float, steer: float):
        # Remember the executed command — the heading model integrates it
        # when no usable gyro is available.
        self._exec_v, self._exec_st = v, steer
        v_n = max(-self._max_cmd, min(self._max_cmd, v / self._max_mps))
        s_n = max(-1.0, min(1.0, steer / self._max_steer))
        with self._lock:
            self._out_v, self._out_s = v_n, s_n

    # ══════════════════════════════════════════════════════════════════
    #  1) DWA — Dynamic Window Approach                        [ACTIVE]
    # ══════════════════════════════════════════════════════════════════
    def _plan_dwa(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Samples the (v, steer) dynamic window and scores every arc.

        Behaviour is unchanged — the skills it already had now come from the
        shared layer instead of being inlined, so the other six get them too.
        The one addition is the fallback sweep when the commitment filter has
        rejected everything."""
        now_t = time.monotonic()
        des_steer, des_clear, fade = self._situation(des_v, des_w)
        des_v_c = max(-self._max_mps, min(self._max_mps, des_v))

        sign = 1.0 if des_v_c >= 0 else -1.0
        v_lo = max(self._min_drive, 0.5 * abs(des_v_c))
        vs = sign * np.linspace(v_lo, max(v_lo, abs(des_v_c)), self._n_v)
        ss = np.linspace(-self._max_steer, self._max_steer, self._n_s)

        # Hard commitment filter: only while the command is faded (an obstacle
        # is actually in the way) and the commitment is fresh.
        use_filter = (self._commit_sign != 0.0 and self._commit_age < 15
                      and fade < 0.95)

        markers = MarkerArray()
        mid = 0
        best = None
        for v in vs:
            for s in ss:
                r = self._score_arc(float(v), float(s), des_steer, des_v_c,
                                    fade, des_w, now_t)
                if r is None:
                    markers.markers.append(self._traj_marker([], mid, lethal=True))
                    mid += 1
                    continue
                if use_filter and s * self._commit_sign < -0.01:
                    markers.markers.append(self._traj_marker(r[1], mid, lethal=True))
                    mid += 1
                    continue
                if best is None or r[0] > best[0]:
                    best = (r[0], float(v), float(s), r[1], r[2], r[3])
                markers.markers.append(self._traj_marker(r[1], mid, lethal=False))
                mid += 1
        self._trajs_pub.publish(markers)

        if best is None and use_filter:
            # Filter starved the search — drop the commitment and re-sweep.
            self._commit_sign, self._commit_age = 0.0, 99
            for v2 in vs:
                for s2 in ss:
                    r = self._score_arc(float(v2), float(s2), des_steer,
                                        des_v_c, fade, des_w, now_t)
                    if r is not None and (best is None or r[0] > best[0]):
                        best = (r[0], float(v2), float(s2), r[1], r[2], r[3])

        if best is None:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0

        if fade < 0.95:
            self.get_logger().info(
                f"Obstacle ahead — cmd weight faded to {fade:.2f} "
                f"(des_clear={des_clear:.2f}m), chose steer "
                f"{math.degrees(best[2]):+.0f}deg", throttle_duration_sec=1.0)

        _, bv, bs, bpts, bcost, bclear = best
        v_out, steer = self._finish(des_v, bs, bcost, bclear, bpts)
        return v_out * (1 if bv >= 0 else -1), steer

    # ══════════════════════════════════════════════════════════════════
    #  2) APF — every skill expressed as a FORCE, which is the whole point
    # ══════════════════════════════════════════════════════════════════
    def _plan_apf(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Artificial Potential Field, upgraded without ceasing to be a field.

        Native translations of the shared skills:
          * cmd fade      -> scales K_att. A blocked command literally weakens
            the attractor, which is the textbook way to say it.
          * early turn    -> a FORWARD PROBE virtual obstacle placed where the
            commanded arc closes. Reacting to where the path ENDS rather than
            to the nearest cell is what makes it turn early.
          * commitment    -> a lateral bias FORCE, not a filter, so two close
            obstacles cannot swap the resultant every cycle.
          * escape exit   -> another attractor.
          * VFH+T trap    -> the detected exit becomes a strong attractor. In a
            concave pocket the repulsive sum forms a saddle that cancels the
            goal pull; an exit attractor is the only thing that breaks it.

        Bugs fixed: repulsion was summed per inflated CELL, so its magnitude
        tracked how many cells the inflation happened to paint — one post at
        1 m saturated the steer at full lock. It is now normalised by cell
        count. And the symmetry breaker slid one way only; it now alternates,
        so a wrong first guess costs one candidate instead of the manoeuvre."""
        now_t = time.monotonic()
        K_ATT, K_REP = 1.0, 12.0
        REP_RANGE = self._horizon_m
        des_steer, des_clear, fade = self._situation(des_v, des_w)

        # attractive — the command, weakened when the command is blocked
        fx = K_ATT * fade * math.cos(des_steer)
        fy = K_ATT * fade * math.sin(des_steer)

        # repulsive — obstacles, normalised by cell count
        with self._grid_lock:
            occ = np.argwhere(self._grid >= COST_INSCRIBED)
        n_cells = max(len(occ), 1)
        for i, j in occ:
            ox = i * self._res - self._half
            oy = j * self._res - self._half
            d = math.hypot(ox, oy)
            if 0.05 < d < REP_RANGE:
                mag = (K_REP / n_cells) * (1.0 / d - 1.0 / REP_RANGE) / (d * d)
                fx -= mag * ox / d
                fy -= mag * oy / d

        # forward probe — APF's own way of implementing the prox term
        if des_clear < self._horizon_m:
            px = des_clear * math.cos(des_steer)
            py = des_clear * math.sin(des_steer)
            d = max(math.hypot(px, py), 0.15)
            probe = self._w_prox * 0.25 * (1.0 - des_clear / self._horizon_m) / d
            fx -= probe * px / d
            fy -= probe * py / d

        # commitment force, gated by (1 - fade): a commitment force left on in
        # open ground is a permanent sideways push and made the wag worse
        if self._commit_sign != 0.0 and self._commit_age < 15:
            fy += self._commit_pen * 0.5 * self._commit_sign * (1.0 - fade)

        # proven escape exit, and the VFH+T pocket exit, as attractors
        if now_t < self._post_until and self._post_dir != 0.0:
            w = self._post_w * (3.0 if self._esc_count >= 2 else 1.0)
            fx += w * math.cos(self._post_steer)
            fy += w * math.sin(self._post_steer)
        if (self._trap_exit is not None
                and (now_t - self._trap_exit_t) < 10.0
                and abs(self._trap_exit) < math.pi * 0.6):
            fx += self._trap_w * math.cos(self._trap_exit)
            fy += self._trap_w * math.sin(self._trap_exit)

        heading = math.atan2(fy, fx)
        base = max(-self._max_steer, min(self._max_steer, heading))

        # resolve the resultant against the costmap, alternating sides
        open_sign = 1.0 if self._side_openness() >= 0 else -1.0
        v_probe = max(abs(des_v), self._min_drive)
        for mag_d in (0.0, 0.12, 0.24, 0.36, 0.50, 0.65):
            signs = (open_sign,) if mag_d == 0.0 else (open_sign, -open_sign)
            for sgn in signs:
                steer = max(-self._max_steer,
                            min(self._max_steer, base + sgn * mag_d))
                pts, max_c, clear_dist = self._rollout(v_probe, steer)
                if max_c < self._lethal:
                    return self._finish(des_v, steer, max_c, clear_dist, pts)
        self._publish_best([])
        self._planner_pinned = False
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════════
    #  3) TEB-style — homotopy classes with hysteresis, which is real TEB
    # ══════════════════════════════════════════════════════════════════
    def _plan_teb(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Elastic band deformation, upgraded.

        Native translations:
          * cmd fade      -> how many homotopy seeds to explore. Open path,
            the three-seed core; boxed in, fan out to five bows.
          * early turn    -> bands are ranked by the clear_dist of the ARC
            that tracks them, not by the band's own peak cost.
          * commitment    -> homotopy-class hysteresis. Real TEB keeps its
            chosen class unless another beats it by a margin.
          * escape / trap -> extra seeds are added along the proven exit and
            the VFH+T pocket exit, which is exactly how TEB explores classes.

        Three real bugs fixed:
          1. The band was checked at 10 waypoints 13 cm apart and the executed
             constant-steer arc was never checked at all — the car could drive
             an arc the planner had not looked at. It is now validated.
          2. Inside the lethal core every cell reads 100, so the gradient is
             exactly zero: a waypoint that landed in an obstacle felt no push
             and sat there. A radial escape term now leaves the plateau.
          3. The lookahead snapped to the first waypoint past LOOK. As the band
             deforms, the winning index flips between 4 and 5 and those two can
             differ by 20 deg of bearing — the servo twitched every cycle even
             with 1.3 m of clear road. It is now interpolated."""
        now_t = time.monotonic()
        N, ITER, STEP, ELASTIC = 10, 10, 0.06, 0.20
        GRAD_SCALE = 25.0
        des_steer, des_clear, fade = self._situation(des_v, des_w)
        v_seed = max(abs(des_v), self._min_drive)
        seg = self._horizon_m / N

        def seed_band(bow):
            band, x, y, yaw = [], 0.0, 0.0, 0.0
            w = (v_seed / self._wheelbase) * math.tan(des_steer)
            for k in range(N):
                yaw += w * (seg / v_seed)
                x += seg * math.cos(yaw)
                y += seg * math.sin(yaw)
                band.append([x, y + bow * math.sin(math.pi * (k + 1) / (N + 1))])
            return band

        def deform(band):
            eps = self._res
            for _ in range(ITER):
                for k in range(1, N):
                    bx, by = band[k]
                    gx = self._cost_at(bx + eps, by) - self._cost_at(bx - eps, by)
                    gy = self._cost_at(bx, by + eps) - self._cost_at(bx, by - eps)
                    if gx == 0 and gy == 0 and self._cost_at(bx, by) >= self._lethal:
                        # Flat lethal plateau — no gradient exists. Push out
                        # radially instead of sitting still.
                        band[k][0] += STEP * 0.5
                        band[k][1] -= STEP * (1.0 if by >= 0 else -1.0)
                    else:
                        band[k][0] -= STEP * gx / GRAD_SCALE
                        band[k][1] -= STEP * gy / GRAD_SCALE
                    if 0 < k < N - 1:
                        mx = 0.5 * (band[k - 1][0] + band[k + 1][0])
                        my = 0.5 * (band[k - 1][1] + band[k + 1][1])
                        band[k][0] += ELASTIC * (mx - band[k][0])
                        band[k][1] += ELASTIC * (my - band[k][1])
            return band

        # The three-seed core is always present. Dropping to a single straight
        # seed when the command looked clear made TEB return zero at exactly
        # the moment an obstacle entered the horizon.
        bows = [0.0, +0.45, -0.45]
        if fade < 0.5:
            bows += [+0.70, -0.70]
        if now_t < self._post_until and self._post_dir != 0.0:
            bows.append(0.55 * (1.0 if self._post_steer > 0 else -1.0))
        if (self._trap_exit is not None
                and (now_t - self._trap_exit_t) < 10.0):
            bows.append(0.60 * (1.0 if self._trap_exit > 0 else -1.0))

        LOOK = 0.55
        v_probe = max(abs(des_v), self._min_drive)
        best = None
        for bow in bows:
            band = deform(seed_band(bow))
            if max((self._cost_at(px, py) for px, py in band),
                   default=0) >= self._lethal:
                continue

            # interpolated lookahead point along the band
            tgt, acc, prev_p = band[-1], 0.0, (0.0, 0.0)
            for p in band:
                d = math.hypot(p[0] - prev_p[0], p[1] - prev_p[1])
                if acc + d >= LOOK and d > 1e-6:
                    f = (LOOK - acc) / d
                    tgt = (prev_p[0] + f * (p[0] - prev_p[0]),
                           prev_p[1] + f * (p[1] - prev_p[1]))
                    break
                acc += d
                prev_p = (p[0], p[1])

            alpha = math.atan2(tgt[1], tgt[0])
            Ld = max(math.hypot(tgt[0], tgt[1]), 0.2)
            steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), Ld)
            steer = max(-self._max_steer, min(self._max_steer, steer))

            # validate the arc we will ACTUALLY drive, not just the band
            pts, max_c, clear_dist = self._rollout(v_probe, steer)
            if max_c >= self._lethal:
                continue
            clear01 = 1.0 - max_c / 100.0
            score = (self._w_clear * clear01
                     + self._w_prox * min(clear_dist / self._horizon_m, 1.0)
                     + self._w_cmd * fade * (1.0 - abs(steer - des_steer) /
                                             (2 * self._max_steer))
                     + self._bias(steer, clear01, v_probe, des_w, now_t))
            # homotopy-class hysteresis
            if self._commit_sign != 0.0 and bow * self._commit_sign > 0:
                score += self._commit_bon
            if best is None or score > best[0]:
                best = (score, steer, max_c, clear_dist, band)

        if best is None:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0
        _, steer, max_c, clear_dist, band = best
        return self._finish(des_v, steer, max_c, clear_dist,
                            [(p[0], p[1]) for p in band])

    # ══════════════════════════════════════════════════════════════════
    #  4) VFH — upgraded to VFH+, whose cost function already has these skills
    # ══════════════════════════════════════════════════════════════════
    def _plan_vfh(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Polar obstacle-density histogram, upgraded from plain VFH to VFH+.

        VFH+ scores each candidate direction with
            G = a*|c - target| + b*|c - wheel| + c*|c - previous|
        and that third term IS the commitment filter — the algorithm already
        had the mechanism, it simply was not implemented here. Native
        translations: `a` carries the cmd fade, `b` becomes sector DEPTH (the
        early-turn term), `c` is the commitment, and the escape / VFH+T exits
        enter as negative cost on sectors pointing that way.

        Three bugs fixed:
          1. THRESH = 25 was compared against a SUM over cells. One cost-100
             cell inside 1.5 m already scores 50, and inflation paints ~50
             cells per obstacle point, so every sector in a real room read as
             blocked and the car stopped dead in a 0.95 m corridor. It is now
             a mean density.
          2. 36 sectors put the centres at +-5, +-15 deg — there was no sector
             at zero, so a straight command came out as -5 deg and the car
             drifted right forever. 72 sectors halve that.
          3. Only the single best sector was rolled out; if that one arc was
             lethal the whole cycle returned zero and handed control to the
             reverse manager. All admissible sectors are now tried in order."""
        now_t = time.monotonic()
        N_SECT = 72
        THRESH = 55.0
        des_steer, des_clear, fade = self._situation(des_v, des_w)

        hist = np.zeros(N_SECT)
        cnt = np.zeros(N_SECT)
        depth = np.full(N_SECT, self._horizon_m)
        with self._grid_lock:
            occ = np.argwhere(self._grid >= COST_INSCRIBED)
            grid = self._grid.copy()
        for i, j in occ:
            ox = i * self._res - self._half
            oy = j * self._res - self._half
            d = math.hypot(ox, oy)
            if d < 0.05 or d > self._horizon_m:
                continue
            k = int(round(((math.atan2(oy, ox) + math.pi) /
                           (2 * math.pi)) * N_SECT)) % N_SECT
            hist[k] += float(grid[i, j]) * (1.0 - d / self._horizon_m)
            cnt[k] += 1.0
            if d < depth[k]:
                depth[k] = d
        dens = np.where(cnt > 0, hist / np.maximum(cnt, 1.0), 0.0)

        cands = []
        for k in range(N_SECT):
            ang = -math.pi + (k + 0.5) * 2 * math.pi / N_SECT
            if abs(ang) > self._max_steer * 1.6 or dens[k] > THRESH:
                continue
            g = (self._w_cmd * fade * abs(ang - des_steer)
                 + self._w_prox * (1.0 - depth[k] / self._horizon_m))
            if self._commit_sign != 0.0 and self._commit_age < 15:
                if ang * self._commit_sign < -0.01:
                    g += self._commit_pen
                else:
                    g -= self._commit_bon
            if now_t < self._post_until and self._post_dir != 0.0:
                w = self._post_w * (3.0 if self._esc_count >= 2 else 1.0)
                g -= w * (1.0 - abs(ang - self._post_steer) / (2 * self._max_steer))
            if (self._trap_exit is not None
                    and (now_t - self._trap_exit_t) < 10.0
                    and abs(self._trap_exit) < math.pi * 0.6):
                clamped = max(-self._max_steer,
                              min(self._max_steer, self._trap_exit))
                g -= self._trap_w * (1.0 - abs(ang - clamped) /
                                     (2 * self._max_steer))
            cands.append((g, ang))
        cands.sort()

        v_probe = max(abs(des_v), self._min_drive)
        for _, ang in cands:
            steer = max(-self._max_steer, min(self._max_steer, ang))
            pts, max_c, clear_dist = self._rollout(v_probe, steer)
            if max_c < self._lethal:
                return self._finish(des_v, steer, max_c, clear_dist, pts)
        self._publish_best([])
        self._planner_pinned = False
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════════
    #  5) PURE PURSUIT — adaptive lookahead, the method's own tuning knob
    # ══════════════════════════════════════════════════════════════════
    def _plan_pure_pursuit(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Geometric lookahead tracking, upgraded.

        Native translations:
          * cmd fade      -> the lookahead SHORTENS when the path is blocked.
            A short Ld turns harder, a long Ld runs smooth and straight; that
            is pure pursuit's own way of trading commitment against agility
            and it is what every real implementation tunes.
          * early turn    -> candidate goal points ranked by the clear_dist of
            the arc that reaches them.
          * commitment    -> the sweep is ordered from the committed side.
          * escape / trap -> the proven exit and the pocket exit are tried as
            goal points first.

        Main bug fixed: `_desired_steer()` returns a STEERING ANGLE and it was
        being fed in as the BEARING to the goal point, then run through the
        pure-pursuit law. The bicycle geometry was applied twice, so a 30 deg
        command came out as 17.8 deg and the car under-turned on every corner.
        The goal point now sits ON the commanded arc: a chord of length Ld on
        an arc of curvature k subtends a bearing of about k*Ld/2."""
        now_t = time.monotonic()
        des_steer, des_clear, fade = self._situation(des_v, des_w)

        LOOK = 0.45 + 0.40 * fade          # 0.45 m boxed in, 0.85 m in the open
        kappa = math.tan(des_steer) / self._wheelbase
        alpha0 = 0.5 * kappa * LOOK

        offs = [0.0, 0.15, -0.15, 0.30, -0.30, 0.45, -0.45,
                0.60, -0.60, 0.80, -0.80]
        if self._commit_sign != 0.0 and self._commit_age < 15:
            offs.sort(key=lambda d: (d * self._commit_sign < -1e-6, abs(d)))
        if now_t < self._post_until and self._post_dir != 0.0:
            offs.insert(0, self._post_steer - alpha0)
        if (self._trap_exit is not None
                and (now_t - self._trap_exit_t) < 10.0
                and abs(self._trap_exit) < math.pi * 0.6):
            offs.insert(0, self._trap_exit - alpha0)

        v_probe = max(abs(des_v), self._min_drive)
        best = None
        for dalpha in offs:
            alpha = alpha0 + dalpha
            steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), LOOK)
            steer = max(-self._max_steer, min(self._max_steer, steer))
            pts, max_c, clear_dist = self._rollout(v_probe, steer)
            if max_c >= self._lethal:
                continue
            clear01 = 1.0 - max_c / 100.0
            score = (self._w_prox * min(clear_dist / self._horizon_m, 1.0)
                     + self._w_clear * clear01
                     - self._w_cmd * fade * abs(dalpha) / 0.8
                     + self._bias(steer, clear01, v_probe, des_w, now_t))
            if best is None or score > best[0]:
                best = (score, steer, max_c, clear_dist, pts)
            if dalpha == 0.0 and clear_dist >= self._horizon_m * 0.95:
                break                      # command is clear, take it and go
        if best is None:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0
        _, steer, max_c, clear_dist, pts = best
        return self._finish(des_v, steer, max_c, clear_dist, pts)

    # ══════════════════════════════════════════════════════════════════
    #  6) FGM — gaps ranked by depth as well as width
    # ══════════════════════════════════════════════════════════════════
    def _plan_fgm(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Follow-the-Gap on the raw scan, upgraded.

        Native translations:
          * cmd fade      -> how strongly the aim is pulled to the goal inside
            the gap, instead of a hardcoded 70/30 blend.
          * early turn    -> gaps ranked by width x DEPTH, so a wide but
            shallow gap (an alcove) loses to a narrower one that leads
            somewhere. This is FGM's version of looking past the obstacle.
          * commitment    -> the gap choice carries hysteresis, so two similar
            gaps stop swapping every cycle.
          * escape / trap -> both exits bias which gap wins.

        Three bugs fixed:
          1. `target = 0.7*centre + 0.3*goal` ran unconditionally, even in an
             empty field where the gap is the whole cone and its centre is
             zero. A 30 deg command produced 9 deg; behind an owner walking a
             circle the car fell 4.2 m behind and lost them.
          2. `closest` was the minimum over the whole +-100 deg cone, so a wall
             beside the car raised the free/blocked threshold for the rays
             pointing dead ahead and an open path read as blocked.
          3. A gap needed 3 rays — about 2.5 cm at 1.3 m, for a 31 cm car."""
        now_t = time.monotonic()
        if self._scan_ang is None or getattr(self, "_scan_rng", None) is None:
            return 0.0, 0.0
        if len(self._scan_ang) == 0:
            return 0.0, 0.0
        SAFE_D = self._robot_r + 0.35
        des_steer, des_clear, fade = self._situation(des_v, des_w)

        ang, rng = self._scan_ang, self._scan_rng
        cone = np.abs(ang) <= math.radians(100)
        a, r = ang[cone], rng[cone]
        if len(a) == 0:
            return 0.0, 0.0
        order = np.argsort(a)
        a, r = a[order], r[order]

        fwd = np.abs(a) <= math.radians(45)
        closest = float(np.min(r[fwd])) if np.any(fwd) else float(np.min(r))
        thresh = max(SAFE_D, min(self._horizon_m, closest + 0.45))
        free = r > thresh

        # collect EVERY gap, not just the widest
        gaps, i0 = [], None
        for idx, f in enumerate(free):
            if f and i0 is None:
                i0 = idx
            elif (not f) and i0 is not None:
                gaps.append((i0, idx))
                i0 = None
        if i0 is not None:
            gaps.append((i0, len(free)))

        scored = []
        for lo_i, hi_i in gaps:
            if hi_i - lo_i < 3:
                continue
            g_ang, g_rng = a[lo_i:hi_i], r[lo_i:hi_i]
            d_gap = float(np.mean(g_rng))
            width = float(g_ang[-1] - g_ang[0])
            need = 2.0 * math.atan2(self._robot_r, max(d_gap, 0.3))
            if width < need:
                continue                   # the car does not fit
            centre = 0.5 * (g_ang[0] + g_ang[-1])
            margin = 0.5 * need
            lo, hi = g_ang[0] + margin, g_ang[-1] - margin
            if lo > hi:
                lo = hi = centre
            aim = fade * float(np.clip(des_steer, lo, hi)) + (1.0 - fade) * centre
            aim = float(np.clip(aim, lo, hi))
            quality = (width / math.radians(60)) * (d_gap / self._horizon_m)
            pref = (quality * self._w_prox
                    - self._w_cmd * fade * abs(aim - des_steer)
                    + self._bias(max(-self._max_steer,
                                     min(self._max_steer, aim)),
                                 1.0, max(abs(des_v), self._min_drive),
                                 des_w, now_t))
            scored.append((pref, aim, lo, hi, centre))
        if not scored:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0
        scored.sort(reverse=True)

        v_probe = max(abs(des_v), self._min_drive)
        for _, aim, lo, hi, centre in scored:
            # sweep INSIDE the gap only — never aim at a door frame
            tries = [aim] + \
                    [float(np.clip(aim + d * (1.0 if centre >= aim else -1.0),
                                   lo, hi)) for d in (0.08, 0.16, 0.24, 0.32)] + \
                    [float(np.clip(centre, lo, hi))]
            for t in tries:
                steer = max(-self._max_steer, min(self._max_steer, float(t)))
                pts, max_c, clear_dist = self._rollout(v_probe, steer)
                if max_c < self._lethal:
                    return self._finish(des_v, steer, max_c, clear_dist, pts)
        self._publish_best([])
        self._planner_pinned = False
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════════
    #  7) LATTICE — the second segment already IS the lookahead
    # ══════════════════════════════════════════════════════════════════
    def _plan_lattice(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Two-segment arc lattice, upgraded.

        Native translations:
          * cmd fade      -> weight on segment-one similarity to the command.
          * early turn    -> inherent, but now scored properly: the reach of
            segment two decides whether the avoidance arc leads anywhere.
          * commitment    -> applies to segment one only, so the car can still
            plan a swing-back in segment two.
          * escape / trap -> both bias segment one through the shared _bias.

        Bug fixed: segment two was penalised for deviating from the commanded
        steer, which fought the entire purpose of a second segment — it wanted
        to go around the obstacle and immediately resume the old bearing."""
        now_t = time.monotonic()
        SEG = 0.65
        des_steer, des_clear, fade = self._situation(des_v, des_w)
        v = max(abs(des_v), self._min_drive)
        s_set = np.linspace(-self._max_steer, self._max_steer, 9)

        best = None
        for s1 in s_set:
            pts1, c1, cd1 = self._rollout(v, float(s1), dist=SEG)
            if c1 >= self._lethal or not pts1:
                continue
            x1, y1 = pts1[-1]
            yaw1 = (SEG / self._wheelbase) * math.tan(float(s1))
            for s2 in s_set[::2]:
                x, y, yaw = x1, y1, yaw1
                kappa2 = math.tan(float(s2)) / self._wheelbase
                step = 0.5 * self._res
                c2, pts2, reach = c1, [], 0.0
                for _ in range(int(SEG / step)):
                    yaw += kappa2 * step
                    x += step * math.cos(yaw)
                    y += step * math.sin(yaw)
                    pts2.append((x, y))
                    c2 = max(c2, self._cost_at(x, y))
                    if c2 >= self._lethal:
                        break
                    reach += step
                if c2 >= self._lethal:
                    continue
                clear01 = 1.0 - c2 / 100.0
                # segment 2 is scored on where it GETS you, not on obedience
                score = (self._w_cmd * fade * (1.0 - abs(float(s1) - des_steer) /
                                               (2 * self._max_steer))
                         + self._w_clear * clear01
                         + self._w_prox * min((cd1 + reach) / (2 * SEG), 1.0)
                         + self._w_speed * abs(v) / self._max_mps
                         + self._bias(float(s1), clear01, v, des_w, now_t))
                if best is None or score > best[0]:
                    best = (score, float(s1), c2, cd1 + reach, pts1 + pts2)
        if best is None:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0
        _, s1, cost, clear_dist, pts = best
        return self._finish(des_v, s1, cost,
                            min(clear_dist, self._horizon_m), pts)

    # ══════════════════════════════════════════════════════════════════
    #  PLANNER SELECTION
    # ══════════════════════════════════════════════════════════════════
    def _resolve_planner(self, name: str) -> str:
        """Normalise a planner name, accepting the obvious aliases.

        An unknown name is a configuration error, and the wrong response to a
        configuration error on a robot is to refuse to start — you find out in
        the field, with the car already on the ground. We warn loudly and run
        the default instead."""
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        key = self._planner_aliases.get(key, key)
        if key not in self._planners:
            self.get_logger().error(
                f"unknown planner '{name}' — valid choices are "
                f"{', '.join(sorted(self._planners))}. Using pure_pursuit.")
            return "pure_pursuit"
        return key

    def _on_param_set(self, params):
        """Allow the planner to be swapped at runtime.

            ros2 param set /smart_planner_node planner fgm

        Switching mid-drive is safe: every planner reads the same costmap and
        writes through the same _finish(), so the only discontinuity is one
        cycle of steering, which the slew limiter smooths anyway. What it is
        NOT safe to change at runtime is output_mode, because that decides who
        owns the serial port — so that one is rejected.
        """
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            if p.name == "planner":
                new = self._resolve_planner(str(p.value))
                if new != self._planner_name:
                    self.get_logger().info(
                        f"planner: {self._planner_name} -> {new}")
                    self._planner_name = new
                    self._planner_fn = self._planners[new]
                    self._planner_faults = 0
                    # Drop cached RRT work: it belongs to the old planner's
                    # idea of the world and would be applied out of context.
                    self._rrt_tree = None
                    self._rrt_path = None
            elif p.name == "planner_fallback":
                self._planner_fallback = self._resolve_planner(str(p.value))
            elif p.name == "output_mode":
                if str(p.value).strip().lower() != self._out_mode:
                    return SetParametersResult(
                        successful=False,
                        reason="output_mode cannot change at runtime — it "
                               "decides which process owns the serial port. "
                               "Restart the node.")
            elif p.name == "rrt_iterations":
                self._rrt_iters = int(p.value)
            elif p.name == "rrt_step_m":
                self._rrt_step = float(p.value)
            elif p.name == "rrt_goal_bias":
                self._rrt_bias = float(p.value)

        return SetParametersResult(successful=True)

    # ══════════════════════════════════════════════════════════════════
    #  8) RRT — Rapidly-exploring Random Tree                  [sampling]
    # ══════════════════════════════════════════════════════════════════
    def _plan_rrt(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Sampling-based local planner.

        ─────────────────────────────────────────────────────────────────
        Why add a sampling planner to a set that already has seven?
        ─────────────────────────────────────────────────────────────────
        The other seven are all, in different ways, *local gradient*
        methods. DWA, Lattice and TEB sample a fan of arcs from the current
        pose; APF descends a potential field; VFH and FGM pick a direction
        from a histogram. Every one of them makes a decision that is a
        function of the immediate neighbourhood, and every one of them can
        therefore be defeated by the same thing: a concave obstacle whose
        exit is not visible from any single-arc rollout.

        The existing VFH+T trap detector exists precisely because of this.
        It is a good detector, but its remedy is an escape manoeuvre —
        reverse out and try again — which is expensive and looks alarming.

        RRT fails differently. It builds a tree of connected motions, so it
        can find a path that goes *away* from the goal before coming back,
        which none of the other seven can express at all. In a doorway, a
        U-shaped corridor, or a cluttered room with one narrow exit, that is
        the difference between threading through and backing out.

        It is not free. RRT is stochastic: two identical situations can give
        two different paths, and the paths are jagged. Both are handled:

          * The tree is **cached and reused** between cycles. A local costmap
            at 10 Hz barely changes between frames, so rebuilding from
            scratch every cycle wastes the work and, worse, produces a
            different answer every 100 ms — which the steering servo turns
            into visible jitter. We keep the portion of the tree that is
            still collision-free and grow into it.

          * The path is **shortcut-smoothed** before use, then handed to the
            same pure-pursuit geometry the rest of the stack uses, so the
            output is an arc like every other planner's.

          * Everything goes through **_finish()**, so the speed law, approach
            easing, slew limit and stall reporting are identical to DWA's.
            Switching planners changes how the direction is chosen, never
            how the car behaves once it is chosen.
        ─────────────────────────────────────────────────────────────────
        """
        import random

        now = time.monotonic()
        des_steer, des_clear, fade = self._situation(des_v, des_w)

        # ---- Goal: where the command wants us, projected to the horizon --
        # RRT needs a point to grow towards. The Jetson gives us a velocity,
        # not a position, so we integrate the commanded arc out to the
        # horizon and use its endpoint.
        #
        # Note that we do NOT apply _bias() here. _bias returns a score
        # adjustment for ranking arcs, not a steering angle, and the whole
        # point of RRT is that the tree — not a bias term — is what finds a
        # way round an obstacle. The bias still gets its say: it is inside
        # _score_arc, which the fallback fan below uses.
        # If the commanded arc is already clear over the whole horizon there is
        # nothing to sample for: take it, exactly as pure pursuit does. Without
        # this, a clear road still got whatever winding path the tree happened
        # to grow, and the smoother cannot straighten it fully because every
        # shortcut has to respect the 0.485 m turning circle. Sampling is for
        # when the single arc does NOT work - that is the whole reason RRT is
        # in the set.
        v_probe = max(abs(des_v), self._min_drive)
        pts0, c0, clr0 = self._rollout(v_probe, des_steer)
        if c0 < self._lethal and clr0 >= self._horizon_m * 0.95:
            self._rrt_path = None
            return self._finish(des_v, des_steer, c0, clr0, pts0)

        gx, gy = self._arc_endpoint(v_probe, des_steer, self._horizon_m)

        # A trap exit, when one is known, overrides the commanded direction
        # entirely — it is the only bearing we have evidence leads out.
        if (self._trap_exit is not None
                and (now - self._trap_exit_t) < 2.0):
            r = self._horizon_m
            gx = r * math.cos(self._trap_exit)
            gy = r * math.sin(self._trap_exit)

        # ---- Seed the tree ---------------------------------------------
        # nodes: list of (x, y, parent_index, heading)
        #
        # The heading is not decoration. A plain geometric RRT connects nodes
        # with straight lines and imposes no constraint on the angle between
        # consecutive segments, which means it will cheerfully return a path
        # that doubles back on itself — a path an Ackermann car cannot drive
        # without stopping and reversing. Carrying the heading and limiting
        # how much it may change per step turns this into a kinodynamic RRT
        # whose every edge is drivable at the car's real minimum turning
        # radius.
        #
        #   R_min      = wheelbase / tan(max_steer)
        #   max turn   = step / R_min      radians per step
        #
        # With the measured geometry that is 0.280/tan(30 deg) = 0.485 m, so a
        # 22 cm step may turn at most 28 degrees. Paths out of a pocket that
        # require reversing are therefore correctly *not* found here — that
        # case belongs to _detect_trap() and the three-point escape, which
        # can reverse. A planner that returns an undrivable path is worse
        # than one that admits defeat, because the car tries to follow it.
        nodes = self._reuse_tree()
        if nodes is None:
            nodes = [(0.0, 0.0, -1, 0.0)]

        reach = self._map_size * 0.5 - self._res
        step = self._rrt_step
        r_min = self._wheelbase / max(math.tan(self._max_steer), 1e-6)
        max_turn = min(math.pi * 0.5, step / r_min)
        best_i = None
        best_d = float("inf")

        # ---- Node coordinates mirrored into numpy ----------------------
        # Nearest-neighbour search is the inner loop of any RRT and it is
        # O(nodes) per iteration. At 600 iterations and up to 900 nodes that
        # is half a million distance computations per plan; in a Python loop
        # on a Pi 4 that alone overruns the 100 ms budget at 10 Hz, which is
        # how a planner ends up "working fine" on a laptop and stuttering on
        # the robot.
        #
        # Keeping the coordinates in preallocated arrays and doing the search
        # with a few numpy operations makes it several times faster and is
        # the difference between RRT being usable on this hardware and not.
        # The `nodes` list stays authoritative; these arrays are a parallel
        # index into it and are appended to in lockstep.
        cap = max(self._rrt_max_nodes + self._rrt_greedy + 8, len(nodes) + 8)
        _xs = np.zeros(cap, dtype=np.float64)
        _ys = np.zeros(cap, dtype=np.float64)
        _hs = np.zeros(cap, dtype=np.float64)
        for _i, (_x, _y, _p, _h) in enumerate(nodes):
            _xs[_i], _ys[_i], _hs[_i] = _x, _y, _h
        _n = len(nodes)

        for _ in range(self._rrt_iters):
            # ---- Sample --------------------------------------------------
            if random.random() < self._rrt_bias:
                sx, sy = gx, gy
            else:
                # Sample a forward-biased cone rather than the whole disc.
                #
                # The width of this cone is a real design decision, and an
                # earlier version got it wrong. Restricting samples to the
                # forward half-plane (+/-90 deg) seemed reasonable — the car
                # drives forwards — but it quietly removes the ability to
                # curl around the tip of a wall, which is most of what a
                # sampling planner is here to do. The tree could never place
                # a node lateral-and-slightly-behind the axle, so it could
                # never find a path that leaves before it arrives.
                #
                # The default is +/-110 deg: wide enough to round a corner,
                # narrow enough not to spend iterations directly behind the
                # rear axle, where an Ackermann car cannot go without
                # reversing. Genuine must-reverse pockets are not RRT's job;
                # they are detected by _detect_trap() and handled by the
                # three-point escape manoeuvre.
                half = self._rrt_span * 0.5
                ang = random.uniform(-half, half)
                rad = math.sqrt(random.random()) * reach
                sx = rad * math.cos(ang)
                sy = rad * math.sin(ang)

            # ---- Nearest node ------------------------------------------
            # Two candidates are tracked: the nearest node that can turn
            # towards the sample within one step, and the nearest node of
            # any orientation.
            #
            # An earlier version kept only the first and skipped the
            # iteration when nothing qualified. That is wrong, and measurably
            # so — it cut the solve rate roughly in half. Early in the search
            # the tree is a single node pointing forward, so with a 28 degree
            # turn limit three quarters of samples have no qualifying parent
            # and the iteration is simply thrown away; the tree then cannot
            # fan out, which is precisely what it needs to do to get around
            # anything.
            #
            # Falling back to the plain nearest node and *clamping* the turn
            # loses nothing: the clamp is what enforces the turning circle,
            # and it is applied unconditionally below. The preference merely
            # produces straighter trees when it can be satisfied.
            dx = sx - _xs[:_n]
            dy = sy - _ys[:_n]
            d2 = dx * dx + dy * dy
            bearing = np.arctan2(dy, dx)
            turn = np.abs(np.arctan2(np.sin(bearing - _hs[:_n]),
                                     np.cos(bearing - _hs[:_n])))
            # Rank by distance with a small penalty for how hard the turn is.
            # np.where keeps disqualified nodes in place rather than
            # compacting the array, so the index returned still indexes
            # `nodes` directly.
            score = np.where(turn <= max_turn,
                             d2 + (turn / max_turn) ** 2 * 0.04, np.inf)
            score = np.where(d2 < 1e-12, np.inf, score)
            ni = int(np.argmin(score))
            if not np.isfinite(score[ni]):
                d2f = np.where(d2 < 1e-12, np.inf, d2)
                ni = int(np.argmin(d2f))
                if not np.isfinite(d2f[ni]):
                    continue

            nx, ny, nh = float(_xs[ni]), float(_ys[ni]), float(_hs[ni])
            nd = math.hypot(sx - nx, sy - ny)
            if nd < 1e-6:
                continue

            # ---- Steer one step, clamped to the turning circle ----------
            want = math.atan2(sy - ny, sx - nx)
            delta = math.atan2(math.sin(want - nh), math.cos(want - nh))
            delta = max(-max_turn, min(max_turn, delta))
            heading = nh + delta
            px, py = self._arc_step(nx, ny, nh, delta, step)

            if abs(px) > reach or abs(py) > reach:
                continue
            if not self._segment_free(nx, ny, px, py):
                continue

            nodes.append((px, py, ni, heading))
            cur = len(nodes) - 1
            _xs[_n], _ys[_n], _hs[_n] = px, py, heading
            _n += 1

            d_goal = math.hypot(px - gx, py - gy)
            if d_goal < best_d:
                best_d, best_i = d_goal, cur

            # ---- Greedy extension (the RRT-Connect idea) ----------------
            # Having added a node, keep stepping towards the SAMPLE for as
            # long as the steps stay legal, instead of going back to draw a
            # new one. One sample then buys a whole run of progress.
            #
            # This matters far more once the turning circle is enforced.
            # Without the kinematic limit a single 22 cm step could point
            # anywhere, so the tree spread quickly; with it, each step may
            # turn only 28 degrees, and plain sample-and-extend crawls.
            #
            # Measured on the two-wall S-curve scenario in
            # scripts/test_rrt_offline.py, at a fixed 600 iterations:
            #
            #   no greedy extension .................... 32% solved
            #   greedy on goal-biased samples only ..... 64% solved
            #   greedy on every sample ................. see below
            #
            # Extending towards random samples as well is what turns the
            # tree from a slowly-fattening blob into something that actually
            # sweeps the free space. The turning-circle clamp is applied on
            # every one of these steps, so nothing about drivability is
            # traded away for the speed.
            for _ in range(self._rrt_greedy):
                want = math.atan2(sy - py, sx - px)
                if math.hypot(sx - px, sy - py) < step * 0.5:
                    break
                delta = math.atan2(math.sin(want - heading),
                                   math.cos(want - heading))
                delta = max(-max_turn, min(max_turn, delta))
                qx, qy = self._arc_step(px, py, heading, delta, step)
                heading = heading + delta
                if abs(qx) > reach or abs(qy) > reach:
                    break
                if not self._segment_free(px, py, qx, qy):
                    break
                nodes.append((qx, qy, cur, heading))
                cur = len(nodes) - 1
                _xs[_n], _ys[_n], _hs[_n] = qx, qy, heading
                _n += 1
                px, py = qx, qy
                d_goal = math.hypot(px - gx, py - gy)
                if d_goal < best_d:
                    best_d, best_i = d_goal, cur
                if d_goal < self._rrt_tol:
                    break
                if len(nodes) > self._rrt_max_nodes or _n >= cap - 2:
                    break

            if best_d < self._rrt_tol:
                break

            # A tree larger than this is not buying accuracy, it is buying
            # latency — and a local planner that misses its 10 Hz slot is
            # worse than one that plans slightly worse paths on time.
            if len(nodes) > self._rrt_max_nodes or _n >= cap - 2:
                break

        self._rrt_tree = nodes

        # ---- No usable tree: hand over to the shared failure path -------
        if best_i is None or len(nodes) < 2:
            self._publish_best([])
            self._planner_pinned = False
            self._rrt_path = None
            return 0.0, 0.0

        # ---- Extract and smooth ----------------------------------------
        path = []
        i = best_i
        guard = 0
        while i >= 0 and guard < 1000:
            x, y, p, _h = nodes[i]
            path.append((x, y))
            i = p
            guard += 1
        path.reverse()
        path = self._shortcut(path)
        self._rrt_path = path
        self._publish_rrt(nodes, path)

        # ---- Pure-pursuit the smoothed path ----------------------------
        # Reusing the same geometry as _plan_pure_pursuit means the steering
        # response to a given path shape is identical no matter which planner
        # produced the path, which is what makes the planners genuinely
        # interchangeable rather than merely swappable.
        tx, ty = self._lookahead_point(path, self._rrt_look)
        if tx is None:
            self._publish_best([])
            self._planner_pinned = False
            return 0.0, 0.0

        ld = math.hypot(tx, ty)
        if ld < 1e-3:
            steer = 0.0
        else:
            # Ackermann steering angle for the arc through (tx, ty):
            #   kappa = 2*y / L_d^2,  steer = atan(kappa * wheelbase)
            kappa = 2.0 * ty / (ld * ld)
            steer = math.atan(kappa * self._wheelbase)
        steer = max(-self._max_steer, min(self._max_steer, steer))

        # Validate the arc we are actually about to drive, not the path we
        # planned. The path is collision-free by construction; the arc that
        # approximates it is not necessarily, and the arc is what the car does.
        pts, max_cost, clear_dist = self._rollout(
            max(abs(des_v), self._min_drive), steer)
        if max_cost >= self._lethal:
            # Fall back to scoring a small fan around the RRT direction —
            # cheaper than replanning and usually enough to recover the
            # intent of the path while staying legal.
            best = None
            for k in range(-3, 4):
                cand = max(-self._max_steer,
                           min(self._max_steer,
                               steer + k * self._max_steer * 0.12))
                sc = self._score_arc(max(abs(des_v), self._min_drive), cand,
                                     des_steer, des_v, fade, des_w, now)
                if sc is None:
                    continue
                score, cpts, cmax, cclear = sc
                if best is None or score > best[0]:
                    best = (score, cand, cmax, cclear, cpts)
            if best is None:
                self._publish_best([])
                self._planner_pinned = False
                return 0.0, 0.0
            _, steer, max_cost, clear_dist, pts = best

        self._commit_update(steer)
        return self._finish(des_v, steer, max_cost,
                            min(clear_dist, self._horizon_m), pts)

    # ------------------------------------------------------------------
    #  RRT support
    # ------------------------------------------------------------------
    def _reuse_tree(self):
        """Keep whatever remains valid of last cycle's tree.

        The robot has moved since the tree was built, and the costmap is in
        the body frame, so every node has shifted. Rather than transform them
        — which needs an odometry delta we may not have — we keep the nodes
        that are still in free space and still reachable from their parent.
        Roughly half survive in practice, which is enough to stabilise the
        output without pinning it to a stale world.
        """
        old = self._rrt_tree
        if not old or self._rrt_reuse <= 0.0:
            return None

        keep = [(0.0, 0.0, -1, 0.0)]
        remap = {0: 0}
        limit = int(len(old) * self._rrt_reuse)
        for idx in range(1, min(len(old), max(2, limit))):
            x, y, p, h = old[idx]
            if p not in remap:
                continue
            px, py, _pp, _ph = old[p]
            if self._cost_at(x, y) >= self._lethal:
                continue
            if not self._segment_free(px, py, x, y):
                continue
            remap[idx] = len(keep)
            keep.append((x, y, remap[p], h))
        return keep if len(keep) > 1 else None

    @staticmethod
    def _arc_step(x, y, heading, delta, arc_len):
        """Advance along a circular arc, not along the new heading.

        This looks like a detail and is not. The obvious way to extend an RRT
        node is to rotate to the new heading and then step forward:

            heading += delta
            x += cos(heading) * step
            y += sin(heading) * step

        That is a first-order integration of a curve, and at 28 degrees of
        turn per 22 cm step the error is large. The resulting polyline cuts
        *inside* the true arc, so a path built from edges that each respect
        the minimum turning radius traces a curve that does not.

        The consequence was concrete and measured: pure pursuit, given a
        lookahead point on such a path, computed a steering angle of 35
        degrees on a car whose servo stops at 30 — on 7 of 12 test seeds. The
        planner then had to discard its own path and fall back to the scored
        fan, which is exactly the work RRT was added to avoid.

        The correct displacement over an arc of length L turning by delta is
        the chord: it points along heading + delta/2, and its length is
        L * sin(delta/2) / (delta/2). That factor is the sinc function, which
        tends to 1 as delta tends to 0, so the straight-line case falls out
        without a special branch — but the guard below avoids dividing by
        zero when delta is exactly 0.
        """
        if abs(delta) < 1e-6:
            return (x + math.cos(heading) * arc_len,
                    y + math.sin(heading) * arc_len)
        half = delta * 0.5
        chord = arc_len * math.sin(half) / half
        mid = heading + half
        return x + math.cos(mid) * chord, y + math.sin(mid) * chord

    def _segment_free(self, x0, y0, x1, y1) -> bool:
        """Collision-check a straight segment against the inflated costmap.

        Sampled at half a cell. Checking only the endpoints is the classic
        RRT bug: with a 22 cm step on a 5 cm grid you can pass clean through
        a 15 cm obstacle and never notice.
        """
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-9:
            return self._cost_at(x0, y0) < self._lethal
        n = max(2, int(d / (self._res * 0.5)) + 1)
        for k in range(n + 1):
            t = k / float(n)
            if self._cost_at(x0 + (x1 - x0) * t,
                             y0 + (y1 - y0) * t) >= self._lethal:
                return False
        return True

    def _shortcut(self, path):
        """Randomised shortcut smoothing, constrained by the turning circle.

        Raw RRT paths zig-zag because every node is a random sample. Repeatedly
        picking two vertices and replacing everything between them with a
        straight line, whenever that line is collision-free, converges quickly
        to something close to taut. It is cheaper and more robust than spline
        fitting, and it cannot introduce a collision because every replacement
        is checked.

        It can, however, introduce something almost as bad, and an earlier
        version did: **a corner the car cannot turn**. Every edge in the tree
        respects the minimum turning radius, but cutting out the vertices
        between two of them leaves a junction whose turn happens all at once.
        The offline test measured a path whose tightest corner was 0.22 m —
        half the car's 0.485 m minimum radius — built entirely out of legal
        edges.

        A path like that is not harmless. Pure pursuit turns it into a
        steering command past full lock, the rollout check then rejects the
        resulting arc, and the planner falls back to its scored fan, having
        thrown away all the work the tree did. So the curvature constraint is
        enforced here too: a shortcut is accepted only if it is both
        collision-free AND leaves every affected corner drivable.
        """
        import random

        if len(path) < 3:
            return path

        r_min = self._wheelbase / max(math.tan(self._max_steer), 1e-6)

        def corner_ok(pts, k):
            """Is the turn at interior vertex k within the turning circle?

            Curvature at a polyline vertex is the heading change divided by
            the distance over which it happens, so a long outgoing segment can
            absorb a large turn and a short one cannot.
            """
            if k <= 0 or k >= len(pts) - 1:
                return True
            ax = pts[k][0] - pts[k - 1][0]
            ay = pts[k][1] - pts[k - 1][1]
            bx = pts[k + 1][0] - pts[k][0]
            by = pts[k + 1][1] - pts[k][1]
            la = math.hypot(ax, ay)
            lb = math.hypot(bx, by)
            if la < 1e-9 or lb < 1e-9:
                return True
            turn = abs(math.atan2(
                math.sin(math.atan2(by, bx) - math.atan2(ay, ax)),
                math.cos(math.atan2(by, bx) - math.atan2(ay, ax))))
            if turn < 1e-9:
                return True
            # The turn is shared between the two segments meeting here.
            return (min(la, lb) / turn) >= r_min

        pts = list(path)
        for _ in range(self._rrt_smooth):
            if len(pts) < 3:
                break
            i = random.randint(0, len(pts) - 3)
            j = random.randint(i + 2, len(pts) - 1)
            if not self._segment_free(pts[i][0], pts[i][1],
                                      pts[j][0], pts[j][1]):
                continue
            cand = pts[:i + 1] + pts[j:]
            # Only the two junctions at the ends of the removed span can have
            # changed, so there is no need to re-check the whole path.
            if corner_ok(cand, i) and corner_ok(cand, i + 1):
                pts = cand
        return pts

    def _arc_endpoint(self, v, steer, dist):
        """Where a constant-(v, steer) Ackermann arc ends after `dist`."""
        if abs(steer) < 1e-4:
            return dist, 0.0
        R = self._wheelbase / math.tan(steer)
        th = dist / R
        return R * math.sin(th), R * (1.0 - math.cos(th))

    def _lookahead_point(self, path, ld):
        """First point on the path at least `ld` metres from the origin.

        Falls back to the far end of the path when the whole path is shorter
        than the lookahead — which happens near a goal, and which must not be
        allowed to return nothing.
        """
        if not path:
            return None, None
        for (x, y) in path:
            if math.hypot(x, y) >= ld:
                return x, y
        x, y = path[-1]
        return (x, y) if math.hypot(x, y) > 1e-3 else (None, None)

    def _publish_rrt(self, nodes, path):
        """Draw the tree and the chosen path in RViz.

        A sampling planner is nearly impossible to debug from logs — the
        interesting question is always "what did it explore?", and that is a
        picture, not a number.
        """
        arr = MarkerArray()

        edges = Marker()
        edges.header.frame_id = self._frame
        edges.header.stamp = self.get_clock().now().to_msg()
        edges.ns = "rrt_tree"
        edges.id = 0
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD if len(nodes) > 1 else Marker.DELETE
        edges.scale.x = 0.004
        edges.color.r, edges.color.g = 0.35, 0.55
        edges.color.b, edges.color.a = 1.0, 0.45
        edges.pose.orientation.w = 1.0
        for (x, y, p, _h) in nodes:
            if p < 0:
                continue
            px, py, _pp, _ph = nodes[p]
            a = Point()
            a.x, a.y, a.z = float(px), float(py), 0.02
            b = Point()
            b.x, b.y, b.z = float(x), float(y), 0.02
            edges.points.append(a)
            edges.points.append(b)
        arr.markers.append(edges)

        line = Marker()
        line.header = edges.header
        line.ns = "rrt_path"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD if path else Marker.DELETE
        line.scale.x = 0.03
        line.color.r, line.color.g = 1.0, 0.75
        line.color.b, line.color.a = 0.1, 1.0
        line.pose.orientation.w = 1.0
        for (x, y) in path:
            pt = Point()
            pt.x, pt.y, pt.z = float(x), float(y), 0.05
            line.points.append(pt)
        arr.markers.append(line)

        self._rrt_pub.publish(arr)

    # ══════════════════════════════════════════════════════════════════
    #  20 Hz DRIVE SENDER
    # ══════════════════════════════════════════════════════════════════
    def _send_thread_fn(self):
        """Dedicated serial sender — runs outside the ROS executor.
        Immune to GIL stalls from scan/plan callbacks."""
        import time as _time
        interval = 1.0 / self._rate  # 50 Hz = 20 ms
        while rclpy.ok():
            self._send_loop()
            _time.sleep(interval)

    def _send_loop(self):
        with self._lock:
            v, s = self._out_v, self._out_s
        if self._smooth_on:
            v, s = self._smoother.step(v, s, time.monotonic())

        if self._out_mode == "topic":
            # Publish unconditionally, including zeros. The bridge has a
            # command timeout and treats silence as a fault, so "stop" has
            # to be said out loud rather than implied by saying nothing.
            t = Twist()
            t.linear.x = float(v)
            t.angular.z = float(s)
            # The reverse / three-point-turn escape goes out on its own topic
            # (/cmd_vel_escape) so twist_mux ranks it above navigation, and
            # teleop / e-stop above both. Navigation stays silent meanwhile;
            # one zero on the escape topic hands back cleanly.
            if self._state in (ST_PRESTEER, ST_REVERSE, ST_MANEUVER):   # a2r_pi_update2
                self._escape_pub.publish(t)
                self._esc_active = True
                self._stopped = (abs(v) < 1e-3 and abs(s) < 1e-3)
                return
            if self._esc_active:
                self._escape_pub.publish(Twist())
                self._esc_active = False
            self._cmd_out_pub.publish(t)
            self._stopped = (abs(v) < 1e-3 and abs(s) < 1e-3)
            return

        if abs(v) < 1e-3 and abs(s) < 1e-3:
            if not self._stopped:
                self._write({"cmd": "STOP"})
                self._stopped = True
            return
        self._stopped = False
        self._write({"cmd": "DRIVE", "v": round(v, 3), "s": round(s, 3)})

    # ══════════════════════════════════════════════════════════════════
    #  RViz OUTPUT
    # ══════════════════════════════════════════════════════════════════
    def _traj_marker(self, pts, mid: int, lethal=False, best=False) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "best" if best else "candidates"
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD if pts else Marker.DELETE
        m.scale.x = 0.035 if best else 0.008
        if best:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        elif lethal:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.30
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.6, 0.45
        m.lifetime.nanosec = int(0.4e9)
        m.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in pts]
        return m

    def _publish_best(self, pts):
        self._best_pub.publish(MarkerArray(
            markers=[self._traj_marker(pts, 0, best=True)]))
        path = Path()
        path.header.frame_id = self._frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in pts:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self._path_pub.publish(path)

    def _publish_wish(self, v: float, w: float):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "desired", 0, Marker.ARROW, Marker.ADD
        v_eff = max(abs(v), self._v_floor)
        ang = math.atan(self._wheelbase * w / v_eff) if abs(v) + abs(w) > 1e-3 else 0.0
        L = 0.3 + 0.6 * min(abs(v) / self._max_mps, 1.0)
        m.points = [Point(x=0.0, y=0.0, z=0.06),
                    Point(x=L * math.cos(ang), y=L * math.sin(ang), z=0.06)]
        m.scale.x, m.scale.y, m.scale.z = 0.02, 0.05, 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.9
        self._wish_pub.publish(m)

    def _publish_status(self, text: str):
        """State, manoeuvre and closest obstacle for the live status panel."""
        now = time.monotonic()
        if now - getattr(self, "_status_t", 0.0) < 0.2:      # 5 Hz is plenty
            return
        self._status_t = now
        st = {"state": text,
              "v": round(float(getattr(self, "_exec_v", 0.0)), 2),
              "steer_deg": round(math.degrees(float(getattr(self, "_exec_st", 0.0))), 1),
              "attempt": int(getattr(self, "_esc_count", 0))}
        since = getattr(self, "_stall_since", None)
        if since is not None and text.startswith("FOLLOW"):
            st["stuck_s"] = round(now - since, 1)
            st["stuck_need_s"] = round(float(getattr(self, "_stall_time", 0.0)), 1)
        rs = math.degrees(float(getattr(self, "_rev_steer", 0.0)))
        ps = math.degrees(float(getattr(self, "_post_steer", 0.0)))
        if text == "PRE-STEER":
            st["maneuver"] = ("swinging the wheels to %+.0f deg before reversing "
                              "%.2f m, then exit %+.0f deg"
                              % (rs, float(getattr(self, "_rev_dist", 0.0)), ps))
        elif text == "ESCAPE-REVERSE":
            st["maneuver"] = ("reversing %.2f m at %+.0f deg, then exit %+.0f deg"
                              % (float(getattr(self, "_rev_dist", 0.0)), rs, ps))
        elif text.startswith("FOLLOW") and now < float(getattr(self, "_post_until", 0.0)):
            st["maneuver"] = "exit turn %+.0f deg - leaving the pocket" % ps
        elif text.startswith("BOXED"):
            hs = float(getattr(self, "_hold_since", 0.0) or 0.0)
            st["maneuver"] = ("no way out on either lock - holding still"
                              + (" %.1f s" % (now - hs) if hs else ""))
        pts = getattr(self, "_scan_pts", None)
        if pts is not None and len(pts):
            d = np.hypot(pts[:, 0], pts[:, 1])
            k = int(np.argmin(d))
            st["closest_m"] = round(float(d[k]), 2)
            st["closest_deg"] = int(round(math.degrees(math.atan2(pts[k, 1], pts[k, 0]))))
        self._status_pub.publish(String(data=json.dumps(st)))

    def _publish_state(self, text: str):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "state", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.z = 0.6
        m.scale.z = 0.12
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.text = text
        self._state_pub.publish(m)
        self._publish_status(text)

    # ══════════════════════════════════════════════════════════════════
    def destroy_node(self):
        try:
            if self._out_mode == "topic":
                # Say stop three times on the way out. Message loss on
                # shutdown is real, and the cost of a dropped stop is a
                # car that keeps going after the node is gone.
                t = Twist()
                for _ in range(3):
                    self._cmd_out_pub.publish(t)
                    time.sleep(0.02)
            else:
                self._write({"cmd": "STOP"})
                time.sleep(0.05)
                self._write({"cmd": "STOP"})
                if self._ser:
                    self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def goal_aware_exit(ang, far, goal_bearing, default, window_deg=25.0):
    """Harmony (a2r_pi_harmony.sh): leave a trap through the open gap NEAREST THE OWNER, not
    through the average of all open directions. ang: beam bearings, far: open beams (mask)."""
    if goal_bearing is None or not np.any(far):
        return default
    a_far = ang[far]
    d = np.abs((a_far - goal_bearing + np.pi) % (2 * np.pi) - np.pi)
    best = a_far[int(np.argmin(d))]
    near = far & (np.abs((ang - best + np.pi) % (2 * np.pi) - np.pi) <= np.radians(window_deg))
    return float(np.arctan2(np.mean(np.sin(ang[near])), np.mean(np.cos(ang[near]))))


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerBridgeNode()
    # a2r_merge: everything built so far (rclpy, numpy, parameters, lookup tables) lives for
    # the whole run - freeze it out of the garbage collector, so a full collection scans only
    # new objects (a full pass was a 20-35 ms stall, ~200 ms on the Pi, at random moments)
    import gc
    gc.collect()
    gc.freeze()
    # a2r_merge: SingleThreadedExecutor. The drive sender already runs in its own thread and
    # the K-turn search in another; with the GIL the 3-thread executor ran callbacks no faster,
    # but rclpy's MultiThreadedExecutor costs a lot of CPU of its own (futures + a thread pool
    # per spin). A2R_PLANNER_MT=1 brings the old one back.
    import os
    if os.environ.get("A2R_PLANNER_MT", "0") == "1":
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    else:
        executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown — sending STOP")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
