#!/usr/bin/env python3
"""
follow_manager_node.py — the "global planner" of the follow pipeline
═══════════════════════════════════════════════════════════════════════════════
Turns the tracked owner into a GOAL for the local controller (smart_planner),
which drives to it around obstacles.

    /follow/owner             (owner_fusion, base_link) ─┐
    /followme/state           (FollowMe, Jetson)         ├─> follow_manager ─> /follow/goal
    /followme/owner_lost      (FollowMe, Jetson)         │
    /followme/owner_last_seen (FollowMe, Jetson)        ─┘

FOLLOW   owner position fresh and FollowMe says FOLLOW/SEARCH (or owner_lost):
         goal = follow_distance_m short of the owner, as before.

SEARCH   (new, v2.1) owner lost - FollowMe says so on /followme/owner_lost
         (patch_ext_search, search_mode:=external) or its state is SEARCH -
         and no fresh owner position. FollowMe is SILENT on /cmd_vel then, so
         the Pi alone drives the search, through the same planner, with full
         obstacle avoidance and the collision monitor:
           APPROACH  go to where the owner was last seen (kept in the odom
                     frame, so it stays put while the car moves), stop
                     search_standoff_m short of it
           SWEEP     full-lock arcs, first toward the side the owner left by,
                     then alternating every search_sweep_s, so the camera
                     pans the room
           GAVE UP   after search_total_s: stand still until FollowMe or
                     owner_fusion sees the owner again
         The moment owner_fusion has the owner again, FOLLOW takes over.

STRAIGHT FOLLOWING (v2.2): the owner is tracked in the odom frame, so the
         car's own turning no longer looks like the owner moving sideways
         (that fed the lead term and made the car weave). The odom track is
         low-pass filtered against camera/lidar jitter.

TRAPS (v2.2, VFH+T memory at the goal level): the planner's VFH+T detector
         (Neamah, Elek & Korondi 2024) only sees a pocket once inside it, and
         forgets it 12 s later - so the goal led straight back in. Here every
         escape manoeuvre (PRE-STEER / ESCAPE-REVERSE on /local_planner/status)
         and every enclosure seen ahead is REMEMBERED as a disc in odom for
         trap_ttl_s. Whenever the line to the goal runs into a remembered trap,
         or the goal direction is enclosed (VFH polar histogram: +-15 deg core
         and 85 % of +-50 deg blocked within 2.5 m), the goal is replaced by a detour waypoint in
         the free valley nearest the goal that does not lead into a trap.
         v2.3: only HOLLOW obstacles count (concavity, as VFH+T eq. 51) - a
         flat wall or sofa blocks but is no trap; search sweeps never create
         traps; overlapping traps merge; detour side held >= 2 s.
"""
import json
import math
import time

import numpy as np

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path                      # a2r_pi_update
from rclpy.node import Node
from std_msgs.msg import Bool, String
try:
    from sensor_msgs.msg import LaserScan
    from rclpy.qos import qos_profile_sensor_data
except ImportError:  # pragma: no cover
    LaserScan = qos_profile_sensor_data = None

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:  # pragma: no cover
    Buffer = TransformListener = None

try:                                                    # a2r_v121
    from a2r_bot_follow.owner_kf import OwnerKF, TRACKING, PREDICTING, LOST
except ImportError:  # pragma: no cover - source tree
    from owner_kf import OwnerKF, TRACKING, PREDICTING, LOST


def compute_goal(ox: float, oy: float, follow_d: float):
    """Owner (ox, oy) in base_link -> goal (gx, gy, yaw) in base_link."""
    d = math.hypot(ox, oy)
    yaw = math.atan2(oy, ox)
    if d <= follow_d or d < 1e-6:
        return 0.0, 0.0, yaw                    # close enough: hold
    k = (d - follow_d) / d
    return ox * k, oy * k, yaw


class SearchCore:
    """Pure search logic (no ROS): where to send the car while the owner is lost."""

    def __init__(self, standoff=0.6, approach_s=8.0, sweep_s=6.0, total_s=30.0,
                 radius=1.5, bearing_deg=110.0, reach_m=0.3, roam_s=6.0, roam_m=2.0):
        self.standoff, self.approach_s, self.sweep_s = standoff, approach_s, sweep_s
        self.total_s, self.radius, self.reach = total_s, radius, reach_m
        self.bearing = math.radians(bearing_deg)
        self.bearing_deg = bearing_deg
        self._ob_cache = {}                    # v2.5: side -> (time, bearing)
        # v2.7 ROAM: after each left+right sweep pair, drive on (roam_s) in the
        # direction the owner was WALKING, then sweep again - until found
        self.roam_s, self.roam_m = roam_s, roam_m
        self.dir_turn, self.roams = 0, 0
        self._rb_cache = None
        self.active = False
        self.done = False           # gave up: do not restart until the owner is back
        self.phase = "IDLE"
        self.side = 1.0
        self.t0 = self.phase_t = 0.0
        # a2r_step3: which directions (odom yaw, 12 x 30 deg) the CAMERA has already looked at in this
        # search. The sweep goes to the nearest direction not yet seen - both sides get checked,
        # the owner's side first - instead of alternating sides on a 6 s clock.
        self.cov_n, self.cov_hfov = 12, math.radians(55.0)
        self.cov, self.yaw, self._cov_roams = [False] * 12, None, -1

    def start(self, now, have_target: bool, side: float):
        self.active, self.done = True, False
        self.t0 = self.phase_t = now
        self.side = 1.0 if side >= 0.0 else -1.0
        self.phase = "APPROACH" if have_target else "SWEEP"

    def _open_bearing(self, now, side, pts, clear_m=1.0, half_w=0.32):
        """v2.5: a sweep bearing on `side` whose first metre is open (log: the
        fixed 110 deg sweep drove a full-lock arc into a wall and stalled).
        Tries 110 deg first, then nearby bearings; cached 0.5 s per side."""
        if pts is None or len(pts) == 0:
            return self.bearing * side
        c = self._ob_cache.get(side)
        if c is not None and now - c[0] < 0.5:
            return c[1]
        b = None
        for deg in (self.bearing_deg, 90.0, 130.0, 70.0, 150.0, 50.0):
            if _ray_clear(pts, math.radians(deg) * side, clear_m, half_w):
                b = math.radians(deg) * side
                break
        self._ob_cache[side] = (now, b)
        return b

    def reset(self):
        self.active, self.done, self.phase = False, False, "IDLE"

    def observe(self, yaw, yaw_rate=0.0, max_rate=1.0):
        """a2r_step3: the camera looks along odom yaw `yaw`: mark those sectors as seen - only while
        the car turns slowly enough for a sharp picture (a blurred sweep has not really looked)."""
        self.yaw = yaw
        if not self.active or yaw is None or abs(yaw_rate) > max_rate:
            return
        w = 2.0 * math.pi / self.cov_n
        for i in range(self.cov_n):
            if abs(_angdiff(-math.pi + (i + 0.5) * w, yaw)) <= 0.5 * self.cov_hfov:
                self.cov[i] = True

    def _cov_bearing(self, now, pts):
        """a2r_step3: (relative bearing of the nearest unseen direction, done?). The owner's side wins
        ties; a walled side is turned the other way round."""
        if self.yaw is None:
            return None, False
        if self._cov_roams != self.roams:          # after every ROAM: look everywhere again
            self.cov, self._cov_roams, self._cov_tgt = [False] * self.cov_n, self.roams, None
        w, tgt = 2.0 * math.pi / self.cov_n, getattr(self, "_cov_tgt", None)
        if tgt is None or self.cov[tgt]:            # keep one target until it is seen (no L/R flicker)
            best = None
            for i in range(self.cov_n):
                if self.cov[i]:
                    continue
                b = _angdiff(-math.pi + (i + 0.5) * w, self.yaw)
                cost = abs(b) - (0.35 if b * self.side > 0.0 else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, i)
            if best is None:
                self._cov_tgt = None
                return None, True
            tgt = self._cov_tgt = best[1]
        b = _angdiff(-math.pi + (tgt + 0.5) * w, self.yaw)
        b = max(-self.bearing, min(self.bearing, b))
        if pts is not None and len(pts) and not _ray_clear(pts, b, 1.0, 0.32):
            s = 1.0 if b >= 0.0 else -1.0
            ob = self._open_bearing(now, s, pts)
            if ob is None:
                ob = self._open_bearing(now, -s, pts)
            b = ob if ob is not None else b
        return b, False

    def _roam_bearing(self, now, dir_b, pts, clear_m=1.2, half_w=0.32):
        """v2.7: the open bearing nearest the owner's walking direction
        (dir_b, base frame; turned +90 deg per earlier blocked attempt)."""
        pref = (dir_b if dir_b is not None else 0.0) + self.dir_turn * math.pi / 2
        pref = (pref + math.pi) % (2.0 * math.pi) - math.pi
        c = self._rb_cache
        if c is not None and now - c[0] < 0.5 and abs(c[2] - pref) < 1e-6:
            return c[1]
        b = None
        for off in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
            cand = pref + math.radians(off)
            if pts is None or len(pts) == 0 or _ray_clear(pts, cand, clear_m, half_w):
                b = cand
                break
        self._rb_cache = (now, b, pref)
        return b

    def step(self, now, target, pts=None, dir_b=None):
        """target: last-seen point in the CURRENT base_link, or None.
        Returns (phase, (gx, gy, yaw)) or (phase, None) once it has given up."""
        if not self.active:
            return self.phase, None
        if self.total_s > 0.0 and now - self.t0 > self.total_s:   # v2.7: total_s <= 0 = never give up
            self.active, self.done, self.phase = False, True, "GAVE_UP"
            return self.phase, None
        if self.phase == "APPROACH":
            if target is None or now - self.phase_t > self.approach_s:
                self.phase, self.phase_t = "SWEEP", now
            else:
                gx, gy, yaw = compute_goal(target[0], target[1], self.standoff)
                if math.hypot(gx, gy) > self.reach:
                    return "APPROACH", (gx, gy, yaw)
                self.phase, self.phase_t = "SWEEP", now     # reached the spot
        # v2.7 SEARCH CYCLE: one sweep to each side, then ROAM on in the
        # direction the owner walked, then sweep again ... until found (log:
        # it swept on the spot, then "search gave up" at 30 s).
        cov_b, cov_done = self._cov_bearing(now, pts) if self.phase == "SWEEP" else (None, False)   # a2r_step3
        sweep_max = (3.0 if self.yaw is not None else 2.0) * self.sweep_s
        if self.phase == "SWEEP" and (cov_done or now - self.phase_t >= sweep_max):
            self.phase, self.phase_t = "ROAM", now
            self.roams += 1
        if self.phase == "ROAM":
            if now - self.phase_t >= self.roam_s:
                self.phase, self.phase_t = "SWEEP", now
            else:
                b = self._roam_bearing(now, dir_b, pts)
                if b is not None:
                    return "ROAM", (self.roam_m * math.cos(b), self.roam_m * math.sin(b), b)
                self.phase, self.phase_t = "SWEEP", now   # that way is blocked: sweep,
                self.dir_turn += 1                         # and roam another way next time
        if cov_b is None and self.phase == "SWEEP" and self.yaw is not None:
            cov_b, _ = self._cov_bearing(now, pts)       # a2r_step3: (ROAM just ended)
        if cov_b is not None:
            return ("SWEEP_L" if cov_b > 0 else "SWEEP_R"), \
                (self.radius * math.cos(cov_b), self.radius * math.sin(cov_b), cov_b)
        k = int((now - self.phase_t) // max(self.sweep_s, 0.5))
        side = self.side if k % 2 == 0 else -self.side
        b = self._open_bearing(now, side, pts)
        if b is None:                          # v2.5: this side is a wall - sweep the other way
            side = -side
            b = self._open_bearing(now, side, pts)
        if b is None:
            b = self.bearing * side            # boxed in: the planner's escape takes over
        return ("SWEEP_L" if side > 0 else "SWEEP_R"), \
            (self.radius * math.cos(b), self.radius * math.sin(b), b)


def _ray_clear(pts, b, length, half_w):
    """True if no point lies within half_w of the ray at bearing b, 0..length.
    a2r_merge: numpy instead of a Python loop over every scan point."""
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return True
    cx, cy = math.cos(b), math.sin(b)
    t = P[:, 0] * cx + P[:, 1] * cy
    lat = np.abs(P[:, 1] * cx - P[:, 0] * cy)
    return not bool(np.any((t > 0.0) & (t < length) & (lat < half_w)))


def _ray_hit(pts, b, length, half_w):
    """a2r_v121: distance along bearing b to the first point within half_w of the ray, or None."""
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return None
    cx, cy = math.cos(b), math.sin(b)
    tt = P[:, 0] * cx + P[:, 1] * cy
    lat = np.abs(P[:, 1] * cx - P[:, 0] * cy)
    m = (tt > 0.0) & (tt < length) & (lat < half_w)
    return float(tt[m].min()) if np.any(m) else None


def _angdiff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def vfh_route(pts, goal, traps, robot_r=0.32, look=2.5, sector_deg=5.0,
              enc_half_deg=50.0, enc_frac=0.85, valley_free_m=1.2,
              min_valley_deg=20.0, prev_side=0.0, side_hyst=0.25, clearance=0.30,
              body_r=0.22, detect=True, pocket_depth=0.35):
    """VFH-style goal routing with trap memory (pure, base_link frame).

    pts    obstacle points [(x, y)] from the scan
    goal   (gx, gy)
    traps  [(x, y, r)] remembered dead ends
    1. Builds a polar obstacle histogram (VFH: per-sector nearest range,
       obstacles enlarged by the robot radius, traps enlarged by r + radius).
    2. Dead end ahead = the +-15 deg core toward the goal AND enc_frac of
       +-enc_half_deg blocked within look - a corridor (walls beside, open
       ahead) is not one. Its points become a trap (centroid, spread).
    3. If a trap lies on the straight line to the goal, aim at a VIA POINT
       beside it (perpendicular, r + robot radius + clearance out), on one
       committed side; the side only changes if its via point is blocked.
    Returns None (drive straight to the goal) or
      {"bearing", "dist", "side", "reason", "enclosure": (x, y, r) or None}."""
    n = int(round(360.0 / sector_deg))
    sec = lambda a: int(((a + math.pi) / (2.0 * math.pi)) * n) % n      # noqa: E731
    minr = [math.inf] * n          # enlarged by robot_r: is a gap drivable?
    menc = [math.inf] * n          # enlarged by body_r: is it really a dead end?
    step = 2.0 * math.pi / n

    # VFH enlargement: a sector is blocked if its CENTRE lies in the enlarged arc
    # (rounding whole sectors up grew corridor walls shut). a2r_merge: one numpy
    # (sectors x points) pass instead of a Python loop per point per sector - same result.
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(P):
        dP = np.hypot(P[:, 0], P[:, 1])
        sel = (dP > 0.05) & (dP < look + 1.0)
        P, dP = P[sel], dP[sel]
    if len(P):
        aP = np.arctan2(P[:, 1], P[:, 0])
        jc = -math.pi + (np.arange(n) + 0.5) * step
        diff = np.abs((jc[:, None] - aP[None, :] + math.pi) % (2.0 * math.pi) - math.pi)
        for half_r, hist in ((robot_r, minr), (body_r, menc)):
            half = np.arcsin(np.minimum(1.0, half_r / dP))
            cand = np.where(diff <= half[None, :], dP[None, :], np.inf).min(axis=1)
            for j in np.flatnonzero(np.isfinite(cand)):
                hist[j] = float(cand[j])

    gx, gy = goal
    bg, dg = math.atan2(gy, gx), math.hypot(gx, gy)
    if dg < 0.3:
        return None
    traps = list(traps)

    # ── dead end ahead? (enclosure of the goal direction) ──
    enclosure = None
    k = int(round(enc_half_deg / sector_deg))
    win = [(sec(bg) + j) % n for j in range(-k, k + 1)]
    kc = int(round(15.0 / sector_deg))
    core = [(sec(bg) + j) % n for j in range(-kc, kc + 1)]
    centre = [(sec(bg) + j) % n for j in (-1, 0, 1)]
    lim = min(look, dg)
    if (all(menc[i] < lim for i in centre)                       # a doorway leaves the centre open
            and sum(1 for i in core if menc[i] < lim) >= 0.8 * len(core)
            and sum(1 for i in win if menc[i] < lim) >= enc_frac * len(win)):
        ps = [(x, y) for x, y in pts if math.hypot(x, y) < lim
              and abs(_angdiff(math.atan2(y, x), bg)) <= math.radians(enc_half_deg)]
        if ps and detect:
            # v2.3 concavity (as VFH+T eq. 51): the obstacle must lie BEYOND the
            # chord between its two ends - a pocket. A flat wall or a sofa lies
            # ON the chord: it blocks the way but is no trap (log: 12 "traps").
            ps.sort(key=lambda p: math.atan2(p[1], p[0]))
            (ax, ay), (bx, by) = ps[0], ps[-1]
            cl = math.hypot(bx - ax, by - ay)
            if cl > 1e-6:
                nx, ny = (by - ay) / cl, -(bx - ax) / cl     # chord normal ...
                if nx * (-ax) + ny * (-ay) > 0.0:             # ... pointing away from the car
                    nx, ny = -nx, -ny
                depth = [(px - ax) * nx + (py - ay) * ny for px, py in ps]
                deep = sum(1 for dd in depth if dd > 0.15)
                if max(depth) < pocket_depth or deep < 0.3 * len(ps):
                    ps = []                                  # not hollow: not a trap
        if ps and detect:
            cx = sum(p[0] for p in ps) / len(ps)
            cy = sum(p[1] for p in ps) / len(ps)
            spread = sum(math.hypot(px - cx, py - cy) for px, py in ps) / len(ps)
            r = max(0.4, min(1.0, 1.2 * spread))
            enclosure = (cx, cy, r)
            traps.append(enclosure)

    # ── first trap on the straight line to the goal ──
    hit = None
    for tx, ty, r in traps:
        t = (tx * gx + ty * gy) / dg                       # along the goal line
        if t <= 0.0 or t > dg + r:
            continue
        if math.hypot(tx - gx, ty - gy) < r + 0.15:        # v3.0: goal (owner / last-seen legs)
            continue                                       # is INSIDE this pocket -> go in, no detour
        px, py = gx * min(t, dg) / dg, gy * min(t, dg) / dg
        dt = math.hypot(tx, ty)
        if math.hypot(tx - px, ty - py) < r + robot_r and dt > 0.3:
            if hit is None or t < hit[3]:
                hit = (tx, ty, r, t)
    if hit is None:
        return None if enclosure is None else \
            {"bearing": None, "dist": 0.0, "side": 0.0, "reason": "dead end, no way round",
             "enclosure": enclosure}
    tx, ty, r, _ = hit
    dt = math.hypot(tx, ty)
    off = r + robot_r + clearance

    def via(side):
        """Nearest via point beside the trap whose straight approach is open."""
        nx, ny = -ty / dt * side, tx / dt * side            # perpendicular to robot->trap
        for extra in (0.0, 0.3, 0.6, 1.0):                  # push out until the line clears
            o = off + extra
            vx, vy = tx + nx * o, ty + ny * o
            dv = math.hypot(vx, vy)
            bv = math.atan2(vy, vx)
            if minr[sec(bv)] > min(dv, valley_free_m):
                return True, bv, dv
        # shadowed: the nearest open VFH valley on this side, turning outward
        # from the via bearing (never more than 150 deg off the goal)
        b0 = math.atan2(ty + ny * off, tx + nx * off)
        need = int(math.ceil(min_valley_deg / sector_deg))
        for j in range(1, n // 2):
            a = b0 + side * j * step
            if abs(_angdiff(a, bg)) > math.radians(150.0):
                break
            if all(minr[sec(a + side * q * step)] > valley_free_m for q in range(need)):
                a += side * (need // 2) * step               # aim into the middle of the gap
                return True, a, valley_free_m
        return False, 0.0, 0.0

    sides = [prev_side, -prev_side] if prev_side else \
        sorted((1.0, -1.0), key=lambda sd: abs(_angdiff(via(sd)[1], bg)))
    for sd in sides:
        ok, bv, dv = via(sd)
        if ok:
            return {"bearing": bv, "dist": dv, "side": sd,
                    "reason": "dead end ahead" if enclosure else "trap in the way",
                    "enclosure": enclosure}
    # both sides shut: the planner's escape reverses out - but the trap is reported
    return None if enclosure is None else \
        {"bearing": None, "dist": 0.0, "side": 0.0, "reason": "dead end, no way round",
         "enclosure": enclosure}


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class FollowManager(Node):
    def __init__(self):
        super().__init__("follow_manager")
        p = self.declare_parameter
        p("owner_topic", "/follow/owner")
        p("followme_state_topic", "/followme/state")
        p("goal_topic", "/follow/goal")
        p("follow_distance_m", 1.2)          # stand-off from the owner
        p("owner_timeout_s", 1.5)            # bridge camera gaps (was 0.5: flapped)
        p("state_timeout_s", 3.0)            # was 2.0: Wi-Fi gaps flapped INACTIVE
        p("lead_s", 0.3)                     # aim where the owner will be
        p("occlusion_coast_s", 2.0)          # a2r_v12: predict the owner this long after the last fix
        p("coast_max_speed_mps", 1.2)        # a2r_v12: cap on the predicted walking speed
        p("coast_goal_max_m", 1.0)           # a2r_v12: shorter goal while coasting = slower
        p("los_check", True)                 # a2r_v12: obstacle between car and owner -> goal behind it
        p("los_half_width_m", 0.18)          # a2r_v12: corridor checked along the line of sight
        p("los_standoff_m", 0.6)             # a2r_v12: stand-off while the line of sight is blocked
        p("kf_q_acc", 0.6)                   # a2r_v121: owner accel noise (m/s^2) - sets how fast the gate grows
        p("kf_sigma_meas_m", 0.08)           # a2r_v121: keeper/camera position noise
        p("kf_gate_max_m", 0.7)              # a2r_v121: physical cap on the association gate
        p("kf_max_sigma_m", 1.0)             # a2r_v121: prediction is dropped once its 1-sigma exceeds this
        p("meas_latency_s", 0.25)            # a2r_v121: assumed age when the stamp is unusable (clock skew)
        p("meas_max_age_s", 1.0)             # a2r_v121: stamps older/newer than this are not trusted
        p("los_hold_s", 0.5)                 # a2r_v121: line-of-sight 'blocked' is held this long (no flicker)
        p("jetson_takeover", True)           # a2r_v121: Pi prediction overrides a Jetson hold/last-known path
        p("goal_rate_hz", 20.0)
        p("require_followme_state", True)
        p("follow_states", "FOLLOW,SEARCH")
        p("blocked_owner_states", "NO_OWNER,CLAIMING")
        # ── search (pairs with patch_ext_search on the Jetson) ──
        p("search_enabled", True)
        p("owner_lost_topic", "/followme/owner_lost")
        p("last_seen_topic", "/followme/owner_last_seen")
        p("owner_lost_timeout_s", 2.0)       # FollowMe heartbeats owner_lost at 2 Hz
        p("search_states", "SEARCH")         # also search on these tokens (unpatched Jetson)
        p("search_standoff_m", 1.0)          # v2.4: was 0.6 - never closer than following
        p("search_approach_s", 8.0)
        p("search_sweep_s", 6.0)
        p("search_total_s", 0.0)             # v2.7: 0 = never give up (was 30)
        p("search_roam_s", 6.0)              # v2.7: drive on this long between sweep pairs
        p("search_roam_m", 2.0)
        p("search_start_s", 2.5)             # v2.4: owner unseen this long AFTER owner_timeout before searching
        p("search_radius_m", 1.5)
        p("search_bearing_deg", 110.0)       # > planner goal_turn_bearing_deg (75): full-lock arcs
        # ── v2.2 straight following + trap memory ──
        p("owner_smooth_alpha", 0.35)        # odom-frame low-pass on the owner track (1 = off)
        p("scan_topic", "/scan")
        p("planner_status_topic", "/local_planner/status")
        p("trap_enabled", True)
        p("trap_radius_m", 0.7)              # remembered dead end = disc of this radius
        p("trap_ttl_s", 90.0)                # v2.3: was 180 - a room changes, so does the owner
        p("trap_max", 6)                     # v2.3: was 12
        p("detour_side_lock_s", 2.0)         # v2.3: keep the detour side at least this long
        p("route_look_m", 2.5)               # enclosure test range (sim: 2.0 spotted the U too late)
        p("route_enclosure_deg", 50.0)       # +-this around the goal bearing ...
        p("route_enclosure_frac", 0.85)      # ... this blocked = dead end ahead
        p("route_valley_free_m", 1.2)        # a valley must be open this far
        p("route_min_valley_deg", 20.0)      # and at least this wide
        p("route_robot_radius_m", 0.32)      # > the planner's avoid clearance, so the gap is drivable
        p("detour_dist_m", 1.4)              # waypoint distance along the chosen valley
        p("detour_hold_s", 1.0)              # straight path must stay clear this long to end it
        p("detour_max_s", 12.0)
        p("route_rate_hz", 5.0)
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("path_topic", "/owner/path")             # a2r_pi_update: the Jetson's A* path to the owner
        p("owner_status_topic", "/owner/status")   # its heartbeat + mode (FOLLOW | LAST_KNOWN | LOST ...)
        p("path_timeout_s", 0.6)                   # an older path is no target
        p("planner_alive_s", 3.0)                  # no /owner/status this long = no planner: old behaviour
        p("lookahead_m", 0.9)                      # goal = this far along the path
        p("path_requires_followme_state", False)  # a2r_pi_update2: FollowMe FSM gates the path?
        p("path_fallback_owner_s", 1.5)
        p("hold_states", "WAIT,NO_OWNER,CLAIMING")  # a2r_pi_update4: FollowMe states that ALWAYS stop the car
        p("phone_follow", True)                    # a2r_pi_update3: follow the phone PDR estimate when lost
        p("phone_topic", "/phone_owner/estimate")  # a2r_pi_update3: phone_pdr_node (Jetson), odom frame
        p("phone_timeout_s", 0.6)                  # a2r_pi_update3: older estimate = no target
        p("mode", "indoor")                        # a2r_step3: indoor | outdoor (launch mode:=, phone page)
        p("mode_topic", "/a2r/mode")
        p("gps_topic", "/gps/owner_estimate")     # a2r_step3: gps_owner_node, odom, point.z = sigma (m)
        p("gps_timeout_s", 12.0)
        p("gps_arrive_m", 4.0)                    # closer than max(this, 1.5 sigma): search here instead
        p("gps_grace_s", 2.0)                     # a2r_step3g: owner unseen this long -> GPS beats the Jetson path
        # a2r_merge: owner lost while the Jetson planner runs -> go to where they were last seen
        # (lidar legs, phone, the end of the Jetson's path), then sweep / roam until found
        p("search_with_planner", True)           # a2r_pi_update2: planner FOLLOW, no path this long -> owner goal (0 = off)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.follow_d = float(g("follow_distance_m"))
        self.owner_to = float(g("owner_timeout_s"))
        self.state_to = float(g("state_timeout_s"))
        self.lead = float(g("lead_s"))
        self.coast_s = float(g("occlusion_coast_s"))           # a2r_v12
        self.coast_v = float(g("coast_max_speed_mps"))
        self.coast_goal = float(g("coast_goal_max_m"))
        self.los_on = bool(g("los_check"))
        self.los_hw = float(g("los_half_width_m"))
        self.los_sd = float(g("los_standoff_m"))
        self.kf = OwnerKF(q_acc=float(g("kf_q_acc")), sigma_meas=float(g("kf_sigma_meas_m")),   # a2r_v121
                          fresh_s=self.owner_to, predict_s=self.coast_s,
                          max_sigma_m=float(g("kf_max_sigma_m")), gate_max_m=float(g("kf_gate_max_m")))
        self.meas_lat, self.meas_max_age = float(g("meas_latency_s")), float(g("meas_max_age_s"))
        self.los_hold = float(g("los_hold_s"))
        self.jetson_takeover = bool(g("jetson_takeover"))
        self._los_until, self._takeover, self._kf_state = -math.inf, False, LOST
        self.require_state = bool(g("require_followme_state"))
        split = lambda v: {x.strip().upper() for x in str(v).split(",") if x.strip()}  # noqa: E731
        self.follow_states = split(g("follow_states"))
        self.blocked = split(g("blocked_owner_states"))
        self.hold_states = split(g("hold_states"))   # a2r_pi_update4
        self.search_enabled = bool(g("search_enabled"))
        self.search_states = split(g("search_states"))
        self.lost_to = float(g("owner_lost_timeout_s"))
        self.search_start = float(g("search_start_s"))
        self.odom_frame, self.base_frame = str(g("odom_frame")), str(g("base_frame"))
        self.alpha = max(0.05, min(1.0, float(g("owner_smooth_alpha"))))
        self.trap_on = bool(g("trap_enabled"))
        self.trap_r, self.trap_ttl, self.trap_max = float(g("trap_radius_m")), float(g("trap_ttl_s")), int(g("trap_max"))
        self.rt = dict(robot_r=float(g("route_robot_radius_m")), look=float(g("route_look_m")),
                       enc_half_deg=float(g("route_enclosure_deg")), enc_frac=float(g("route_enclosure_frac")),
                       valley_free_m=float(g("route_valley_free_m")), min_valley_deg=float(g("route_min_valley_deg")))
        self.detour_d, self.detour_hold, self.detour_max = float(g("detour_dist_m")), float(g("detour_hold_s")), float(g("detour_max_s"))
        self.route_dt = 1.0 / max(0.5, float(g("route_rate_hz")))
        self.side_lock = float(g("detour_side_lock_s"))
        self.traps = []             # [{"x","y","r","t"}] in odom
        self.scan_pts, self.scan_t, self.scan_tf = [], -math.inf, None
        self.detour = None          # {"bearing_odom", "side", "t0", "clear_since", "reason"}
        self._route_t, self._route_res = -math.inf, None
        self._planner_state = ""
        self.owner_odom = None      # filtered (x, y, t) in odom
        self.vel_odom = (0.0, 0.0)
        self.search = SearchCore(float(g("search_standoff_m")), float(g("search_approach_s")),
                                 float(g("search_sweep_s")), float(g("search_total_s")),
                                 float(g("search_radius_m")), float(g("search_bearing_deg")),
                                 roam_s=float(g("search_roam_s")), roam_m=float(g("search_roam_m")))

        self.owner = None           # (x, y, t) base_link
        self.vel = (0.0, 0.0)
        self.state_tokens, self.state_t = set(), -math.inf
        self.lost_flag, self.lost_t = False, -math.inf
        self.last_known = None      # {"odom": (x,y)|None, "base": (x,y), "t": t, "side": +-1}
        self._last_status = ""

        self.tf_buf = None
        if Buffer is not None:
            self.tf_buf = Buffer()
            self._tf_listener = TransformListener(self.tf_buf, self)

        self.pub_goal = self.create_publisher(PoseStamped, g("goal_topic"), 10)
        self.pub_status = self.create_publisher(String, "/follow/status", 10)
        # a2r_v8 test hold: scripts/a2r_test.py (lab mode) sends its own goal; pause ours meanwhile
        self._test_hold = False
        from rclpy.qos import DurabilityPolicy as _Dur, QoSProfile as _QoS
        self.create_subscription(Bool, "/test/hold", self._on_test_hold,
                                 _QoS(depth=1, durability=_Dur.TRANSIENT_LOCAL))
        self.create_subscription(PointStamped, g("owner_topic"), self._on_owner, 1)  # newest only
        self.create_subscription(String, g("followme_state_topic"), self._on_state, 10)
        self.create_subscription(Bool, g("owner_lost_topic"), self._on_lost, 10)
        self.create_subscription(PointStamped, g("last_seen_topic"), self._on_last_seen, 10)
        self.create_subscription(String, g("planner_status_topic"), self._on_planner_status, 10)
        # a2r_pi_update: the Jetson planner's path
        self.path_to, self.alive_to = float(g("path_timeout_s")), float(g("planner_alive_s"))
        self.lookahead = float(g("lookahead_m"))
        self.path_need_state = bool(g("path_requires_followme_state"))   # a2r_pi_update2
        self.owner_fb_s = float(g("path_fallback_owner_s"))
        self.phone_to, self._phone = float(g("phone_timeout_s")), None   # a2r_pi_update3
        if bool(g("phone_follow")):
            self.create_subscription(PointStamped, g("phone_topic"), self._on_phone, 10)
        # a2r_step3: indoor / outdoor mode (latched topic from gps_owner) + the GPS owner estimate
        self.mode = "outdoor" if str(g("mode")).strip().lower() == "outdoor" else "indoor"
        self._gps, self.gps_to, self.gps_arrive = None, float(g("gps_timeout_s")), float(g("gps_arrive_m"))
        self.gps_grace = float(g("gps_grace_s"))     # a2r_step3g
        self._cov_prev = None
        from rclpy.qos import DurabilityPolicy, QoSProfile
        self.create_subscription(String, g("mode_topic"), self._on_mode,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(PointStamped, g("gps_topic"), self._on_gps, 10)
        self._gps_path, self._gps_path_t, self._gps_k, self._gps_path_id = None, -math.inf, 0, None   # a2r_step3f
        self.create_subscription(Path, "/gps/owner_path", self._on_gps_path, 10)
        self.search_with_planner = bool(g("search_with_planner"))   # a2r_merge
        self._target_t = -math.inf                 # a2r_merge: last time ANY target (path/owner/phone) drove
        self._nopath_since, self._path_odom = None, None
        self._path, self._path_t, self._path_yaw = None, -math.inf, None
        self._planner_t, self._planner_mode, self._path_tf = -math.inf, "", {}
        self.create_subscription(Path, g("path_topic"), self._on_path, 1)
        self.create_subscription(String, g("owner_status_topic"), self._on_owner_status, 10)
        if LaserScan is not None:
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data)
        self.create_timer(1.0 / max(1.0, float(g("goal_rate_hz"))), self._tick)
        self.get_logger().info(
            f"follow_manager up: stand-off {self.follow_d:.2f} m, search "
            f"{'ON' if self.search_enabled else 'OFF'} ({g('owner_lost_topic')}), "
            f"trap memory {'ON' if self.trap_on else 'OFF'}")

    # ── frames ────────────────────────────────────────────────────────────
    def _tf(self, target, source):
        if self.tf_buf is None:
            return None
        try:
            t = self.tf_buf.lookup_transform(target, source, rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, _yaw(tr.rotation)

    @staticmethod
    def _apply(T, x, y):
        tx, ty, yaw = T
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x - s * y, ty + s * x + c * y

    def _remember(self, x, y, now):
        T = self._tf(self.odom_frame, self.base_frame)
        self.last_known = {"odom": self._apply(T, x, y) if T else None, "base": (x, y),
                           "t": now, "side": 1.0 if y >= 0.0 else -1.0, "rp": T}

    def _last_known_in_base(self, now):
        lk = self.last_known
        if lk is None:
            return None
        if lk["odom"] is not None:
            T = self._tf(self.base_frame, self.odom_frame)
            Tr = self._tf(self.odom_frame, self.base_frame)
            rp = lk.get("rp")
            if T is not None and Tr is not None and rp is not None \
                    and math.hypot(Tr[0] - rp[0], Tr[1] - rp[1]) > 1.0 * (now - lk["t"]) + 0.5:
                T = None    # v2.5: odom jumped further than the car can drive -
                            # the remembered spot is garbage (log: "last seen 8.2 m")
            if T is not None:
                return self._apply(T, *lk["odom"])
        # no odometry: the base_link point is only valid while the car has barely moved
        return lk["base"] if now - lk["t"] < 1.0 else None

    # ── inputs ────────────────────────────────────────────────────────────
    def _meas_time(self, msg, now):
        """a2r_v121: monotonic time the measurement was TAKEN. Uses the header stamp when it is
        plausible (clocks synced), else arrival - meas_latency_s."""
        try:
            st = rclpy.time.Time.from_msg(msg.header.stamp)
            age = (self.get_clock().now() - st).nanoseconds * 1e-9
            if st.nanoseconds > 0 and -0.05 <= age <= self.meas_max_age:
                return now - max(age, 0.0), st, False
        except Exception:  # noqa: BLE001
            pass
        self._stamp_bad = getattr(self, "_stamp_bad", 0) + 1
        return now - self.meas_lat, None, True

    def _odom_at(self, stamp):
        """odom <- base at the measurement time (tf2 history), else the latest."""
        if self.tf_buf is not None and stamp is not None:
            try:
                tr = self.tf_buf.lookup_transform(self.odom_frame, self.base_frame, stamp).transform
                return tr.translation.x, tr.translation.y, _yaw(tr.rotation)
            except Exception:  # noqa: BLE001
                pass
        return self._tf(self.odom_frame, self.base_frame)

    def _on_owner(self, msg):
        now = time.monotonic()
        x, y = msg.point.x, msg.point.y
        # a2r_v121: Kalman gate in odom at the measurement time - a rejected point never becomes the owner
        t_m, stamp, guessed = self._meas_time(msg, now)
        To = self._odom_at(stamp)
        if To is not None:
            res = self.kf.update(t_m, self._apply(To, x, y))
            if res not in ("init", "accepted", "reinit"):
                if res == "gated":
                    self.get_logger().info(f"owner point rejected by the track gate ({x:.2f}, {y:.2f})",
                                           throttle_duration_sec=2.0)
                return
            if res == "reinit":
                self.get_logger().warn("owner track RE-INITIALISED after loss (3 consistent points)")
        if self.owner is not None and now - self.owner[2] < 0.5:
            dt = now - self.owner[2]
            if dt > 1e-3:
                vx, vy = (x - self.owner[0]) / dt, (y - self.owner[1]) / dt
                # base_link velocity includes the robot's own motion; heavy
                # smoothing and a small lead keep this a gentle correction
                self.vel = (0.7 * self.vel[0] + 0.3 * vx, 0.7 * self.vel[1] + 0.3 * vy)
        else:
            self.vel = (0.0, 0.0)
        self.owner = (x, y, now)
        self._remember(x, y, now)
        T = self._tf(self.odom_frame, self.base_frame)
        if T is not None:
            wx, wy = self._apply(T, x, y)
            o = self.owner_odom
            if o is None or now - o[2] > 1.0:
                self.owner_odom, self.vel_odom = (wx, wy, now), (0.0, 0.0)
            else:
                a = self.alpha
                fx, fy = o[0] + a * (wx - o[0]), o[1] + a * (wy - o[1])
                dt = max(now - o[2], 1e-3)
                vx, vy = (fx - o[0]) / dt, (fy - o[1]) / dt
                sp = math.hypot(vx, vy)
                if sp > 2.5:                     # a person does not move 2.5 m/s: clamp jumps
                    vx, vy = vx * 2.5 / sp, vy * 2.5 / sp
                self.vel_odom = (0.8 * self.vel_odom[0] + 0.2 * vx, 0.8 * self.vel_odom[1] + 0.2 * vy)
                self.owner_odom = (fx, fy, now)
                if math.hypot(*self.vel_odom) > 0.15:   # v2.7: which way the owner WALKED
                    self._owner_dir = (math.atan2(self.vel_odom[1], self.vel_odom[0]), now)

    def _on_state(self, msg):
        self.state_tokens = {t.strip().upper() for t in str(msg.data).split("|") if t.strip()}
        self.state_t = time.monotonic()

    def _on_lost(self, msg):
        self.lost_flag, self.lost_t = bool(msg.data), time.monotonic()

    def _on_last_seen(self, msg):
        now = time.monotonic()
        lk = self.last_known
        if lk is None or now - lk["t"] > 0.3:       # owner_fusion's fresher track wins
            self._remember(msg.point.x, msg.point.y, now)

    def _on_scan(self, msg):
        now = time.monotonic()
        if now - self.scan_t < self.route_dt:            # 5 Hz is plenty for routing
            return
        if self.scan_tf is None:
            self.scan_tf = self._tf(self.base_frame, msg.header.frame_id) or (-0.140, 0.0, 0.0)
        tx, ty, yaw = self.scan_tf
        # v2.8: ~240 rays, but each one the NEAREST of its group of beams.
        # It used to keep every step-th beam and skip the rest - with the
        # high-resolution lidar (~2500 beams) that threw away 9 of 10, and a
        # chair leg between two kept beams simply was not there.
        r = np.asarray(msg.ranges, dtype=np.float64)
        n = r.size
        if n == 0:
            return
        a = msg.angle_min + msg.angle_increment * np.arange(n)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(r) & (r > msg.range_min) & (r < 4.0)
        r = np.where(ok, r, np.inf)
        step = max(1, n // 240)
        m = (n // step) * step
        if m:
            rg = r[:m].reshape(-1, step)
            k = np.argmin(rg, axis=1) + np.arange(rg.shape[0]) * step
            if m < n:
                k = np.r_[k, m + int(np.argmin(r[m:]))]
        else:
            k = np.array([int(np.argmin(r))])
        k = k[np.isfinite(r[k])]
        lx, ly = r[k] * np.cos(a[k]), r[k] * np.sin(a[k])
        c, s = math.cos(yaw), math.sin(yaw)
        pts = list(zip((tx + c * lx - s * ly).tolist(), (ty + s * lx + c * ly).tolist()))
        self.scan_pts, self.scan_t = pts, now

    def _on_planner_status(self, msg):
        try:
            st = str(json.loads(msg.data).get("state", ""))
        except Exception:
            return
        if st in ("PRE-STEER", "ESCAPE-REVERSE") and \
                self._planner_state not in ("PRE-STEER", "ESCAPE-REVERSE"):
            T = self._tf(self.odom_frame, self.base_frame)
            if T is not None:                            # the dead end is just ahead of the bumper
                self._add_trap(*self._apply(T, 0.40, 0.0), why="escape")
        self._planner_state = st

    def _add_trap(self, x, y, why="", r=None):
        if not self.trap_on:
            return
        now = time.monotonic()
        rn = max(self.trap_r, min(r or self.trap_r, 1.2))
        for t in self.traps:
            gap = math.hypot(t["x"] - x, t["y"] - y)
            if gap < 0.8 * (t["r"] + rn):        # v2.3: overlapping discs become ONE trap
                # one disc covering both, so the "first trap on the line" cannot hop between them
                R = min(1.2, max(t["r"], rn, 0.5 * (gap + t["r"] + rn)))
                w = t["r"] / (t["r"] + rn)
                t["x"], t["y"] = w * t["x"] + (1 - w) * x, w * t["y"] + (1 - w) * y
                t["t"], t["r"] = now, R
                return
        self.traps.append({"x": x, "y": y, "r": rn, "t": now})
        self.traps = sorted(self.traps, key=lambda t: -t["t"])[:self.trap_max]
        self.get_logger().info(f"trap remembered ({why}) at odom ({x:.1f}, {y:.1f}) - {len(self.traps)} known")

    def _route(self, now, gx, gy, detect=True):
        """Goal (base_link) -> (gx, gy, yaw, label) - detoured if a trap is in the way."""
        if not self.trap_on or now - self.scan_t > 1.0:
            return gx, gy, math.atan2(gy, gx), ""
        self.traps = [t for t in self.traps if now - t["t"] < self.trap_ttl]
        Tb = self._tf(self.base_frame, self.odom_frame)
        traps_b = [(*self._apply(Tb, t["x"], t["y"]), t["r"]) for t in self.traps] if Tb else []
        if now - self._route_t >= self.route_dt:
            self._route_t = now
            prev = self.detour["side"] if self.detour else 0.0
            self._route_res = vfh_route(self.scan_pts, (gx, gy), traps_b, prev_side=prev, detect=detect, **self.rt)
            res = self._route_res
            if res is not None and res.get("enclosure") is not None:
                To = self._tf(self.odom_frame, self.base_frame)
                if To is not None:
                    ex, ey, er = res["enclosure"]
                    self._add_trap(*self._apply(To, ex, ey), why="dead end seen ahead", r=er)
        res = self._route_res
        if res is not None and res.get("bearing") is None:
            res = None                       # nowhere to go round: planner reverses out, trap kept
        if res is None:
            if self.detour is not None:
                if self.detour["clear_since"] is None:
                    self.detour["clear_since"] = now
                if now - self.detour["clear_since"] >= self.detour_hold:
                    self.get_logger().info("detour done - straight to the goal again")
                    self.detour = None
            if self.detour is None:
                return gx, gy, math.atan2(gy, gx), ""
        else:
            if self.detour is None:
                self.detour = {"side": res["side"], "t0": now, "side_t": now, "clear_since": None,
                               "reason": res["reason"]}
                self.get_logger().info(f"detour {'left' if res['side'] > 0 else 'right'}: {res['reason']}")
            if res["side"] != self.detour["side"]:
                if now - self.detour["side_t"] < self.side_lock:
                    res = None                   # v2.3: no L/R flip within side_lock_s (log: R-L-R in 1 s)
                else:
                    self.detour.update(side=res["side"], side_t=now)
            if res is not None:
                self.detour.update(clear_since=None, bearing=res["bearing"], dist=res["dist"])
        if now - self.detour["t0"] > self.detour_max:
            self.detour = None
            return gx, gy, math.atan2(gy, gx), ""
        b = self.detour.get("bearing", math.atan2(gy, gx))
        d = max(0.6, min(self.detour_d, self.detour.get("dist", self.detour_d)))
        return d * math.cos(b), d * math.sin(b), b, " DETOUR " + ("L" if self.detour["side"] > 0 else "R")

    # ── output ────────────────────────────────────────────────────────────
    def _status(self, text, extra=None):
        d = {"state": text}
        if extra:
            d.update(extra)
        self.pub_status.publish(String(data=json.dumps(d)))
        if text != self._last_status:
            self.get_logger().info(f"follow: {text}", throttle_duration_sec=0.5)   # a2r_v10 log
            self._last_status = text

    def _publish_goal(self, gx, gy, yaw):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y = float(gx), float(gy)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        if not self._test_hold:                     # a2r_v8 test hold
            self.pub_goal.publish(ps)

    # ── a2r_pi_update: follow the Jetson's A* path ─────────────────────────
    def _on_owner_status(self, msg):
        self._planner_t = time.monotonic()
        try:
            self._planner_mode = str(json.loads(msg.data).get("mode", ""))
        except Exception:
            self._planner_mode = ""

    def _on_path(self, msg):
        now = time.monotonic()
        self._planner_t = now
        fr = msg.header.frame_id or self.base_frame
        if fr == self.base_frame:
            T = (0.0, 0.0, 0.0)
        else:
            T = self._path_tf.get(fr) or self._tf(self.base_frame, fr)
            if T is None:
                self._path = None                      # no TF yet: no target rather than a wrong one
                return
            self._path_tf[fr] = T                      # the lidar mount is static
        pts = [self._apply(T, q.pose.position.x, q.pose.position.y) for q in msg.poses]
        yaw = None
        if msg.poses:
            o = msg.poses[-1].pose.orientation
            yaw = T[2] + 2.0 * math.atan2(o.z, o.w)
        self._path, self._path_t, self._path_yaw = pts, now, yaw
        self._path_odom = self._tf(self.odom_frame, self.base_frame)   # a2r_pi_update2

    def _lookahead(self, pts):
        """The point lookahead_m along the path, from the path point nearest the robot."""
        if len(pts) == 1:
            return pts[0]
        i0 = min(range(len(pts)), key=lambda i: pts[i][0] ** 2 + pts[i][1] ** 2)
        left, (px, py) = self.lookahead, pts[i0]
        for qx, qy in pts[i0 + 1:]:
            d = math.hypot(qx - px, qy - py)
            if d >= left:
                f = left / d
                return px + f * (qx - px), py + f * (qy - py)
            left -= d
            px, py = qx, qy
        return pts[-1]

    def _on_phone(self, msg):                       # a2r_pi_update3
        now = time.monotonic()
        self._phone = (msg.header.frame_id or self.odom_frame, msg.point.x, msg.point.y, now)
        # a2r_merge: a phone fix is also a last-seen position, unless the lidar/camera track is fresher
        lk = self.last_known
        if lk is None or now - lk["t"] > 0.3:
            ph = self._phone_in_base(now)
            if ph is not None:
                self._remember(ph[0], ph[1], now)

    def _on_test_hold(self, msg):                   # a2r_v8 test hold
        if bool(msg.data) != self._test_hold:
            self._test_hold = bool(msg.data)
            self.get_logger().info("test hold ON - owner goals paused (a2r_test lab run)"
                                   if self._test_hold else "test hold OFF - following again")

    def _on_mode(self, msg):                        # a2r_step3
        m = str(msg.data).strip().lower()
        if m in ("indoor", "outdoor") and m != self.mode:
            self.mode = m
            self.get_logger().info(f"mode: {m.upper()}")

    def _on_gps(self, msg):                         # a2r_step3
        self._gps = (msg.header.frame_id or self.odom_frame, msg.point.x, msg.point.y,
                     max(1.0, msg.point.z), time.monotonic())

    def _gps_goal(self, now, tag=""):
        """a2r_step3 OUTDOOR: camera, lidar and phone steps have nothing -> drive toward the owner's GPS
        position. Within GPS accuracy of it, the normal search (approach, look both ways, roam) takes
        over from there. True if a GPS goal was published."""
        gp = self._gps
        if self.mode != "outdoor" or gp is None or now - gp[4] > self.gps_to:
            if self.mode == "outdoor":                 # a2r_step3b: why GPS is not driving
                self.get_logger().info("outdoor: no GPS owner estimate (see /gps/status: modem, fix, "
                                       "aligned, owner_age_s)", throttle_duration_sec=15.0)
            return False
        T = (0.0, 0.0, 0.0) if gp[0] == self.base_frame else self._tf(self.base_frame, gp[0])
        if T is None:
            return False
        x, y = self._apply(T, gp[1], gp[2])
        d, sig = math.hypot(x, y), gp[3]
        lk = self.last_known
        if lk is None or now - lk["t"] > 2.0:
            self._remember(x, y, now)               # the search starts from the GPS position
        if d <= max(self.gps_arrive, 1.5 * sig):
            return False                            # inside GPS accuracy: lidar/camera search from here
        # a2r_step3f TRAJECTORY TRACKING (Lee, Cho & Lim, J. KSPE 2018, Sect. 3): follow the owner's
        # smoothed GPS trail point by point - the way they walked round buildings - and cut straight
        # to a later trail point (or to the owner) whenever the lidar sees that line clear ("direct
        # tracking"). A point is reached within the paper's eq. 14 radius (robot + owner sigma).
        mode, tx, ty = "direct", x, y
        trail = self._gps_trail_in_base(now)
        if trail:
            reach = max(1.5, 0.5 * sig)
            k = max(self._gps_k if self._gps_path_id == self._gps_path_t else 0,
                    min(range(len(trail)), key=lambda i: math.hypot(*trail[i])))
            while k < len(trail) - 1 and math.hypot(*trail[k]) <= reach:
                k += 1
            self._gps_k, self._gps_path_id = k, self._gps_path_t
            pts = self.scan_pts if now - self.scan_t < 1.0 else None
            if pts and not _ray_clear(pts, math.atan2(y, x), min(d, 6.0), 0.30):
                mode, (tx, ty) = "trail", trail[k]
                for j in range(len(trail) - 1, k, -1):      # furthest trail point reachable straight
                    dj = math.hypot(*trail[j])
                    if dj <= 6.0 and _ray_clear(pts, math.atan2(trail[j][1], trail[j][0]), dj, 0.30):
                        tx, ty = trail[j]
                        break
        if mode == "direct":
            gx, gy, yaw = compute_goal(tx, ty, self.follow_d)
        else:
            gx, gy, yaw = tx, ty, math.atan2(ty, tx)
        gx, gy, yaw, rt = self._route(now, gx, gy)
        self.search.reset()
        self._target_t = now
        self._publish_goal(gx, gy, yaw)
        self._status("ACTIVE GPS " + mode.upper() + rt + tag,
                     {"gps_m": round(d, 1), "sigma_m": round(sig, 1), "trail": len(trail) if trail else 0})
        return True

    def _on_gps_path(self, msg):                    # a2r_step3f: the owner's smoothed GPS trail (odom)
        self._gps_path = (msg.header.frame_id or self.odom_frame,
                          [(p.pose.position.x, p.pose.position.y) for p in msg.poses])
        self._gps_path_t = time.monotonic()

    def _gps_trail_in_base(self, now):
        gpth = self._gps_path
        if gpth is None or now - self._gps_path_t > self.gps_to or len(gpth[1]) < 2:
            return None
        T = self._tf(self.base_frame, gpth[0])
        return None if T is None else [self._apply(T, px, py) for px, py in gpth[1]]

    def _observe_cov(self, now):                    # a2r_step3: camera direction -> search coverage
        T = self._tf(self.odom_frame, self.base_frame)
        if T is None:
            self.search.observe(None)
            return
        rate, prev = 0.0, self._cov_prev
        if prev is not None and now - prev[0] > 1e-3:
            rate = _angdiff(T[2], prev[1]) / (now - prev[0])
        self._cov_prev = (now, T[2])
        self.search.observe(T[2], rate)

    def _phone_in_base(self, now):                  # a2r_pi_update3: fresh phone estimate in base_link, or None
        ph = self._phone
        if ph is None or now - ph[3] > self.phone_to:
            return None
        if ph[0] == self.base_frame:
            return ph[1], ph[2]
        T = self._tf(self.base_frame, ph[0])
        return None if T is None else self._apply(T, ph[1], ph[2])

    def _path_now(self, now):
        """a2r_pi_update2: the last path, moved with the car since it arrived (odom TF), or None."""
        if not self._path or now - self._path_t > self.path_to:
            return None, None
        To = getattr(self, "_path_odom", None)
        Tb = self._tf(self.base_frame, self.odom_frame) if To is not None else None
        if To is None or Tb is None:
            return self._path, self._path_yaw
        pts = [self._apply(Tb, *self._apply(To, x, y)) for x, y in self._path]
        yaw = None if self._path_yaw is None else self._path_yaw + To[2] + Tb[2]
        return pts, yaw

    def _pi_takeover(self, now):
        """a2r_v121: True = the Pi's own owner track drives this tick instead of the Jetson path.
        Only when the Pi track is TRACKING/PREDICTING and the Jetson offers no usable route
        (no path, a one-point 'hold' while the owner is clearly beyond the stand-off, or a
        LAST_KNOWN/LOST path while the Pi still has a valid prediction). FollowMe hold states win."""
        take = False
        if self.jetson_takeover:
            tokens = self.state_tokens if now - self.state_t <= self.state_to else set()
            st, kxy, _, _, _ = self.kf.status(now)
            if not (tokens & self.hold_states) and st in (TRACKING, PREDICTING) and kxy is not None:
                pth, _ = self._path_now(now)
                Tb = self._tf(self.base_frame, self.odom_frame)
                d_o = math.hypot(*self._apply(Tb, *kxy)) if Tb is not None else 0.0
                mode = self._planner_mode or ""
                take = (not pth
                        or (len(pth) == 1 and d_o > self.follow_d + 0.3)
                        or (mode != "FOLLOW" and st == PREDICTING))
        if take != self._takeover:
            self.get_logger().info("authority -> " + ("PI owner track (Jetson path held/stale)" if take
                                                      else "JETSON path"))
            self._takeover = take
        return take

    def _search_tick(self, now, tag=""):
        """a2r_merge: owner lost while the Jetson planner runs. APPROACH the last-seen position
        (lidar legs / phone / path end, kept in odom), then SWEEP and ROAM until something
        tracks the owner again. True if a search goal was published."""
        if not (self.search_with_planner and self.search_enabled) or self.last_known is None:
            return False
        seen_t = max(self._target_t, self.owner[2] if self.owner is not None else -math.inf)
        if not self.search.active and (self.search.done or now - seen_t < self.owner_to + self.search_start):
            return False
        if not self.search.active:
            tb = self._last_known_in_base(now)
            self.search.start(now, tb is not None, self.last_known["side"])
            self.get_logger().info(
                "owner lost - search: " + (f"go to last seen ({tb[0]:.1f}, {tb[1]:.1f}) m"
                                           if tb else "no last-seen fix, sweeping"))
        dir_b, od = None, getattr(self, "_owner_dir", None)
        To = self._tf(self.odom_frame, self.base_frame) if od is not None else None
        if To is not None and now - od[1] < 60.0:
            dir_b = _angdiff(od[0], To[2])
        else:
            bx, by = self.last_known["base"]
            dir_b = math.atan2(by, bx)
        self._observe_cov(now)                                  # a2r_step3
        phase, goal = self.search.step(now, self._last_known_in_base(now),
                                       self.scan_pts if now - self.scan_t < 1.0 else None, dir_b=dir_b)
        if goal is None:
            return False
        gx, gy, yaw, rtag = self._route(now, goal[0], goal[1], detect=phase == "APPROACH")
        self._publish_goal(gx, gy, yaw if rtag else goal[2])
        self._status("SEARCH " + phase + rtag + tag, {"t_s": round(now - self.search.t0, 1)})
        return True

    def _tick(self):
        now = time.monotonic()
        state_fresh = now - self.state_t <= self.state_to
        tokens = self.state_tokens if state_fresh else set()
        lost = self.lost_flag and (now - self.lost_t) <= self.lost_to
        blocked = bool(tokens & self.blocked)
        owner_fresh = self.owner is not None and now - self.owner[2] <= self.owner_to
        state_ok = (not self.require_state) or (
            state_fresh and not blocked and (bool(tokens & self.follow_states) or lost))

        # ── a2r_pi_update: the Jetson planner runs -> its path is the ONLY target ──
        if now - self._planner_t <= self.alive_to and not self._pi_takeover(now):
            pth, pyaw = self._path_now(now)          # a2r_pi_update2: moved with the car (odom)
            mode = self._planner_mode or "no status"
            # a2r_pi_update2: FollowMe's FSM (WAIT, NO_OWNER, ...) no longer vetoes the Jetson's path
            # unless path_requires_followme_state - the keeper/planner pair is the owner authority
            path_ok = state_ok or not self.path_need_state
            # a2r_pi_update4: an explicit WAIT / no owner from FollowMe always stops the car
            if state_fresh and (tokens & self.hold_states):
                self._nopath_since = None
                self.search.reset()
                self._status("HOLD (FollowMe: " + "|".join(sorted(tokens)) + ")")
                return
            if not pth or not path_ok:
                if self._nopath_since is None:
                    self._nopath_since = now
                if (path_ok and owner_fresh and self.owner_fb_s > 0.0 and mode in ("FOLLOW", "LAST_KNOWN")
                        and now - self._nopath_since >= self.owner_fb_s):
                    # the planner sees the owner but has no path (start or goal inside its
                    # inflation): steer for the owner; the Pi planner / K-turn avoids obstacles
                    gx, gy, yaw = compute_goal(self.owner[0], self.owner[1], self.follow_d)
                    self.search.reset()
                    self._target_t = now
                    self._publish_goal(gx, gy, yaw)
                    self._status("ACTIVE OWNER (Jetson planner: " + mode + ", no path)",
                                 {"owner_m": round(math.hypot(self.owner[0], self.owner[1]), 2)})
                    return
                # a2r_pi_update3: no path, no camera owner -> the phone's estimate (never for NO_OWNER)
                ph = self._phone_in_base(now) if (path_ok and not owner_fresh and mode != "NO_OWNER") else None
                if ph is not None:
                    gx, gy, yaw = compute_goal(ph[0], ph[1], self.follow_d)
                    tag = ""
                    if math.hypot(gx, gy) > 1e-6:
                        gx, gy, yaw, tag = self._route(now, gx, gy)
                    self.search.reset()
                    self._target_t = now
                    self._publish_goal(gx, gy, yaw)
                    self._status("ACTIVE PHONE (Jetson planner: " + mode + ")" + tag,
                                 {"phone_m": round(math.hypot(ph[0], ph[1]), 2)})
                    return
                # a2r_step3 OUTDOOR: the owner's GPS position before a local search
                if path_ok and not owner_fresh and mode != "NO_OWNER" and \
                        self._gps_goal(now, " (Jetson planner: " + mode + ")"):
                    return
                # a2r_merge: nothing tracks the owner right now -> search from the last-seen position
                # (it used to publish no goal at all here: the car stopped wherever it was, often
                # nose-to-wall, and waited)
                if path_ok and mode != "NO_OWNER" and self._search_tick(now, tag=" (Jetson planner: " + mode + ")"):
                    return
                if not path_ok:
                    self._status("INACTIVE (FollowMe: " + "|".join(sorted(tokens)) + "; Jetson planner: " + mode + ")")
                else:
                    self._status("INACTIVE (Jetson planner: " + mode + ", no fresh path)")
                return
            # a2r_step3g OUTDOOR: while the owner is NOT seen, the Jetson's path only leads to where they
            # were last seen (LAST_KNOWN / search). The robot followed it and ignored GPS (2026-10-10:
            # "GPS 17 m" / "GPS 22 m" on the map, robot searching the last-seen spot). Now GPS wins
            # once the owner has been unseen for gps_grace_s - unless they are within GPS accuracy.
            if self.mode == "outdoor" and not owner_fresh and mode != "NO_OWNER":
                unseen = math.inf if self.owner is None else now - self.owner[2]
                if (mode != "FOLLOW" or unseen > self.gps_grace) and \
                        self._gps_goal(now, " (over Jetson " + mode + " path)"):
                    return
            self._nopath_since = None
            self.search.reset()
            self._target_t = now
            if not owner_fresh and len(pth) > 0:          # a2r_merge: path end = where the owner is
                ex, ey = pth[-1]
                lk = self.last_known
                if lk is None or now - lk["t"] > 0.3:
                    self._remember(ex, ey, now)
            gx, gy = self._lookahead(pth)
            if owner_fresh:
                yaw = math.atan2(self.owner[1], self.owner[0])   # owner bearing (FACE-OWNER uses it)
            elif pyaw is not None:
                yaw = pyaw
            else:
                yaw = math.atan2(gy, gx)
            if len(pth) == 1:
                gx, gy = 0.0, 0.0                               # at the stand-off: hold
            self._publish_goal(gx, gy, yaw)
            self._status("ACTIVE PATH " + self._planner_mode,
                         {"points": len(pth), "goal_m": round(math.hypot(gx, gy), 2)})
            return

        # ── FOLLOW (a2r_v121: Kalman track - measured vs predicted, LOS with hysteresis) ──
        st, kxy, kv, ksig, age = self.kf.status(now)
        if st != self._kf_state:
            self.get_logger().info(f"owner track {self._kf_state} -> {st} (unseen {age:.1f} s, sigma {ksig:.2f} m)")
            self._kf_state = st
        Tb = self._tf(self.base_frame, self.odom_frame)
        coasting = st == PREDICTING and not blocked and Tb is not None
        if (owner_fresh or coasting) and (state_ok or self._takeover) and Tb is not None and kxy is not None:
            self.search.reset()
            lead = self.lead if st == TRACKING else 0.0          # prediction already runs to 'now'
            ox, oy = self._apply(Tb, kxy[0] + kv[0] * lead, kxy[1] + kv[1] * lead)
            d_o = math.hypot(ox, oy)
            fd, los = self.follow_d, ""
            if self.los_on and d_o > 0.6:
                blk = False
                if now - self.scan_t < 1.0 and self.scan_pts:
                    hit = _ray_hit(self.scan_pts, math.atan2(oy, ox), d_o - 0.35, self.los_hw)
                    # predicted owner 'inside' the obstacle = prediction ran into a wall: do NOT aim behind it
                    blk = hit is not None and (st == TRACKING or hit < d_o - 0.5)
                if blk:
                    self._los_until = now + self.los_hold
                if blk or now < self._los_until:          # hysteresis: held los_hold_s after it clears
                    fd, los = min(self.follow_d, self.los_sd), " LOS-BLOCKED"
            gx, gy, yaw = compute_goal(ox, oy, fd)
            if coasting:
                gd = math.hypot(gx, gy)
                if gd > self.coast_goal:                # slower while it is only a prediction
                    gx, gy = gx * self.coast_goal / gd, gy * self.coast_goal / gd
            if math.hypot(gx, gy) > 1e-6:
                gx, gy, yaw, tag = self._route(now, gx, gy)
            else:
                tag = ""
            self._publish_goal(gx, gy, yaw)
            self._status(("ACTIVE PREDICT" if coasting else "ACTIVE") + los + tag,
                         {"owner_m": round(d_o, 2), "goal_m": round(math.hypot(gx, gy), 2),
                          "track": st, "sigma_m": round(ksig, 2), "unseen_s": round(age, 2),
                          "authority": "pi-takeover" if self._takeover else "pi"})
            return

        # a2r_step3 OUTDOOR: the owner's GPS position before a local search
        # a2r_step3b: no 'state_fresh' here - the GPS fallback matters MOST when the Jetson (FollowMe,
        # keeper, phone page) has gone quiet; Firebase through the Pi's modem does not need the Jetson.
        if not owner_fresh and not blocked and self._gps_goal(now):
            return
        # ── SEARCH ────────────────────────────────────────────────────────
        unseen = math.inf if self.owner is None else now - self.owner[2]
        # v2.4: a short detection drop-out is not a lost owner - the log showed
        # "approach last seen" after every 1.5 s gap, driving to where you WERE
        want = (self.search_enabled and not blocked and self.last_known is not None
                and (self.search.active or unseen >= self.owner_to + self.search_start)
                and (lost or bool(tokens & self.search_states)))
        if want:
            if not self.search.active and not self.search.done:
                tb = self._last_known_in_base(now)
                self.search.start(now, tb is not None, self.last_known["side"])
                self.get_logger().info(
                    "owner lost - Pi search: " + (f"approach last seen ({tb[0]:.1f}, {tb[1]:.1f}) m"
                                                  if tb else "no last-seen fix, sweeping"))
            dir_b, od = None, getattr(self, "_owner_dir", None)
            To = self._tf(self.odom_frame, self.base_frame) if od is not None else None
            if To is not None and now - od[1] < 60.0:
                dir_b = _angdiff(od[0], To[2])          # owner's walking direction, base frame
            elif self.last_known is not None:
                bx, by = self.last_known["base"]
                dir_b = math.atan2(by, bx)              # else: where the owner was last seen
            self._observe_cov(now)                              # a2r_step3
            phase, goal = self.search.step(now, self._last_known_in_base(now),
                                           self.scan_pts if now - self.scan_t < 1.0 else None,
                                           dir_b=dir_b)
            if goal is not None:
                gx, gy, yaw, tag = self._route(now, goal[0], goal[1], detect=phase == "APPROACH")
                self._publish_goal(gx, gy, yaw if tag else goal[2])
                self._status("SEARCH " + phase + tag, {"t_s": round(now - self.search.t0, 1)})
                return
            self._status("INACTIVE (search gave up - waiting for the owner)")
            return
        if not lost and not (tokens & self.search_states):
            self.search.reset()                     # FollowMe stopped searching: re-arm

        # ── INACTIVE: say why ─────────────────────────────────────────────
        if self.require_state and not state_fresh:
            self._status("INACTIVE (no /followme/state)")
        elif self.require_state and not state_ok:
            self._status("INACTIVE (FollowMe: " + "|".join(sorted(tokens)) + ")")
        else:
            self._status("INACTIVE (owner not tracked)")


def main(args=None):
    rclpy.init(args=args)
    node = FollowManager()
    try:
        import threading     # v2.3: tf2 listener thread - ignore the Ctrl-C shutdown race
        _hook = threading.excepthook
        threading.excepthook = lambda a, _h=_hook: None if a.exc_type.__name__ == "ExternalShutdownException" else _h(a)
        import gc               # a2r_merge: init objects out of the GC (no full-pass stalls)
        gc.collect()
        gc.freeze()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:        # v2.3: Ctrl-C race - context closed while a message was taken
        if rclpy.ok():
            raise            # a real error while running is still reported
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
