#!/usr/bin/env python3
"""
owner_fusion_node.py — camera <-> lidar owner fusion, on the Pi (the lidar PC)
═══════════════════════════════════════════════════════════════════════════════
Port of the Jetson package's lidar_owner_fusion.py (followme_ros), which says
it runs "on the PC with the lidar". The clustering and the association /
tracking logic (cluster_points, OwnerLidarTracker) are the original, unchanged.
What is added for this robot:

  * SELF FILTER. The lidar sits at the rear of the upper deck and sees parts
    of the robot itself. Points inside the body outline are dropped, so the
    robot's own standoffs can never become "the owner".
  * EGO-MOTION. The original tracks in the laser frame and ignores the
    robot's motion. Here the track lives in the odometry frame
    (/odometry/filtered from the EKF), so the robot driving and turning does
    not move the owner's track. Falls back to base_link if odom is missing.
  * CAMERA FALLBACK. When the camera sees the owner but no lidar cluster
    matches yet (e.g. first frames, or the owner is further than the lidar
    clusters well), the camera point itself is published, marked "camera".
  * BOTH TOPIC NAMES. /followme/owner_position (what FollowMe publishes) and
    /owner_position are both accepted.
  * TF FALLBACK. If camera_color_optical_frame is not in TF (RealSense TF not
    reaching the Pi), the URDF's camera_optical_frame is used instead - same
    optical convention, same mount.

Publishes
  /owner_fused_position   PointStamped, base_link  - FollowMe (Jetson) uses this
                                                     to keep following when its
                                                     camera loses the owner
  /follow/owner           PointStamped, base_link  - for follow_manager_node
  /follow/owner_status    String (JSON)            - source, distance, bearing
  /owner_fused_markers    MarkerArray              - RViz: green = camera+lidar,
                                                     orange = lidar only,
                                                     blue = camera only
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────── core (no ROS)
# cluster_points and OwnerLidarTracker are the originals from the Jetson package.


def scan_to_points(ranges: Sequence[float], angle_min: float, angle_increment: float,
                   range_min: float, range_max: float) -> np.ndarray:
    """LaserScan -> (N, 2) points in the laser frame, ordered by angle (invalid dropped)."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= range_max)
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1)


@dataclass
class Cluster:
    centroid: np.ndarray        # (2,)
    width: float                # extent of the points (m)
    n: int


def cluster_points(pts: np.ndarray, gap: float = 0.10, merge: float = 0.45,
                   min_pts: int = 3, min_width: float = 0.04, max_width: float = 1.0,
                   leg_max: float = 0.25) -> list:
    """Split the angle-ordered scan where neighbours are > gap apart (360 deg wrap
    joined), merge LEG-SIZED segments (<= leg_max) into the first leg-only group
    whose centre is < merge away, keep person-sized groups. Same rules as the
    original, but segment stats come from one reduceat pass and group sums /
    extents are kept incrementally (the original re-stacked arrays inside an
    O(segments x groups) Python loop - hundreds of ms on the Pi)."""
    n = len(pts)
    if n < 2:
        return []
    d = np.hypot(*np.diff(pts, axis=0).T)
    cuts = np.flatnonzero(d > gap) + 1
    starts = np.r_[0, cuts]
    lens = np.diff(np.r_[starts, n])
    mins = np.minimum.reduceat(pts, starts, axis=0)
    maxs = np.maximum.reduceat(pts, starts, axis=0)
    sums = np.add.reduceat(pts, starts, axis=0)
    ok = lens >= 2
    starts, lens, mins, maxs, sums = starts[ok], lens[ok], mins[ok], maxs[ok], sums[ok]
    if len(starts) > 1 and np.hypot(*(pts[starts[0]] - pts[starts[-1] + lens[-1] - 1])) <= gap:
        # 360 deg wrap: the last segment continues into the first (joined one stays first)
        mins[0] = np.minimum(mins[0], mins[-1]); maxs[0] = np.maximum(maxs[0], maxs[-1])
        sums[0] += sums[-1]; lens[0] += lens[-1]
        starts, lens, mins, maxs, sums = starts[:-1], lens[:-1], mins[:-1], maxs[:-1], sums[:-1]
    width = np.hypot(*(maxs - mins).T)
    cent = sums / lens[:, None]
    leg = width <= leg_max
    out = []                                   # (first index, Cluster), sorted at the end
    for i in np.flatnonzero(~leg & (lens >= min_pts) & (width >= min_width) & (width <= max_width)):
        out.append((int(starts[i]), Cluster(cent[i], float(width[i]), int(lens[i]))))
    li = np.flatnonzero(leg)
    Gs, Gn = np.zeros((len(li), 2)), np.zeros(len(li))
    Gmin, Gmax = np.zeros((len(li), 2)), np.zeros((len(li), 2))
    Gfirst, G = np.zeros(len(li), dtype=int), 0
    for i in li:
        if G:
            gc = Gs[:G] / Gn[:G, None]
            hit = np.flatnonzero(np.hypot(gc[:, 0] - cent[i, 0], gc[:, 1] - cent[i, 1]) < merge)
            if len(hit):
                k = hit[0]
                Gs[k] += sums[i]; Gn[k] += lens[i]
                Gmin[k] = np.minimum(Gmin[k], mins[i]); Gmax[k] = np.maximum(Gmax[k], maxs[i])
                continue
        Gs[G], Gn[G], Gmin[G], Gmax[G], Gfirst[G] = sums[i], lens[i], mins[i], maxs[i], starts[i]
        G += 1
    for k in range(G):
        w = float(np.hypot(*(Gmax[k] - Gmin[k])))
        if Gn[k] >= min_pts and min_width <= w <= max_width:
            out.append((int(Gfirst[k]), Cluster(Gs[k] / Gn[k], w, int(Gn[k]))))
    out.sort(key=lambda t: t[0])
    return [c for _, c in out]


class OwnerLidarTracker:
    """Associates the camera's owner point with a lidar cluster and keeps tracking it.
    (Original logic; positions are whatever frame the caller uses consistently.)
    One addition: if the camera sees the owner clearly somewhere else (further
    than reinit_gate from the lidar track) and no cluster matches the camera,
    the lidar track is on the wrong object (a table, a stranger) - drop it and
    re-acquire, instead of following the wrong object lidar-only for 30 s."""

    def __init__(self, assoc_gate: float = 0.6, track_gate: float = 0.4, max_speed: float = 1.8,
                 camera_fresh_s: float = 0.4, lidar_only_max_s: float = 30.0, lost_s: float = 1.0,
                 reinit_gate: float = 1.0) -> None:
        self.assoc_gate, self.track_gate, self.max_speed = assoc_gate, track_gate, max_speed
        self.reinit_gate = reinit_gate
        self.camera_fresh_s, self.lidar_only_max_s, self.lost_s = camera_fresh_s, lidar_only_max_s, lost_s
        self.pos: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.t_seen = -math.inf
        self.t_camera = -math.inf
        self.source = "none"

    def update(self, clusters: list, now: float,
               camera_xy: Optional[np.ndarray] = None, camera_t: float = -math.inf,
               lidar_blocked: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        cents = np.array([c.centroid for c in clusters]) if clusters else np.zeros((0, 2))
        cam_fresh = camera_xy is not None and now - camera_t <= self.camera_fresh_s
        chosen = None
        if cam_fresh and len(cents):
            d = np.linalg.norm(cents - camera_xy, axis=1)
            k = int(np.argmin(d))
            if d[k] <= self.assoc_gate:
                chosen, self.t_camera, self.source = cents[k], now, "camera+lidar"
        if chosen is None and cam_fresh and self.pos is not None \
                and np.linalg.norm(self.pos - camera_xy) > self.reinit_gate:
            self.pos, self.vel, self.source = None, np.zeros(2), "none"   # wrong object: re-acquire
            return None
        if chosen is None and self.pos is not None and len(cents) \
                and now - self.t_camera <= self.lidar_only_max_s:
            dt = max(0.0, now - self.t_seen)
            pred = self.pos + self.vel * min(dt, 0.5)
            gate = self.track_gate + self.max_speed * min(dt, 0.5)
            d = np.linalg.norm(cents - pred, axis=1)
            if lidar_blocked is not None:
                d = np.where(lidar_blocked, np.inf, d)    # furniture: camera must confirm
            k = int(np.argmin(d))
            if d[k] <= gate:
                chosen, self.source = cents[k], "lidar"
        if chosen is None and cam_fresh and self.pos is None:
            return None
        if chosen is not None:
            if self.pos is not None and now > self.t_seen:
                v = (chosen - self.pos) / (now - self.t_seen)
                self.vel = 0.6 * self.vel + 0.4 * v
            self.pos, self.t_seen = chosen.copy(), now
            return self.pos.copy()
        if self.pos is not None and now - self.t_seen > self.lost_s:
            self.pos, self.vel, self.source = None, np.zeros(2), "none"
        return None


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Pose2D:
    """Planar pose (x, y, yaw); maps points between base_link and the tracking frame."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw

    def to_world(self, p: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.atleast_2d(p)
        return np.stack([self.x + c * p[:, 0] - s * p[:, 1], self.y + s * p[:, 0] + c * p[:, 1]], axis=1)

    def to_base(self, p: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.atleast_2d(p)
        dx, dy = p[:, 0] - self.x, p[:, 1] - self.y
        return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


class StaticMap:
    """Remembers lidar clusters that stay put in the odometry frame.

    A cluster that has sat still for static_s while the owner was somewhere
    else is furniture (a chair, a table leg, a bin). Lidar-only tracking may
    not jump onto it - only a camera confirmation can. A spot where the OWNER
    stood still is flagged as the owner's, so an owner standing still out of
    the camera's view is still tracked.

    Vectorised with numpy and capped at max_lms landmarks: the first version
    looped in plain Python over every landmark for every cluster, got slower
    as landmarks piled up, and on the Pi made each scan take longer than the
    next one took to arrive - the owner position then lagged by seconds."""

    def __init__(self, match=0.25, static_s=2.0, block_r=0.30, owner_r=0.12,
                 forget_s=30.0, max_lms=300):
        self.match, self.static_s, self.block_r = match, static_s, block_r
        self.owner_r, self.forget_s, self.max_lms = owner_r, forget_s, max_lms
        self.pos = np.zeros((0, 2))
        self.t0 = np.zeros(0)
        self.t1 = np.zeros(0)
        self.own = np.zeros(0, dtype=bool)

    @property
    def lms(self):          # compatibility / debugging view
        return [[self.pos[i], self.t0[i], self.t1[i], bool(self.own[i])] for i in range(len(self.pos))]

    def update(self, cents: np.ndarray, now: float, owner_pos: Optional[np.ndarray]):
        cents = np.atleast_2d(np.asarray(cents, dtype=float)) if len(cents) else np.zeros((0, 2))
        new_p, new_own = [], []
        for c in cents:
            near_owner = owner_pos is not None and float(np.hypot(*(c - owner_pos))) < self.owner_r
            if len(self.pos):
                d = np.hypot(self.pos[:, 0] - c[0], self.pos[:, 1] - c[1])
                k = int(np.argmin(d))
                if d[k] < self.match:
                    self.pos[k] = 0.8 * self.pos[k] + 0.2 * c
                    self.t1[k] = now
                    self.own[k] = self.own[k] or near_owner
                    continue
            new_p.append(c); new_own.append(near_owner)
        if new_p:
            n = len(new_p)
            self.pos = np.vstack([self.pos, np.array(new_p)])
            self.t0 = np.concatenate([self.t0, np.full(n, now)])
            self.t1 = np.concatenate([self.t1, np.full(n, now)])
            self.own = np.concatenate([self.own, np.array(new_own, dtype=bool)])
        keep = (now - self.t1) < self.forget_s
        if keep.sum() > self.max_lms:                     # keep the most recently seen
            idx = np.flatnonzero(keep)
            idx = idx[np.argsort(self.t1[idx])[-self.max_lms:]]
            keep = np.zeros_like(keep); keep[idx] = True
        if not keep.all():
            self.pos, self.t0, self.t1, self.own = self.pos[keep], self.t0[keep], self.t1[keep], self.own[keep]

    def blocked(self, cents: np.ndarray) -> np.ndarray:
        cents = np.atleast_2d(np.asarray(cents, dtype=float)) if len(cents) else np.zeros((0, 2))
        out = np.zeros(len(cents), dtype=bool)
        m = (~self.own) & ((self.t1 - self.t0) >= self.static_s)
        if not len(cents) or not m.any():
            return out
        L = self.pos[m]
        d = np.hypot(cents[:, None, 0] - L[None, :, 0], cents[:, None, 1] - L[None, :, 1])
        return (d < self.block_r).any(axis=1)


class FusionCore:
    """Everything except ROS I/O, so it can be tested without ROS.

    Scan points arrive in the LASER frame; lidar_pose is the laser in base_link;
    odom is the robot pose in the tracking frame (identity if no odometry).
    The tracker runs in the tracking frame; outputs are in base_link."""

    def __init__(self, tracker: OwnerLidarTracker, body_front: float, body_rear: float,
                 body_half_w: float, lidar_pose: Pose2D, camera_fallback: bool = True,
                 seed_window_s: float = 2.0, still_drop_s: float = 8.0, still_speed: float = 0.12):
        self.tracker = tracker
        self.still_drop_s, self.still_speed = still_drop_s, still_speed
        self._still_since = None
        self.seed_window = seed_window_s
        self._cam_last = None          # (world xy, time) of the last camera sighting
        self._cam_vel = np.zeros(2)
        self.static = StaticMap()
        self.bf, self.br, self.bw = body_front, body_rear, body_half_w
        self.lidar = lidar_pose
        self.camera_fallback = camera_fallback
        self.source = "none"

    def step(self, laser_pts: np.ndarray, odom: Pose2D, now: float,
             cam_base: Optional[np.ndarray], cam_t: float):
        """Returns (owner_in_base_link | None, clusters_in_base, source)."""
        if len(laser_pts):
            base = self.lidar.to_world(laser_pts)                 # laser -> base_link
            body = (base[:, 0] < self.bf) & (base[:, 0] > self.br) & (np.abs(base[:, 1]) < self.bw)
            laser_pts = laser_pts[~body]                          # keep angle order
        clusters = cluster_points(laser_pts)
        # cluster centroids: laser -> base_link -> tracking frame
        cl_base = [Cluster(self.lidar.to_world(c.centroid)[0], c.width, c.n) for c in clusters]
        cl_world = [Cluster(odom.to_world(c.centroid)[0], c.width, c.n) for c in cl_base]
        cam_world = odom.to_world(cam_base)[0] if cam_base is not None else None
        cam_fresh = cam_world is not None and now - cam_t <= self.tracker.camera_fresh_s
        if cam_fresh:
            if self._cam_last is not None and cam_t > self._cam_last[1]:
                v = (cam_world - self._cam_last[0]) / (cam_t - self._cam_last[1])
                if np.linalg.norm(v) < 3.0:
                    self._cam_vel = 0.7 * self._cam_vel + 0.3 * v
            self._cam_last = (cam_world.copy(), cam_t)
        # SEED: the camera just lost the owner and the lidar never locked on
        # (e.g. a table hid the owner's legs from the lidar while the camera
        # saw over it). Look for a cluster where the camera last saw them.
        elif (self.tracker.pos is None and self._cam_last is not None
              and now - self._cam_last[1] <= self.seed_window and cl_world):
            dt = now - self._cam_last[1]
            pred = self._cam_last[0] + self._cam_vel * min(dt, 1.0)
            cents = np.array([c.centroid for c in cl_world])
            d = np.linalg.norm(cents - pred, axis=1)
            k = int(np.argmin(d))
            if d[k] <= self.tracker.assoc_gate + self.tracker.max_speed * min(dt, 0.5):
                self.tracker.pos, self.tracker.vel = cents[k].copy(), self._cam_vel.copy()
                self.tracker.t_seen, self.tracker.t_camera = now, self._cam_last[1]
        cents_w = np.array([c.centroid for c in cl_world]) if cl_world else np.zeros((0, 2))
        prev_owner = self.tracker.pos.copy() if self.tracker.pos is not None else None
        blocked = self.static.blocked(cents_w) if len(cents_w) else None
        fused = self.tracker.update(cl_world, now, cam_world, cam_t, blocked)
        self.static.update(cents_w, now, fused if fused is not None else prev_owner)
        # BACKSTOP: a lidar-only track that has not moved for still_drop_s is
        # probably furniture the track slid onto (e.g. while the owner was
        # hidden behind it). Drop it: follow_manager goes inactive, FollowMe
        # searches toward the last-seen side and the camera re-acquires. A
        # camera confirmation resets the timer, so a visible standing owner
        # is never dropped.
        if fused is not None and self.tracker.source == "lidar" and self.still_drop_s > 0:
            if float(np.linalg.norm(self.tracker.vel)) < self.still_speed:
                if self._still_since is None:
                    self._still_since = now
                elif now - self._still_since > self.still_drop_s:
                    self.tracker.pos, self.tracker.vel = None, np.zeros(2)
                    self.tracker.source, self._still_since = "none", None
                    fused = None
            else:
                self._still_since = None
        else:
            self._still_since = None
        if fused is not None:
            self.source = self.tracker.source
            return odom.to_base(fused)[0], cl_base, self.source
        cam_fresh = cam_base is not None and now - cam_t <= self.tracker.camera_fresh_s
        if self.camera_fallback and cam_fresh:
            self.source = "camera"
            return np.asarray(cam_base, dtype=float), cl_base, self.source
        self.source = "none"
        return None, cl_base, self.source


# ─────────────────────────────────────────────────────────────── ROS node


def main() -> None:  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import Point, PointStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    class OwnerFusionNode(Node):
        def __init__(self) -> None:
            super().__init__("owner_fusion")
            p = self.declare_parameter
            p("scan_topic", "/scan")
            p("owner_topics", "/followme/owner_position,/owner_position")
            p("camera_frame_fallback", "camera_optical_frame")
            p("odom_topic", "/odometry/filtered")
            p("base_frame", "base_link")
            p("fused_topic", "/owner_fused_position")
            p("owner_out_topic", "/follow/owner")
            p("assoc_gate_m", 0.6)
            p("track_gate_m", 0.4)
            p("camera_fresh_s", 0.4)
            p("lidar_only_max_s", 30.0)       # FollowMe's value: keep following on lidar this long
            p("lost_s", 1.0)
            p("reinit_gate_m", 1.0)           # camera disagrees by more -> lidar track dropped
            p("still_drop_s", 8.0)            # lidar-only track motionless this long -> dropped (0 = off)
            p("camera_fallback", True)
            p("body_front_m", 0.215)          # measured: plate I front / sonar face
            p("body_rear_m", -0.210)
            p("body_half_width_m", 0.155)
            p("lidar_offset_x", -0.140)       # used only if TF base_link <- laser is missing (= URDF lidar_x)
            p("lidar_offset_y", 0.001)
            p("lidar_yaw_offset", 0.0)
            p("max_range_m", 6.0)             # ignore lidar points further than this
            p("stop_tf_listener", True)       # stop parsing /tf once the static TFs are cached
            p("owner_source", "pi")           # a2r_pi_owner_only: pi = track here | jetson = the keeper tracks
            p("keeper_topic", "/owner_fused_position")      # lidar_owner_keeper output (Jetson)
            p("keeper_status_topic", "/owner_keeper/status")  # its heartbeat (2 Hz)
            p("keeper_fresh_s", 0.5)          # keeper point older than this = owner not tracked now
            p("keeper_alive_s", 3.0)          # no heartbeat this long = no keeper: camera only
            g = lambda n: self.get_parameter(n).value  # noqa: E731

            self.base = str(g("base_frame"))
            self.max_range = float(g("max_range_m"))
            self.stop_tf = bool(g("stop_tf_listener"))
            self.cam_fallback_frame = str(g("camera_frame_fallback"))
            self._param_lidar = Pose2D(float(g("lidar_offset_x")), float(g("lidar_offset_y")),
                                       float(g("lidar_yaw_offset")))
            tracker = OwnerLidarTracker(assoc_gate=float(g("assoc_gate_m")),
                                        track_gate=float(g("track_gate_m")),
                                        camera_fresh_s=float(g("camera_fresh_s")),
                                        lidar_only_max_s=float(g("lidar_only_max_s")),
                                        lost_s=float(g("lost_s")),
                                        reinit_gate=float(g("reinit_gate_m")))
            self.core = FusionCore(tracker, float(g("body_front_m")), float(g("body_rear_m")),
                                   float(g("body_half_width_m")), self._param_lidar,
                                   bool(g("camera_fallback")), still_drop_s=float(g("still_drop_s")))
            self._lidar_from_tf = False
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=False)
            self.owner_msg, self.owner_t = None, -math.inf
            self.odom: Optional[Pose2D] = None
            self.odom_t = -math.inf
            self._warned = set()
            self._cam_tf = {}

            self.pub_fused = (None if str(g("owner_source")).strip().lower() == "jetson"  # a2r_pi_owner_only:
                              else self.create_publisher(PointStamped, g("fused_topic"), 10))  # keeper owns it
            self.pub_owner = self.create_publisher(PointStamped, g("owner_out_topic"), 10)
            # a2r_pi_update2: owner point ONLY for scan_leg_fill to cut the owner out of the
            # planner scan - keeper point or, failing that, the camera point. Never a follow target.
            self.declare_parameter("scrub_topic", "/follow/owner_scrub")
            self.declare_parameter("diag_period_s", 15.0)
            # a2r_pi_update3: keeper alive but sending no owner point -> use the camera owner point
            self.declare_parameter("keeper_silent_camera", True)
            self.keeper_silent_cam = bool(self.get_parameter("keeper_silent_camera").value)
            self.pub_scrub = self.create_publisher(PointStamped, str(self.get_parameter("scrub_topic").value), 10)
            self._diag = {"keeper": 0, "kept": 0, "ignored": 0, "frames": set(), "cam": 0, "scrub": 0, "hb": 0}
            self.pub_status = self.create_publisher(String, "/follow/owner_status", 10)
            self.pub_mk = self.create_publisher(MarkerArray, "/owner_fused_markers", 1)
            # depth 1 everywhere: if a scan is ever slow, OLD messages are dropped
            # instead of queued - a queue of 10 owner points at a few Hz was a
            # 2-3 s lag between where the owner was and where the robot steered.
            latest = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.RELIABLE)
            latest_be = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                   reliability=ReliabilityPolicy.BEST_EFFORT)
            for t in [x.strip() for x in str(g("owner_topics")).split(",") if x.strip()]:
                self.create_subscription(PointStamped, t, self._on_owner, latest_be)
            # a2r_pi_owner_only: the Jetson's lidar_owner_keeper is the one owner tracker
            self.jetson_mode = str(g("owner_source")).strip().lower() == "jetson"
            if self.jetson_mode:
                # a2r_merge: the keeper tracks; this node only relays. It used to subscribe to the
                # raw ~2300-beam /scan (and 50 Hz odometry) just to get a tick - pure CPU. A timer
                # gives the same 7-10 Hz tick without deserialising a single scan.
                self.create_timer(0.1, lambda: self._jetson_scan(time.monotonic()))
            else:
                self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 20)
                self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, latest_be)
            self.keeper_pos, self.keeper_t, self.keeper_alive_t = None, -math.inf, -math.inf
            self.keeper_fresh_s, self.keeper_alive_s = float(g("keeper_fresh_s")), float(g("keeper_alive_s"))
            if self.jetson_mode:
                self.create_subscription(PointStamped, g("keeper_topic"), self._on_keeper, latest_be)
                self.create_subscription(String, g("keeper_status_topic"), self._on_keeper_status, 10)
                self.get_logger().info(f"owner source: Jetson lidar_owner_keeper ({g('keeper_topic')}) - "
                                       "the Pi's own lidar tracker is off; camera only without a keeper")
                self.create_timer(max(2.0, float(self.get_parameter("diag_period_s").value)), self._diag_report)
                # a2r_keeper_link: the camera fallback (keeper silent) must agree with the phone
                self.declare_parameter("phone_gate", True)
                self.declare_parameter("phone_gate_m", 1.0)
                self._a2r_ph = self._a2r_ph_st = None
                self._a2r_veto = 0
                self.create_subscription(PointStamped, "/phone_owner/view",
                                         lambda m: setattr(self, "_a2r_ph", (m, time.monotonic())), 5)
                self.create_subscription(String, "/phone_owner/status", self._a2r_on_ph_st, 5)
            self.Point, self.PointStamped, self.String = Point, PointStamped, String
            self.Marker, self.MarkerArray = Marker, MarkerArray
            self.TransformException = TransformException
            self.get_logger().info("owner_fusion up (camera <-> lidar owner tracking)")

        # ── inputs ──────────────────────────────────────────────────────
        # ── a2r_pi_owner_only: owner from the Jetson keeper ─────────────
        def _on_keeper_status(self, msg):
            self.keeper_alive_t = time.monotonic()
            self._diag["hb"] += 1                       # a2r_pi_update2

        def _diag_report(self):
            """a2r_pi_update2: proves on the Pi whether keeper points arrive and are used."""
            d, now = self._diag, time.monotonic()
            hb = "never" if self.keeper_alive_t == -math.inf else f"{now - self.keeper_alive_t:.1f} s ago"
            self.get_logger().info(
                f"keeper: {d['hb']} heartbeats (last {hb}), {d['keeper']} points "
                f"(frames {sorted(d['frames']) or '-'}), {d['kept']} used, {d['ignored']} ignored (no TF); "
                f"camera: {d['cam']} points; scrub points: {d['scrub']}")
            for k in ("keeper", "kept", "ignored", "cam", "scrub", "hb"):
                d[k] = 0

        def _publish_scrub(self, p):
            out = self.PointStamped()
            out.header.stamp, out.header.frame_id = self.get_clock().now().to_msg(), self.base
            out.point = self.Point(x=float(p[0]), y=float(p[1]), z=0.0)
            self.pub_scrub.publish(out)
            self._diag["scrub"] += 1

        def _to_base(self, msg):
            """PointStamped in a static frame (lidar mount) -> base_link xy; transform cached."""
            src = msg.header.frame_id or self.base
            if src == self.base:
                return np.array([msg.point.x, msg.point.y])
            tf = self._cam_tf.get(src)
            if tf is None:
                tf = self._lookup(self.base, src)
                if tf is None:
                    self._warn_once("tf:" + src, f"no TF {self.base} <- {src}: keeper point ignored")
                    return None
                self._cam_tf[src] = tf
            r, t = tf.transform.rotation, tf.transform.translation
            R = quat_to_matrix(r.x, r.y, r.z, r.w)
            return (R @ np.array([msg.point.x, msg.point.y, msg.point.z]) + np.array([t.x, t.y, t.z]))[:2]

        def _publish_owner(self, p, stamp=None):
            out = self.PointStamped()
            # a2r_v121: keep the MEASUREMENT time (keeper scan stamp) instead of 'now'
            st = stamp if (stamp is not None and (stamp.sec or stamp.nanosec)) else self.get_clock().now().to_msg()
            out.header.stamp, out.header.frame_id = st, self.base
            out.point = self.Point(x=float(p[0]), y=float(p[1]), z=0.0)
            self.pub_owner.publish(out)

        def _on_keeper(self, msg):
            now = time.monotonic()
            self.keeper_alive_t = now
            self._diag["keeper"] += 1                    # a2r_pi_update2
            self._diag["frames"].add(msg.header.frame_id or "(empty)")
            p = self._to_base(msg)
            if p is None:
                self._diag["ignored"] += 1
                return
            # a2r_v4 hijack guard: never follow a keeper point the camera or physics disagree with
            if not hasattr(self, "_kg"):
                for n, d in (("keeper_cam_gate_m", 0.7), ("keeper_jump_gate_m", 0.6),
                             ("keeper_jump_speed_mps", 1.8)):
                    if not self.has_parameter(n):
                        self.declare_parameter(n, d)
                self._kg = {n: float(self.get_parameter(n).value) for n in
                            ("keeper_cam_gate_m", "keeper_jump_gate_m", "keeper_jump_speed_mps")}
                self._kg_last, self._kg_last_t, self._kg_rej = None, -math.inf, 0
            cam = None
            if self.owner_msg is not None and now - self.owner_t <= self.core.tracker.camera_fresh_s:
                cam = self._cam_in_base()
            if cam is not None:
                if math.hypot(p[0] - cam[0], p[1] - cam[1]) > self._kg["keeper_cam_gate_m"]:
                    self._diag["ignored"] += 1; self._kg_rej += 1
                    self._warn_once("kgcam", "keeper point disagrees with the camera - ignored "
                                    "(likely a chair/box taken for legs); camera owner used")
                    return
            elif self._kg_last is not None and now - self._kg_last_t < 1.0:
                dt = now - self._kg_last_t
                lim = self._kg["keeper_jump_gate_m"] + self._kg["keeper_jump_speed_mps"] * dt
                if math.hypot(p[0] - self._kg_last[0], p[1] - self._kg_last[1]) > lim:
                    self._diag["ignored"] += 1; self._kg_rej += 1
                    return
            else:
                # a2r_v121: after > 1 s without a keeper point and no camera, ONE point used to be
                # accepted unchecked (chair legs / another person). Now two consistent points first.
                c = getattr(self, "_kg_cand", None)
                self._kg_cand = (p, now)
                if c is None or now - c[1] > 0.6 or math.hypot(p[0] - c[0][0], p[1] - c[0][1]) > 0.35:
                    self._diag["ignored"] += 1
                    return
            self._kg_cand = None
            self._kg_last, self._kg_last_t = p, now
            self._diag["kept"] += 1
            self.keeper_pos, self.keeper_t = p, now
            self._publish_owner(p, msg.header.stamp)     # a2r_v121: measurement stamp kept
            self._publish_scrub(p)

        def _jetson_scan(self, now):
            """Per scan: status, and the camera point ONLY if no keeper is running. A running keeper
            that is silent is abstaining (owner not visible, or someone too close): no owner then."""
            source, owner = "none", None
            alive = now - self.keeper_alive_t <= self.keeper_alive_s
            if self.keeper_pos is not None and now - self.keeper_t <= self.keeper_fresh_s:
                source, owner = "jetson_keeper", self.keeper_pos
            elif self.owner_msg is not None and now - self.owner_t <= self.core.tracker.camera_fresh_s:
                # a2r_pi_update2: keeper silent - still cut the camera's owner out of the PLANNER scan
                # (the collision monitor keeps them: /scan/merged), so their legs never wall off the way
                cam = self._cam_in_base()
                if cam is not None:
                    self._publish_scrub(cam)
            silent_cam = alive and getattr(self, "keeper_silent_cam", False)   # a2r_pi_update3
            if source == "none" and (not alive or silent_cam):
                if not alive:
                    self._warn_once("nokeeper", "no lidar_owner_keeper heartbeat: following the camera only")
                if self.owner_msg is not None and now - self.owner_t <= self.core.tracker.camera_fresh_s:
                    cam = self._cam_in_base()
                    if cam is not None and hasattr(self, "_a2r_veto") and not self._a2r_phone_ok(cam):
                        cam = None                                # a2r_keeper_link
                    if cam is not None:
                        source, owner = ("camera_keeper_silent" if alive else "camera"), np.asarray(cam, dtype=float)
                        self._publish_owner(owner)
            st = {"source": source, "tracked": owner is not None, "keeper": alive}
            st["phone_veto"] = getattr(self, "_a2r_veto", 0)   # a2r_keeper_link
            if owner is not None:
                st["dist_m"] = round(float(math.hypot(owner[0], owner[1])), 2)
                st["bearing_deg"] = round(math.degrees(math.atan2(owner[1], owner[0])), 1)
            st["camera_age_s"] = round(now - self.owner_t, 2) if self.owner_t > -math.inf else None
            self.pub_status.publish(self.String(data=json.dumps(st)))

        # ── a2r_keeper_link: the camera fallback must agree with the phone ──
        def _a2r_on_ph_st(self, msg):
            try:
                self._a2r_ph_st = (json.loads(msg.data), time.monotonic())
            except ValueError:
                pass

        def _a2r_phone_ok(self, cam) -> bool:
            """Camera owner point (used while the keeper is silent) vs the phone estimate.
            False = more than max(phone_gate_m, 3 sigma) apart: probably a person crossing the
            owner's path taken for the owner - not followed. True when there is no usable phone."""
            if not bool(self.get_parameter("phone_gate").value) or self._a2r_ph is None or self._a2r_ph_st is None:
                return True
            (m, t), (st, ts), now = self._a2r_ph, self._a2r_ph_st, time.monotonic()
            sig = float(m.point.z)                       # phone_pdr_node puts its sigma in z
            if now - t > 0.6 or now - ts > 2.0 or not st.get("aligned") or not 0.0 <= sig <= 1.2:
                return True
            tf = self._lookup(self.base, m.header.frame_id or "odom")
            if tf is None:
                return True
            r, tt = tf.transform.rotation, tf.transform.translation
            ph = (quat_to_matrix(r.x, r.y, r.z, r.w) @ np.array([m.point.x, m.point.y, 0.0])
                  + np.array([tt.x, tt.y, tt.z]))[:2]
            gate = max(float(self.get_parameter("phone_gate_m").value), 3.0 * sig)
            if float(np.hypot(*(np.asarray(cam, dtype=float)[:2] - ph))) <= gate:
                return True
            self._a2r_veto += 1
            self._warn_once("a2r_phveto", "camera owner point disagrees with the phone while the keeper is "
                                          "silent - not followed (count: /follow/owner_status phone_veto)")
            return False

        def _on_owner(self, msg):
            self.owner_msg, self.owner_t = msg, time.monotonic()   # receive time: no clock sync needed
            if hasattr(self, "_diag"):
                self._diag["cam"] += 1                   # a2r_pi_update2

        def _on_odom(self, msg):
            q = msg.pose.pose.orientation
            self.odom = Pose2D(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(q))
            self.odom_t = time.monotonic()

        def _warn_once(self, key, text):
            if key not in self._warned:
                self._warned.add(key)
                self.get_logger().warn(text)

        def _lookup(self, target, source):
            try:
                return self.tf.lookup_transform(target, source, rclpy.time.Time())
            except self.TransformException:
                return None

        def _cam_in_base(self) -> Optional[np.ndarray]:
            msg = self.owner_msg
            src = msg.header.frame_id or self.cam_fallback_frame
            tf = self._cam_tf.get(src)
            if tf is None:
                tf = self._lookup(self.base, src)
            if tf is None and src != self.cam_fallback_frame and src not in self._cam_tf:
                tf = self._lookup(self.base, self.cam_fallback_frame)
                if tf is not None:
                    self._warn_once("camfb", f"no TF {self.base} <- {src}; using {self.cam_fallback_frame} "
                                    "(URDF camera mount) instead")
            if tf is None:
                self._warn_once("camtf", f"no TF {self.base} <- {src}: camera point ignored")
                return None
            if src not in self._cam_tf and self._lookup(self.base, src) is not None:
                if "camfb" in self._warned:
                    self.get_logger().info(f"real camera TF {self.base} <- {src} found - fallback no longer used")
                self._cam_tf[src] = tf
                self._maybe_stop_tf()
            r, t = tf.transform.rotation, tf.transform.translation
            R = quat_to_matrix(r.x, r.y, r.z, r.w)
            p = R @ np.array([msg.point.x, msg.point.y, msg.point.z]) + np.array([t.x, t.y, t.z])
            return p[:2]

        def _maybe_stop_tf(self):
            if not (self.stop_tf and self._lidar_from_tf and self._cam_tf and self.tfl is not None):
                return
            try:
                self.tfl.unregister()
                self.get_logger().info("static TFs cached - TF listener stopped")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"could not stop TF listener: {e}")
            self.tfl = None

        def _refresh_lidar_pose(self, frame):
            if self._lidar_from_tf:
                return
            tf = self._lookup(self.base, frame)
            if tf is None:
                self._warn_once("lidtf", f"no TF {self.base} <- {frame}; using lidar_offset_* params")
                return
            q = tf.transform.rotation
            self.core.lidar = Pose2D(tf.transform.translation.x, tf.transform.translation.y, yaw_of(q))
            self._lidar_from_tf = True
            self._maybe_stop_tf()

        # ── main ────────────────────────────────────────────────────────
        def _on_scan(self, scan):
            t_start, c_start = time.monotonic(), time.process_time()
            self._on_scan_impl(scan)
            dt, cpu = time.monotonic() - t_start, time.process_time() - c_start
            if dt > 0.08:
                self.get_logger().warn(
                    f"owner_fusion: scan {dt*1000:.0f} ms wall / {cpu*1000:.0f} ms CPU - "
                    f"{len(self.core.static.pos)} landmarks",
                    throttle_duration_sec=5.0)

        def _on_scan_impl(self, scan):
            now = time.monotonic()
            if self.jetson_mode:                         # a2r_pi_owner_only: the keeper tracks
                self._jetson_scan(now)
                return
            self._refresh_lidar_pose(scan.header.frame_id)
            pts = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                 scan.range_min, min(scan.range_max, self.max_range))
            odom_ok = self.odom is not None and now - self.odom_t < 0.5
            if not odom_ok:
                self._warn_once("odom", "no fresh odometry: tracking in base_link (robot motion not compensated)")
            odom = self.odom if odom_ok else Pose2D()
            cam = None
            if self.owner_msg is not None and now - self.owner_t <= self.core.tracker.camera_fresh_s:
                cam = self._cam_in_base()
            owner, clusters, source = self.core.step(pts, odom, now, cam, self.owner_t)

            stamp = self.get_clock().now().to_msg()
            if owner is not None:
                for pub in (self.pub_fused, self.pub_owner):
                    out = self.PointStamped()
                    out.header.stamp, out.header.frame_id = stamp, self.base
                    out.point = self.Point(x=float(owner[0]), y=float(owner[1]), z=0.0)
                    pub.publish(out)
            st = {"source": source, "tracked": owner is not None}
            if owner is not None:
                st["dist_m"] = round(float(math.hypot(owner[0], owner[1])), 2)
                st["bearing_deg"] = round(math.degrees(math.atan2(owner[1], owner[0])), 1)
            st["camera_age_s"] = round(now - self.owner_t, 2) if self.owner_t > -math.inf else None
            self.pub_status.publish(self.String(data=json.dumps(st)))
            self._markers(stamp, clusters, owner, source)

        def _markers(self, stamp, clusters, owner, source):
            if self.pub_mk.get_subscription_count() == 0:
                return
            t_mk = time.monotonic()
            if t_mk - getattr(self, "_mk_t", 0.0) < 0.5:
                return
            self._mk_t = t_mk
            ma = self.MarkerArray()
            for i, c in enumerate(clusters):
                m = self.Marker()
                m.header.stamp, m.header.frame_id = stamp, self.base
                m.ns, m.id, m.type, m.action = "clusters", i, self.Marker.CYLINDER, self.Marker.ADD
                m.pose.position = self.Point(x=float(c.centroid[0]), y=float(c.centroid[1]), z=0.0)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = max(0.1, c.width)
                m.scale.z = 0.05
                m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.5, 0.6
                m.lifetime.nanosec = 300_000_000
                ma.markers.append(m)
            if owner is not None:
                m = self.Marker()
                m.header.stamp, m.header.frame_id = stamp, self.base
                m.ns, m.id, m.type, m.action = "owner", 0, self.Marker.CYLINDER, self.Marker.ADD
                m.pose.position = self.Point(x=float(owner[0]), y=float(owner[1]), z=0.85)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 0.5
                m.scale.z = 1.7
                col = {"camera+lidar": (0.0, 1.0, 0.0), "lidar": (1.0, 0.6, 0.0)}.get(source, (0.2, 0.5, 1.0))
                m.color.r, m.color.g, m.color.b, m.color.a = (*col, 0.5)
                m.lifetime.nanosec = 300_000_000
                ma.markers.append(m)
            self.pub_mk.publish(ma)

    rclpy.init()
    node = OwnerFusionNode()
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
