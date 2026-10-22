#!/usr/bin/env python3
"""Lidar <-> camera owner fusion (runs on the PC with the lidar; ROS 2 Humble).

The Jetson's FollowMe node publishes the owner's 3-D position from the camera
(``/followme/owner_position``, PointStamped in the camera optical frame, image
time stamp). This node:

1. turns each ``/scan`` into clusters (people's legs/bodies) in the laser frame,
2. transforms the camera owner point into the laser frame (TF) and picks the
   lidar cluster next to it -> "this lidar object is the owner",
3. keeps tracking that cluster with the lidar alone when the camera loses the
   owner (turning a corner, outside the 70 deg camera view: the lidar sees 360 deg),
4. publishes ``/owner_fused_position`` (PointStamped, laser frame) which FollowMe
   uses to keep following while the camera cannot see the owner, plus
   ``/owner_fused_markers`` for RViz.

TF needed: laser frame <-> camera optical frame, i.e. base_link -> laser (your URDF)
and base_link -> camera_link (static transform with the camera mount position);
camera_link -> camera_color_optical_frame comes from the RealSense driver.

Run:  python3 lidar_owner_fusion.py            (ROS_DOMAIN_ID must match the Jetson)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------- core (no ROS)


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


def cluster_points(pts: np.ndarray, gap: float = 0.15, merge: float = 0.45,
                   min_pts: int = 3, min_width: float = 0.04, max_width: float = 1.0) -> list[Cluster]:
    """Split the angle-ordered scan where neighbours are > ``gap`` apart, merge segments
    whose centres are < ``merge`` apart (two legs -> one person), keep person-sized ones."""
    if len(pts) == 0:
        return []
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cuts = np.flatnonzero(d > gap) + 1
    segs = [s for s in np.split(pts, cuts) if len(s) >= 2]
    # the scan wraps around: join the last and first segment if they touch
    if len(segs) > 1 and np.linalg.norm(segs[0][0] - segs[-1][-1]) <= gap:
        segs[0] = np.vstack([segs[-1], segs[0]])
        segs.pop()
    groups: list[np.ndarray] = []
    for s in segs:
        c = s.mean(axis=0)
        for i, g in enumerate(groups):
            if np.linalg.norm(g.mean(axis=0) - c) < merge:
                groups[i] = np.vstack([g, s])
                break
        else:
            groups.append(s)
    out = []
    for g in groups:
        width = float(np.linalg.norm(g.max(axis=0) - g.min(axis=0)))
        if len(g) >= min_pts and min_width <= width <= max_width:
            out.append(Cluster(g.mean(axis=0), width, len(g)))
    return out


class OwnerLidarTracker:
    """Associates the camera's owner point with a lidar cluster and keeps tracking it.

    * camera point fresh: take the cluster nearest to it (within ``assoc_gate``)
    * camera point stale: follow the cluster nearest to the predicted track
      (within ``track_gate`` + distance the owner can walk), for at most
      ``lidar_only_max_s`` without any camera confirmation
    * no matching cluster for ``lost_s``: track dropped
    Tracking is done in the laser frame; at FollowMe's low speeds the robot's own motion
    between scans is small compared to the gates.
    """

    def __init__(self, assoc_gate: float = 0.6, track_gate: float = 0.4, max_speed: float = 1.8,
                 camera_fresh_s: float = 0.4, lidar_only_max_s: float = 10.0, lost_s: float = 1.0) -> None:
        self.assoc_gate, self.track_gate, self.max_speed = assoc_gate, track_gate, max_speed
        self.camera_fresh_s, self.lidar_only_max_s, self.lost_s = camera_fresh_s, lidar_only_max_s, lost_s
        self.pos: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.t_seen = -math.inf         # last time a cluster was associated
        self.t_camera = -math.inf       # last camera confirmation
        self.source = "none"

    def update(self, clusters: list[Cluster], now: float,
               camera_xy: Optional[np.ndarray] = None, camera_t: float = -math.inf) -> Optional[np.ndarray]:
        cents = np.array([c.centroid for c in clusters]) if clusters else np.zeros((0, 2))
        cam_fresh = camera_xy is not None and now - camera_t <= self.camera_fresh_s
        chosen = None
        if cam_fresh and len(cents):
            d = np.linalg.norm(cents - camera_xy, axis=1)
            k = int(np.argmin(d))
            if d[k] <= self.assoc_gate:
                chosen, self.t_camera, self.source = cents[k], now, "camera+lidar"
        if chosen is None and self.pos is not None and len(cents) \
                and now - self.t_camera <= self.lidar_only_max_s:
            dt = max(0.0, now - self.t_seen)
            pred = self.pos + self.vel * min(dt, 0.5)
            gate = self.track_gate + self.max_speed * min(dt, 0.5)
            d = np.linalg.norm(cents - pred, axis=1)
            k = int(np.argmin(d))
            if d[k] <= gate:
                chosen, self.source = cents[k], "lidar"
        if chosen is None and cam_fresh and self.pos is None:
            return None                                  # camera sees the owner, lidar does not (yet)
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


# ---------------------------------------------------------------- ROS node


def main() -> None:  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import Point, PointStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    class FusionNode(Node):
        def __init__(self) -> None:
            super().__init__("lidar_owner_fusion")
            p = self.declare_parameter
            p("scan_topic", "/scan")
            p("owner_topic", "/followme/owner_position")
            p("fused_topic", "/owner_fused_position")
            p("assoc_gate_m", 0.6)
            p("track_gate_m", 0.4)
            p("lidar_only_max_s", 10.0)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.tracker = OwnerLidarTracker(assoc_gate=g("assoc_gate_m"), track_gate=g("track_gate_m"),
                                             lidar_only_max_s=g("lidar_only_max_s"))
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=True)
            self.owner_msg = None
            self.owner_t = -math.inf
            self.pub = self.create_publisher(PointStamped, g("fused_topic"), 10)
            self.mk_pub = self.create_publisher(MarkerArray, "/owner_fused_markers", 1)
            self.create_subscription(PointStamped, g("owner_topic"), self._on_owner, 10)
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data)
            self._warned = False
            self.get_logger().info("lidar_owner_fusion up")

        def _on_owner(self, msg):
            self.owner_msg, self.owner_t = msg, time.monotonic()

        def _to_frame(self, msg, frame: str) -> Optional[np.ndarray]:
            try:
                tf = self.tf.lookup_transform(frame, msg.header.frame_id, rclpy.time.Time())
            except TransformException as exc:
                if not self._warned:
                    self.get_logger().warn(f"no TF {frame} <- {msg.header.frame_id}: {exc}")
                    self._warned = True
                return None
            r, t = tf.transform.rotation, tf.transform.translation
            R = quat_to_matrix(r.x, r.y, r.z, r.w)
            return R @ np.array([msg.point.x, msg.point.y, msg.point.z]) + np.array([t.x, t.y, t.z])

        def _on_scan(self, scan: LaserScan):
            now = time.monotonic()
            pts = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                 scan.range_min, scan.range_max)
            clusters = cluster_points(pts)
            cam_xy = None
            if self.owner_msg is not None and now - self.owner_t <= self.tracker.camera_fresh_s:
                p = self._to_frame(self.owner_msg, scan.header.frame_id)
                cam_xy = p[:2] if p is not None else None
            fused = self.tracker.update(clusters, now, cam_xy, self.owner_t)
            ma = MarkerArray()
            for i, c in enumerate(clusters):
                m = Marker()
                m.header = scan.header
                m.ns, m.id, m.type, m.action = "clusters", i, Marker.CYLINDER, Marker.ADD
                m.pose.position = Point(x=float(c.centroid[0]), y=float(c.centroid[1]), z=0.0)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = max(0.1, c.width)
                m.scale.z = 0.05
                m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.5, 0.6
                m.lifetime.nanosec = 300_000_000
                ma.markers.append(m)
            if fused is not None:
                out = PointStamped()
                out.header = scan.header
                out.point = Point(x=float(fused[0]), y=float(fused[1]), z=0.0)
                self.pub.publish(out)
                m = Marker()
                m.header = scan.header
                m.ns, m.id, m.type, m.action = "owner", 0, Marker.CYLINDER, Marker.ADD
                m.pose.position = out.point
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 0.5
                m.scale.z = 1.7
                cam = self.tracker.source == "camera+lidar"
                m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 0.5) if cam else (1.0, 0.6, 0.0, 0.5)
                m.lifetime.nanosec = 300_000_000
                ma.markers.append(m)
            self.mk_pub.publish(ma)

    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
