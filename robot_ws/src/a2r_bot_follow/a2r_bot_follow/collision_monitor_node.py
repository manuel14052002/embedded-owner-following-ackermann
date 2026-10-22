#!/usr/bin/env python3
"""
collision_monitor_node.py — last line of defence before the motors
═══════════════════════════════════════════════════════════════════════════════
    twist_mux --/cmd_vel_mux--> collision_monitor --/local_planner/cmd_out--> serial bridge

Ignores the costmap and every planner. Looks only at the RAW lidar points and
the final command, the way Nav2's collision_monitor does in "approach" mode:

  1. Sweep the car's measured footprint (ackermann.py) along the exact
     kinematic-bicycle path the command would drive, up to a speed-dependent distance.
  2. d = how far the car can travel before a lidar point enters the
     footprint (+ margin).
  3. STOP zone:     d <= stop_distance + v*reaction  -> v = 0 (steering kept)
     SLOWDOWN zone: d <  that + v*slowdown_time      -> v scaled down, but
                    never below min_drive_speed (the car stalls below it)
                    unless it is inside the stop zone.
  4. No fresh scan -> v = 0. A monitor that cannot see does not wave traffic on.

COMMAND CONVENTION (same as the whole a2r_bot drive chain): Twist
  linear.x  = throttle fraction  -1..1  (x max_speed_mps = m/s)
  angular.z = steering fraction  -1..1  (x max_steer_deg)
Steering-only commands (v = 0, e.g. the planner's pre-steer) pass untouched.

The output keeps the old topic name /local_planner/cmd_out on purpose: the
serial bridge, the EKF, wheel_state_node and odometry_node all read the
EXECUTED command from it, and now that is exactly what it carries.
"""
import json
import math
import time

import numpy as np

try:
    from a2r_bot_follow.ackermann import (Ackermann, WHEELBASE_M, FOOT_FRONT_M,
                                          FOOT_REAR_M, FOOT_HALF_W_M, LIDAR_X_M, LIDAR_Y_M)
except ImportError:                       # run straight from the source tree
    from ackermann import (Ackermann, WHEELBASE_M, FOOT_FRONT_M,
                           FOOT_REAR_M, FOOT_HALF_W_M, LIDAR_X_M, LIDAR_Y_M)


class CollisionCore:
    """Pure geometry, testable without ROS. Kinematics and footprint come from
    ackermann.py, the same module the planner uses, so the two can no longer
    disagree about what is reachable."""

    def __init__(self, front=FOOT_FRONT_M, rear=FOOT_REAR_M, half_w=FOOT_HALF_W_M, margin=0.03,
                 wheelbase=WHEELBASE_M, max_mps=1.2, max_steer=math.radians(30.0),
                 stop_dist=0.10, reaction_s=0.15, slowdown_s=1.0,
                 min_drive=0.28, max_look=2.0, release_hyst=0.05,
                 poly_gain_s=0.0, poly_floor=0.5):
        self.car = Ackermann(wheelbase, max_steer, front, rear, half_w)
        self.f, self.r, self.w, self.m = front, rear, half_w, margin
        self.L, self.max_mps, self.max_steer = wheelbase, max_mps, max_steer
        self.stop_dist, self.react, self.slow_s = stop_dist, reaction_s, slowdown_s
        self.min_drive, self.max_look, self.hyst = min_drive, max_look, release_hyst
        # v2.5 velocity polygon: a second footprint, widened SIDEWAYS only by
        # poly_gain_s x speed (front and back unchanged), swept on the same poses
        self.poly_gain, self.poly_floor = poly_gain_s, poly_floor
        self.car_wide = Ackermann(wheelbase, max_steer, front, rear, half_w)
        self.stopped = False

    def collision_distance(self, pts: np.ndarray, v: float, steer: float, look: float) -> float:
        """Travel distance along the arc before a point enters the (margin-grown)
        footprint. Points already inside the margin band only count once they
        reach the bare body, so the car can always back away from a wall it
        stopped close to (it used to be blocked in every direction)."""
        if pts is None or len(pts) == 0:
            return math.inf
        dist = look if v >= 0 else -look
        s, x, y, yaw = self.car.poses(steer, dist, 0.02)
        return self.car.first_hit(pts, s, x, y, yaw, self.m)

    def _two_hits(self, pts, v, steer, look, extra_w):
        """v2.5: ONE sweep of poses, two footprints - the core one and one
        widened sideways by extra_w. Same answer as two collision_distance
        calls for the core, at the cost of one extra (vectorised) hit test."""
        if pts is None or len(pts) == 0:
            return math.inf, math.inf
        s, x, y, yaw = self.car.poses(steer, look if v >= 0 else -look, 0.02)
        d = self.car.first_hit(pts, s, x, y, yaw, self.m)
        if extra_w <= 0.0:
            return d, d
        self.car_wide.w = self.w + extra_w
        return d, self.car_wide.first_hit(pts, s, x, y, yaw, self.m)

    def filter(self, v_n: float, s_n: float, pts, scan_fresh: bool):
        """Normalised command in -> (v_n, s_n, state, collision_distance)."""
        v = v_n * self.max_mps
        steer = s_n * self.max_steer
        if abs(v) < 0.01:
            self.stopped = False
            return v_n, s_n, "PASS (steer only)", math.inf
        if not scan_fresh:
            self.stopped = True
            self._stop_sign = math.copysign(1.0, v)
            return 0.0, s_n, "STOP (no fresh scan)", 0.0
        speed = abs(v)
        if self.stopped and getattr(self, "_stop_sign", 0.0) != math.copysign(1.0, v):
            self.stopped = False   # a forward STOP must not lock reverse (and vice versa)
        # a2r_pi_update2: during a planned multi-point turn at crawl speed (the planner says so on
        # /collision_monitor/crawl) the stop zone is crawl_stop + v*crawl_react instead
        crawl = bool(getattr(self, "crawl", False)) and speed <= float(getattr(self, "crawl_vmax", 0.0))
        stop_d = ((self.crawl_stop if crawl else self.stop_dist)
                  + speed * (self.crawl_react if crawl else self.react) + 0.06)  # v2.9: +ESC/scan coast
        slow_d = stop_d + speed * self.slow_s + 0.15
        look = min(self.max_look, slow_d + 0.05)
        d, dw = self._two_hits(pts, v, steer, look, self.poly_gain * speed)
        release = stop_d + (self.hyst if self.stopped else 0.0)
        if d <= release:
            self.stopped = True
            self._stop_sign = math.copysign(1.0, v)
            return 0.0, s_n, "STOP", d
        self.stopped = False
        # v2.5 VELOCITY POLYGON. An obstacle that enters only the sideways-
        # widened footprint is one the car would pass close BESIDE: speed is
        # cut in proportion to how soon it enters, never to a stop (the core
        # path is clear). Ahead/behind obstacles keep the proportional SLOW
        # and the STOP above exactly as before.
        k_long = 1.0 if d >= slow_d else (d - stop_d) / max(slow_d - stop_d, 1e-6)
        k_lat = 1.0
        if dw < min(d, slow_d):
            k_lat = max(self.poly_floor, (dw - stop_d) / max(slow_d - stop_d, 1e-6))
        k = min(k_long, k_lat)
        if k < 1.0:
            new = max(speed * k, min(speed, self.min_drive))
            tag = "SLOW" if k_long <= k_lat else "SLOW (beside)"
            return math.copysign(new / self.max_mps, v_n), s_n, tag, d
        return v_n, s_n, "PASS", d


def main(args=None):  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class CollisionMonitor(Node):
        def __init__(self):
            super().__init__("collision_monitor")
            p = self.declare_parameter
            p("input_topic", "/cmd_vel_mux")
            p("output_topic", "/local_planner/cmd_out")
            p("scan_topic", "/scan")
            p("base_frame", "base_link")
            p("footprint_front_m", FOOT_FRONT_M)
            p("footprint_rear_m", FOOT_REAR_M)
            p("footprint_half_width_m", FOOT_HALF_W_M)
            p("margin_m", 0.03)
            p("stop_distance_m", 0.10)
            p("reaction_time_s", 0.15)
            p("slowdown_time_s", 1.0)
            p("min_drive_speed", 0.28)
            p("max_speed_mps", 1.2)
            p("max_steer_deg", 30.0)
            p("wheelbase_m", WHEELBASE_M)
            p("scan_timeout_s", 0.5)
            p("ignore_stale_scan", False)
            p("scan_min_range", 0.10)
            p("lidar_offset_x", LIDAR_X_M)    # used only if TF base_link <- laser is missing (= URDF lidar_x)
            p("lidar_offset_y", LIDAR_Y_M)
            p("lidar_yaw_offset", 0.0)
            p("velocity_polygon_gain_s", 0.0)   # v2.5: side margin + gain x speed (0 = off)
            p("velocity_polygon_floor", 0.5)    # "beside" never slows below this fraction
            p("scan_voxel_m", 0.015)            # v2.8: one point per 1.5 cm square (0 = keep all)
            p("crawl_topic", "/collision_monitor/crawl")   # a2r_pi_update2: planner's K-turn flag
            p("crawl_stop_distance_m", 0.06)   # stop zone during a K-turn (< 0 = feature off)
            p("crawl_reaction_time_s", 0.35)
            p("crawl_max_speed_mps", 0.20)     # never above this speed
            p("crawl_hold_s", 0.4)             # flag older than this = normal profile
            p("state_rate_hz", 10.0)            # a2r_merge: /collision_monitor/state rate (+ every change)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.core = CollisionCore(float(g("footprint_front_m")), float(g("footprint_rear_m")),
                                      float(g("footprint_half_width_m")), float(g("margin_m")),
                                      float(g("wheelbase_m")), float(g("max_speed_mps")),
                                      math.radians(float(g("max_steer_deg"))),
                                      float(g("stop_distance_m")), float(g("reaction_time_s")),
                                      float(g("slowdown_time_s")), float(g("min_drive_speed")),
                                      poly_gain_s=float(g("velocity_polygon_gain_s")),
                                      poly_floor=float(g("velocity_polygon_floor")))
            self.base = str(g("base_frame"))
            self.scan_to = float(g("scan_timeout_s"))
            self.ignore_stale = bool(g("ignore_stale_scan"))
            self.min_r = float(g("scan_min_range"))
            self.voxel = float(g("scan_voxel_m"))
            self.state_period = 1.0 / max(1.0, float(g("state_rate_hz")))
            if abs(float(g("max_speed_mps")) - 1.2) > 1e-6:
                # v2.8: the drive command is a FRACTION; every node must turn it
                # into m/s with the same number (planner + EKF use 1.2). At 1.8
                # this node thought 0.24 m/s was 0.36 m/s, stopped early, and
                # its SLOW floor (0.168/1.8 = 0.093) sat below the planner's
                # stall floor (0.168/1.2 = 0.14): the car stalled and lurched.
                self.get_logger().warn(
                    f"collision_monitor max_speed_mps={g('max_speed_mps')} - must match the planner's "
                    "max_speed_mps (1.2) or stop/slow distances and the stall floor are wrong")
            self.lidar = (float(g("lidar_offset_x")), float(g("lidar_offset_y")), float(g("lidar_yaw_offset")))
            self._lidar_tf = False
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=False)
            self.TransformException = TransformException
            self.pts, self.scan_t = None, -math.inf
            self._last_state = ""
            self.pub = self.create_publisher(Twist, g("output_topic"), 10)
            self.pub_state = self.create_publisher(String, "/collision_monitor/state", 10)
            self.create_subscription(Twist, g("input_topic"), self._on_cmd, 10)
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data)
            self.Twist, self.String = Twist, String
            # a2r_pi_update2: crawl profile while the planner drives a multi-point turn
            self.core.crawl_stop = float(g("crawl_stop_distance_m"))
            self.core.crawl_react = float(g("crawl_reaction_time_s"))
            self.core.crawl_vmax = float(g("crawl_max_speed_mps")) if self.core.crawl_stop >= 0.0 else 0.0
            self.core.crawl, self._crawl_t, self._crawl_hold = False, -math.inf, float(g("crawl_hold_s"))
            from std_msgs.msg import Bool
            self.create_subscription(Bool, g("crawl_topic"), self._on_crawl, 10)
            self.get_logger().info(f"collision_monitor: {g('input_topic')} -> {g('output_topic')}")

        def _on_scan(self, scan):
            if not self._lidar_tf:
                try:
                    tf = self.tf.lookup_transform(self.base, scan.header.frame_id, rclpy.time.Time())
                    q = tf.transform.rotation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    self.lidar = (tf.transform.translation.x, tf.transform.translation.y, yaw)
                    self._lidar_tf = True
                    try:                         # v2.4: static pose cached - stop parsing ~100 /tf msgs/s
                        self.tfl.unregister()    # in Python inside the safety node (as owner_fusion does)
                    except Exception:            # noqa: BLE001
                        pass
                except self.TransformException:
                    pass
            r = np.asarray(scan.ranges, dtype=np.float64)
            lx, ly, lyaw = self.lidar
            # a2r_merge: beam cos/sin cached per scan layout (was ~2300 cos + sin every scan)
            key = (len(r), round(float(scan.angle_min), 7), round(float(scan.angle_increment), 9), lyaw)
            if getattr(self, "_trig_key", None) != key:
                ab = scan.angle_min + scan.angle_increment * np.arange(len(r)) + lyaw
                self._trig_key, self._cos, self._sin = key, np.cos(ab), np.sin(ab)
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(r) & (r > max(scan.range_min, self.min_r)) & (r < scan.range_max)
            x = lx + r[ok] * self._cos[ok]
            y = ly + r[ok] * self._sin[ok]
            pts = np.stack([x, y], axis=1)
            c = self.core      # drop the robot's own body (the lidar sees its own deck)
            body = (pts[:, 0] < c.f) & (pts[:, 0] > c.r) & (np.abs(pts[:, 1]) < c.w)
            pts = pts[~body]
            if self.voxel > 0.0 and len(pts):       # v2.8: ~2500 pts/turn from the lidar now
                k = np.floor(pts / self.voxel).astype(np.int64)
                _, first = np.unique(k[:, 0] * 1_000_003 + k[:, 1], return_index=True)
                pts = pts[np.sort(first)]
            self.pts, self.scan_t = pts, time.monotonic()

        def _on_crawl(self, msg):                   # a2r_pi_update2
            self._crawl_t = time.monotonic() if msg.data else -math.inf

        def _on_cmd(self, msg):
            fresh = self.ignore_stale or (time.monotonic() - self.scan_t <= self.scan_to)
            self.core.crawl = (time.monotonic() - self._crawl_t) <= self._crawl_hold   # a2r_pi_update2
            # v2.5: commands arrive at 50 Hz, the scan at 10 Hz. Same command +
            # same scan + same freshness = same answer, so reuse it (exact - the
            # STOP hysteresis sees the same inputs and takes the same branch).
            key = (float(msg.linear.x), float(msg.angular.z), self.scan_t, fresh, self.core.crawl)
            if key == getattr(self, "_ck", None):
                v, s, state, d = self._cv
            else:
                v, s, state, d = self.core.filter(float(msg.linear.x), float(msg.angular.z), self.pts, fresh)
                self._ck, self._cv = key, (v, s, state, d)
            out = self.Twist()
            out.linear.x, out.angular.z = float(v), float(s)
            self.pub.publish(out)
            base = state.split(" ")[0]
            if base != self._last_state and base in ("STOP", "SLOW") :
                self.get_logger().warn(f"collision_monitor {state}: obstacle {d:.2f} m along the path",
                                       throttle_duration_sec=1.0)   # a2r_v10 log: was every flap
            # a2r_merge: state at <= 10 Hz unless it changes (was a JSON message per command, 50 Hz,
            # parsed by the planner twice). The planner only needs it within 0.5 s.
            now = time.monotonic()
            changed = base != self._last_state
            self._last_state = base
            if not changed and now - getattr(self, "_state_t", -1.0) < self.state_period:
                return
            self._state_t = now
            self.pub_state.publish(self.String(data=json.dumps(
                {"state": state, "clear_m": None if math.isinf(d) else round(d, 2), "crawl": self.core.crawl,
                 "in_v": round(float(msg.linear.x), 3), "out_v": round(float(v), 3)})))

    rclpy.init(args=args)
    node = CollisionMonitor()
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
