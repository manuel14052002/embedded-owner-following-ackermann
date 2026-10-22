#!/usr/bin/env python3
"""
tracker_node.py — person detection + follow controller.
Subscribes:
  /camera/color/image_raw   (bgr8)
  /camera/depth/image_raw   (16UC1, aligned)
Publishes:
  /cmd_vel        (geometry_msgs/Twist)   — throttle + steering intent
  /robot_state    (std_msgs/String)       — IDLE / FOLLOWING / LOST
  /owner_distance (std_msgs/Float32)      — metres to tracked person

Control:
  distance error → linear.x  (PID, forward-only)
  bearing error  → angular.z (PID, toward person)
  depth measured from median of torso patch (metric, robust)
  target smoothing (EMA) + accel-limited output → no jitter

Safety:
  * All output clamped to conservative max_linear / max_angular.
  * Person lost > lost_timeout → publish zero Twist (→ bridge STOP) + LOST.
  * Never reverses in follow mode.
"""

import math
import time
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.lim = kp, ki, kd, limit
        self.i = 0.0
        self.prev = 0.0
        self.t = time.monotonic()

    def reset(self):
        self.i = 0.0
        self.prev = 0.0
        self.t = time.monotonic()

    def step(self, err):
        now = time.monotonic()
        dt = max(1e-3, now - self.t)
        self.t = now
        self.i = max(-self.lim, min(self.lim, self.i + err * dt))
        d = (err - self.prev) / dt
        self.prev = err
        return max(-self.lim, min(self.lim, self.kp*err + self.ki*self.i + self.kd*d))


class TrackerNode(Node):
    MIN_DRIVE_TURN = 0.12   # Ackermann: needs motion to steer

    def __init__(self):
        super().__init__("tracker_node")

        # ── Parameters (START LOW) ─────────────────────────────────────
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("target_dist_m", 1.2)
        self.declare_parameter("safe_distance_m", 0.8)
        self.declare_parameter("max_linear", 0.5)     # conservative
        self.declare_parameter("max_angular", 0.8)
        self.declare_parameter("max_accel", 0.3)
        self.declare_parameter("confidence", 0.45)
        self.declare_parameter("lost_timeout_s", 1.0)
        self.declare_parameter("ema_alpha", 0.20)
        self.declare_parameter("hfov_deg", 69.4)      # D435 color HFOV
        self.declare_parameter("deadband_dist", 0.18)
        self.declare_parameter("deadband_bear", 0.14)
        self.declare_parameter("auto_start", True)    # follow without a trigger

        g = lambda n: self.get_parameter(n).value
        self._tgt      = float(g("target_dist_m"))
        self._safe     = float(g("safe_distance_m"))
        self._max_lin  = float(g("max_linear"))
        self._max_ang  = float(g("max_angular"))
        self._max_acc  = float(g("max_accel"))
        self._conf     = float(g("confidence"))
        self._lost_to  = float(g("lost_timeout_s"))
        self._alpha    = float(g("ema_alpha"))
        self._hfov     = math.radians(float(g("hfov_deg")))
        self._db_d     = float(g("deadband_dist"))
        self._db_b     = float(g("deadband_bear"))
        self._active   = bool(g("auto_start"))

        self._lin_pid = PID(0.6, 0.0, 0.05, self._max_lin)
        self._ang_pid = PID(1.1, 0.0, 0.04, self._max_ang)

        # ── State ──────────────────────────────────────────────────────
        self._color = None
        self._depth = None
        self._ema_d: Optional[float] = None
        self._ema_b: Optional[float] = None
        self._out_lin = 0.0
        self._out_ang = 0.0
        self._last_seen = 0.0
        self._last_tick = time.monotonic()
        self._state = "IDLE"

        # ── YOLO ───────────────────────────────────────────────────────
        self._model = None
        if HAS_YOLO:
            try:
                self._model = YOLO(g("model_path"))
                self.get_logger().info(f"YOLO loaded: {g('model_path')}")
            except Exception as e:
                self.get_logger().error(f"YOLO load failed: {e}")

        # ── I/O ────────────────────────────────────────────────────────
        self.create_subscription(Image, "/camera/color/image_raw", self._color_cb, 5)
        self.create_subscription(Image, "/camera/depth/image_raw", self._depth_cb, 5)
        self.create_subscription(String, "/robot_state_cmd", self._statecmd_cb, 10)
        self._cmd_pub  = self.create_publisher(Twist,   "/cmd_vel",       10)
        self._st_pub   = self.create_publisher(String,  "/robot_state",   10)
        self._dist_pub = self.create_publisher(Float32, "/owner_distance", 10)

        self.create_timer(1.0 / 15.0, self._control)
        self.get_logger().info(f"Tracker ready (auto_start={self._active})")

    # ── Camera callbacks ───────────────────────────────────────────────
    def _color_cb(self, msg: Image):
        if not HAS_NP: return
        self._color = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _depth_cb(self, msg: Image):
        if not HAS_NP: return
        self._depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)

    def _statecmd_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd in ("FOLLOW", "START"):
            self._active = True
        elif cmd in ("STOP", "IDLE"):
            self._active = False
            self._publish_zero()

    # ── Control loop ────────────────────────────────────────────────────
    def _control(self):
        if not self._active or self._model is None or self._color is None:
            return
        if self._depth is None:
            return

        results = self._model.predict(self._color, classes=[0],
                                      conf=self._conf, verbose=False)
        box = self._largest_person(results)
        if box is None:
            self._handle_lost()
            return

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        w = self._color.shape[1]

        dist = self._torso_depth(x1, y1, x2, y2)
        if dist is None:
            self._handle_lost()
            return

        bearing = ((cx - w/2.0) / (w/2.0)) * (self._hfov / 2.0)

        # EMA smoothing
        if self._ema_d is None:
            self._ema_d, self._ema_b = dist, bearing
        else:
            if abs(dist - self._ema_d) < 0.6:      # outlier gate
                self._ema_d += self._alpha * (dist - self._ema_d)
            self._ema_b += self._alpha * (bearing - self._ema_b)

        self._last_seen = time.monotonic()
        self._set_state("FOLLOWING")
        self._dist_pub.publish(Float32(data=float(self._ema_d)))

        # HARD SAFE-DISTANCE STOP — overrides follow, never approach a close person
        if self._ema_d <= self._safe:
            self._lin_pid.reset()
            a = -self._ang_pid.step(self._ema_b) * 0.5
            self._publish(0.0, a if abs(self._ema_b) > self._db_b else 0.0)
            return

        # PID
        dist_err = self._ema_d - self._tgt
        lin = max(0.0, self._lin_pid.step(dist_err))     # forward-only
        ang = -self._ang_pid.step(self._ema_b)

        # Ackermann: motion needed to steer
        if abs(ang) > 0.1 and 0.0 < lin < self.MIN_DRIVE_TURN:
            lin = self.MIN_DRIVE_TURN
        # Arrived
        if abs(dist_err) < self._db_d and abs(self._ema_b) < self._db_b:
            lin = ang = 0.0

        self._publish(lin, ang)

    # ── Helpers ─────────────────────────────────────────────────────────
    def _largest_person(self, results) -> Optional[Tuple]:
        boxes: List[Tuple] = []
        for r in results:
            for b in r.boxes:
                boxes.append(tuple(map(int, b.xyxy[0].tolist())))
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

    def _torso_depth(self, x1, y1, x2, y2) -> Optional[float]:
        d = self._depth
        h, w = d.shape[:2]
        bw, bh = x2-x1, y2-y1
        px1 = max(0, int(x1+0.30*bw)); px2 = min(w, int(x1+0.70*bw))
        py1 = max(0, int(y1+0.25*bh)); py2 = min(h, int(y1+0.60*bh))
        patch = d[py1:py2, px1:px2].astype(np.float32) / 1000.0   # mm→m
        valid = patch[(patch > 0.25) & (patch < 6.0)]
        if valid.size < 20:
            return None
        return float(np.median(valid))

    def _handle_lost(self):
        if time.monotonic() - self._last_seen > self._lost_to:
            self._set_state("LOST")
            self._ema_d = self._ema_b = None
            self._publish_zero()

    def _publish(self, lin, ang):
        now = time.monotonic()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now
        # accel limit
        self._out_lin += max(-self._max_acc*dt, min(self._max_acc*dt, lin-self._out_lin))
        self._out_ang += max(-1.2*dt, min(1.2*dt, ang-self._out_ang))
        t = Twist()
        t.linear.x  = max(0.0, min(self._max_lin, self._out_lin))
        t.angular.z = max(-self._max_ang, min(self._max_ang, self._out_ang))
        self._cmd_pub.publish(t)

    def _publish_zero(self):
        self._out_lin = self._out_ang = 0.0
        self._cmd_pub.publish(Twist())

    def _set_state(self, s):
        if s != self._state:
            self._state = s
            self._st_pub.publish(String(data=s))


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
