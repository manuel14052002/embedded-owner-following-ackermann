#!/usr/bin/env python3
"""
wheel_state_node.py                              (Raspberry Pi — a2r_bot_description)
═══════════════════════════════════════════════════════════════════════════════
MAKES RVIZ TELL THE TRUTH ABOUT THE DRIVE CHANNEL.

robot_state_publisher needs a /joint_states message for the six movable joints
in a2r_bot.urdf.xacro. Nothing on this robot has an encoder, so instead of
faking zeros (which makes RViz a pretty but useless picture) this node
publishes what the vehicle was *actually commanded to do*:

    steering angle   ← the last Twist the local planner sent to the firmware
    wheel spin       ← that command's linear velocity, integrated

Why that is worth a node
────────────────────────
It turns RViz into a live debugger for the two channels you cannot otherwise
see without putting the car on the floor:

  • Servo wag.  If the DWA/VFH candidate winner is hopping one steer bucket
    per cycle, the front wheels in RViz visibly shiver. That is the single
    most common failure mode in this planner stack and it is invisible in a
    log full of plausible-looking numbers.
  • Dead drive.  If the wheels are not spinning in RViz, the planner never
    commanded motion — so stop probing the ESC and look upstream.
  • Escape manoeuvres.  Pre-steer → reverse → opposite-lock exit reads as an
    obvious three-point turn instead of a wall of state strings.

Ackermann, not a single angle
─────────────────────────────
The planner produces ONE virtual steering angle δ for the bicycle model. The
two real front wheels must differ, or the model looks subtly wrong on tight
turns. Using turn radius R = L / tan δ:

    δ_inner = atan( L / (R − T/2) )        δ_outer = atan( L / (R + T/2) )

with L the wheelbase and T the track. At δ = 30° and this geometry the inner
wheel sits ~7° beyond the outer one — small, but it is the difference between
"a car" and "a box with four cylinders".

Subscribes
──────────
  /local_planner/cmd_out   geometry_msgs/Twist   executed command (preferred)
  /cmd_vel                 geometry_msgs/Twist   fallback if the planner is
                                                 not running (e.g. teleop)
Publishes
─────────
  /joint_states            sensor_msgs/JointState
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState


class WheelStateNode(Node):

    def __init__(self):
        super().__init__("wheel_state_node")

        # ── Geometry: MUST match common.xacro ────────────────────────────
        self.declare_parameter("wheelbase_m", 0.280)
        self.declare_parameter("track_m", 0.250)
        self.declare_parameter("wheel_radius_m", 0.060)
        self.declare_parameter("max_steer_deg", 30.0)

        # ── Sources ─────────────────────────────────────────────────────
        self.declare_parameter("cmd_out_topic", "/local_planner/cmd_out")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_rate_hz", 50.0)
        # How long after the last command we keep animating. Past this the
        # wheels freeze and the steering returns to centre, so a stale RViz
        # view is visibly stale rather than quietly lying.
        self.declare_parameter("cmd_timeout_s", 1.0)
        # Smooths the steering visual only. Do NOT raise this to hide servo
        # wag — the whole point is that you can see it.
        self.declare_parameter("steer_filter_hz", 12.0)

        g = lambda n: self.get_parameter(n).value
        self._L = float(g("wheelbase_m"))
        self._T = float(g("track_m"))
        self._r = float(g("wheel_radius_m"))
        self._max_steer = math.radians(float(g("max_steer_deg")))
        self._rate = float(g("publish_rate_hz"))
        self._timeout = float(g("cmd_timeout_s"))
        self._filt_hz = float(g("steer_filter_hz"))

        # ── State ───────────────────────────────────────────────────────
        self._v = 0.0                    # m/s, commanded
        self._steer = 0.0                # rad, commanded (bicycle model)
        self._steer_f = 0.0              # rad, filtered for display
        self._spin = 0.0                 # rad, accumulated wheel rotation
        self._last_cmd_t: Optional[float] = None
        self._last_tick = time.monotonic()
        self._have_planner = False       # planner wins over raw /cmd_vel

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Twist, str(g("cmd_out_topic")),
                                 self._cmd_out_cb, qos)
        self.create_subscription(Twist, str(g("cmd_vel_topic")),
                                 self._cmd_vel_cb, qos)

        self._pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / self._rate, self._tick)

        self.get_logger().info(
            f"wheel_state_node → /joint_states @ {self._rate:.0f} Hz "
            f"(L={self._L:.3f} m, T={self._T:.3f} m, r={self._r:.3f} m)")

    # ── Command sources ──────────────────────────────────────────────────

    def _cmd_out_cb(self, msg: Twist):
        """Executed command from the local planner: normalised throttle and
        steer (-1..1), see smart_planner_node's _set_out()."""
        self._have_planner = True
        self._v = float(msg.linear.x)
        # cmd_out carries NORMALISED steer (-1..1), see smart_planner_node
        # _set_out(); it was previously read as radians.
        s = max(-1.0, min(1.0, float(msg.angular.z)))
        self._steer = s * self._max_steer
        self._last_cmd_t = time.monotonic()

    def _cmd_vel_cb(self, msg: Twist):
        """Fallback: a standard Twist where angular.z is a YAW RATE. Convert
        with the bicycle model  δ = atan(ω·L / v). Ignored entirely while the
        planner is publishing, so we never animate two sources at once."""
        if self._have_planner:
            return
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        self._v = v
        if abs(v) > 1e-3:
            self._steer = math.atan2(w * self._L, abs(v))
        else:
            # Stationary with a yaw request: show full lock in that direction,
            # which is what the car would do the instant it started rolling.
            self._steer = math.copysign(self._max_steer, w) if abs(w) > 1e-3 else 0.0
        self._steer = max(-self._max_steer, min(self._max_steer, self._steer))
        self._last_cmd_t = time.monotonic()

    # ── Ackermann split ──────────────────────────────────────────────────

    def _ackermann(self, delta: float):
        """One virtual steering angle → the two real front-wheel angles."""
        if abs(delta) < 1e-4:
            return 0.0, 0.0
        R = self._L / math.tan(abs(delta))
        half_t = self._T / 2.0
        # Guard: at full lock on a short wheelbase R can approach half_t,
        # which would blow up the inner angle.
        inner = math.atan2(self._L, max(R - half_t, 1e-3))
        outer = math.atan2(self._L, R + half_t)
        inner = min(inner, self._max_steer * 1.35)
        if delta > 0:                       # turning left → left wheel inner
            return inner, outer
        return -outer, -inner

    # ── Publish ──────────────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / self._rate

        stale = (self._last_cmd_t is None) or (now - self._last_cmd_t) > self._timeout
        v = 0.0 if stale else self._v
        target_steer = 0.0 if stale else self._steer

        # First-order lag on the displayed steering, matching a servo's own
        # response so the picture is honest rather than instantaneous.
        alpha = 1.0 - math.exp(-2.0 * math.pi * self._filt_hz * dt)
        self._steer_f += (target_steer - self._steer_f) * min(1.0, alpha)

        # Integrate wheel rotation: ω = v / r
        self._spin += (v / self._r) * dt
        if self._spin > 2.0 * math.pi:
            self._spin -= 2.0 * math.pi
        elif self._spin < -2.0 * math.pi:
            self._spin += 2.0 * math.pi

        left_steer, right_steer = self._ackermann(self._steer_f)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            "front_left_steer_joint",
            "front_right_steer_joint",
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ]
        js.position = [
            left_steer, right_steer,
            self._spin, self._spin, self._spin, self._spin,
        ]
        js.velocity = [0.0, 0.0] + [v / self._r] * 4
        self._pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = WheelStateNode()
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
