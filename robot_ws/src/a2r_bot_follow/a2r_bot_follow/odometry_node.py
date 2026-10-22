#!/usr/bin/env python3
"""
odometry_node.py                                   (Raspberry Pi — a2r_bot_follow)
═══════════════════════════════════════════════════════════════════════════════
IMU + COMMAND-INTEGRATED DEAD RECKONING (no wheel encoder needed)

  YAW (θ)   ← integrate gyro angular_velocity.z (bias-corrected)   [accurate]
  X, Y      ← integrate the COMMANDED velocity we sent to the firmware
              at the current heading                               [approximate]

Over 2–3 seconds this is close enough for the planner to remember what it
just swerved around. Longer-term it drifts (wheel slip, throttle
non-linearity) — out of scope for local reactive planning.

Subscribes:
  /imu/data_raw            sensor_msgs/Imu
  /local_planner/cmd_out   geometry_msgs/Twist  (last cmd the planner sent
                                                 to the firmware, in m/s)
Publishes:
  /odom                    nav_msgs/Odometry
  TF: odom → base_link
═══════════════════════════════════════════════════════════════════════════════
"""
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class OdometryNode(Node):
    def __init__(self):
        super().__init__("odometry_node")

        self.declare_parameter("imu_topic",        "/imu/data_raw")
        self.declare_parameter("cmd_out_topic",    "/local_planner/cmd_out")
        self.declare_parameter("odom_topic",       "/odom")
        self.declare_parameter("odom_frame",       "odom")
        self.declare_parameter("base_frame",       "base_link")
        self.declare_parameter("publish_rate_hz",  30.0)
        self.declare_parameter("imu_bias_samples", 100)
        # Fudge factor: commanded 0.4 m/s → actual maybe 0.36 m/s. 0.9 default.
        self.declare_parameter("cmd_scale",        0.9)

        g = lambda n: self.get_parameter(n).value
        self._imu_topic  = str(g("imu_topic"))
        self._cmd_topic  = str(g("cmd_out_topic"))
        self._odom_topic = str(g("odom_topic"))
        self._odom_frame = str(g("odom_frame"))
        self._base_frame = str(g("base_frame"))
        self._pub_rate   = float(g("publish_rate_hz"))
        self._bias_n     = int(g("imu_bias_samples"))
        self._cmd_scale  = float(g("cmd_scale"))

        self._x = self._y = self._yaw = 0.0
        self._vx = self._wz = 0.0
        self._last_imu_t: Optional[float] = None
        self._last_cmd_t: Optional[float] = None
        self._gyro_bias = 0.0
        self._bias_sum = 0.0
        self._bias_count = 0
        self._bias_done = False

        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Imu,   self._imu_topic, self._imu_cb, sub_qos)
        self.create_subscription(Twist, self._cmd_topic, self._cmd_out_cb, sub_qos)
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, pub_qos)
        self._tf_bc = TransformBroadcaster(self)
        self.create_timer(1.0 / self._pub_rate, self._publish_loop)

        self.get_logger().info(
            f"OdometryNode (IMU-only, cmd-integrated) → /odom + TF "
            f"{self._odom_frame}→{self._base_frame} | calibrating gyro bias "
            f"({self._bias_n} samples, keep car still)…")

    def _imu_cb(self, msg: Imu):
        wz = msg.angular_velocity.z
        now = time.monotonic()
        if not self._bias_done:
            self._bias_sum += wz
            self._bias_count += 1
            if self._bias_count >= self._bias_n:
                self._gyro_bias = self._bias_sum / self._bias_count
                self._bias_done = True
                self.get_logger().info(
                    f"Gyro bias: {self._gyro_bias:+.4f} rad/s "
                    f"({math.degrees(self._gyro_bias):+.2f}°/s)")
            self._last_imu_t = now
            return
        wz_c = wz - self._gyro_bias
        self._wz = wz_c
        if self._last_imu_t is not None:
            dt = now - self._last_imu_t
            if 0.0 < dt < 0.5:
                self._yaw += wz_c * dt
                if self._yaw > math.pi:    self._yaw -= 2 * math.pi
                elif self._yaw < -math.pi: self._yaw += 2 * math.pi
        self._last_imu_t = now

    def _cmd_out_cb(self, msg: Twist):
        now = time.monotonic()
        v_cmd = msg.linear.x * self._cmd_scale
        self._vx = v_cmd
        if self._last_cmd_t is not None:
            dt = now - self._last_cmd_t
            if 0.0 < dt < 0.5:
                self._x += v_cmd * dt * math.cos(self._yaw)
                self._y += v_cmd * dt * math.sin(self._yaw)
        self._last_cmd_t = now

    def _publish_loop(self):
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.rotation = yaw_to_quat(self._yaw)
        self._tf_bc.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation = yaw_to_quat(self._yaw)
        odom.twist.twist.linear.x = self._vx
        odom.twist.twist.angular.z = self._wz
        odom.pose.covariance = [0.10, 0.0,  0.0,  0.0,  0.0,  0.0,
                                0.0,  0.10, 0.0,  0.0,  0.0,  0.0,
                                0.0,  0.0,  1e6,  0.0,  0.0,  0.0,
                                0.0,  0.0,  0.0,  1e6,  0.0,  0.0,
                                0.0,  0.0,  0.0,  0.0,  1e6,  0.0,
                                0.0,  0.0,  0.0,  0.0,  0.0,  0.15]
        self._odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
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
