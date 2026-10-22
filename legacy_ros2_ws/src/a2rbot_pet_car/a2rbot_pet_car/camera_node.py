#!/usr/bin/env python3
"""
camera_node.py — RealSense D435 publisher.
Publishes:
  /camera/color/image_raw   (sensor_msgs/Image, bgr8)
  /camera/depth/image_raw   (sensor_msgs/Image, 16UC1, aligned to color)
Depth is aligned to color so a pixel in the color image maps to the same
pixel in depth — required by the tracker for per-person distance.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import numpy as np
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        w = self.get_parameter("width").value
        h = self.get_parameter("height").value
        fps = self.get_parameter("fps").value

        self._color_pub = self.create_publisher(Image, "/camera/color/image_raw", 5)
        self._depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 5)

        if not HAS_RS:
            self.get_logger().error("pyrealsense2 not installed")
            raise RuntimeError("pyrealsense2 missing")

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color)
        self.get_logger().info(f"RealSense D435 started {w}x{h}@{fps}")
        self.create_timer(1.0 / fps, self._tick)

    def _tick(self):
        frames = self._pipe.poll_for_frames()
        if not frames:
            return
        frames = self._align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            return
        stamp = self.get_clock().now().to_msg()

        color = np.asanyarray(c.get_data())
        msg_c = Image()
        msg_c.header.stamp = stamp
        msg_c.header.frame_id = "camera_color_optical_frame"
        msg_c.height, msg_c.width = color.shape[:2]
        msg_c.encoding = "bgr8"
        msg_c.step = msg_c.width * 3
        msg_c.data = color.tobytes()
        self._color_pub.publish(msg_c)

        depth = np.asanyarray(d.get_data())  # uint16, mm
        msg_d = Image()
        msg_d.header.stamp = stamp
        msg_d.header.frame_id = "camera_depth_optical_frame"
        msg_d.height, msg_d.width = depth.shape[:2]
        msg_d.encoding = "16UC1"
        msg_d.step = msg_d.width * 2
        msg_d.data = depth.tobytes()
        self._depth_pub.publish(msg_d)

    def destroy_node(self):
        try:
            self._pipe.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
