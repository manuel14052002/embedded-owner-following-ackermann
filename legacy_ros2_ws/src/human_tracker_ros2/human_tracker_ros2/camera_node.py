"""
Camera Node
Handles Intel RealSense D435 camera interface and streaming
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealsenseCameraNode(Node):
    """
    ROS 2 node for Intel RealSense D435 camera.
    Publishes color and depth images, along with camera info.
    """
    
    def __init__(self):
        super().__init__('realsense_camera_node')
        
        # Declare parameters
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('rgb_width', 640)
        self.declare_parameter('rgb_height', 480)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('enable_depth', True)
        
        # Get parameters
        self.frame_rate = self.get_parameter('frame_rate').value
        self.rgb_width = self.get_parameter('rgb_width').value
        self.rgb_height = self.get_parameter('rgb_height').value
        self.depth_width = self.get_parameter('depth_width').value
        self.depth_height = self.get_parameter('depth_height').value
        self.enable_depth = self.get_parameter('enable_depth').value
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # ROS 2 Publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        # Initialize RealSense pipeline
        self.pipeline = None
        self.config = None
        self.align = None
        
        try:
            self.initialize_camera()
            
            # Create timer for publishing frames
            timer_period = 1.0 / self.frame_rate
            self.timer = self.create_timer(timer_period, self.timer_callback)
            
            self.get_logger().info('RealSense camera node initialized successfully')
            self.get_logger().info(f'Publishing at {self.frame_rate} FPS')
            self.get_logger().info(f'RGB: {self.rgb_width}x{self.rgb_height}')
            self.get_logger().info(f'Depth: {self.depth_width}x{self.depth_height}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize RealSense camera: {str(e)}')
            raise
    
    def initialize_camera(self):
        """Initialize the RealSense camera pipeline."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        self.get_logger().info(f'Found device: {device.get_info(rs.camera_info.name)}')
        
        # Configure streams
        self.config.enable_stream(
            rs.stream.color,
            self.rgb_width,
            self.rgb_height,
            rs.format.bgr8,
            self.frame_rate
        )
        
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                rs.format.z16,
                self.frame_rate
            )
        
        # Start streaming
        profile = self.pipeline.start(self.config)
        
        # Create align object to align depth to color
        if self.enable_depth:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
        
        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        if self.enable_depth:
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Let camera warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()
    
    def create_camera_info_msg(self, intrinsics, stamp):
        """
        Create CameraInfo message from RealSense intrinsics.
        
        Args:
            intrinsics: RealSense camera intrinsics
            stamp: ROS timestamp
            
        Returns:
            CameraInfo message
        """
        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = 'camera_link'
        
        camera_info.width = intrinsics.width
        camera_info.height = intrinsics.height
        
        # Camera matrix K
        camera_info.k = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        # Distortion coefficients
        camera_info.d = [
            intrinsics.coeffs[0],  # k1
            intrinsics.coeffs[1],  # k2
            intrinsics.coeffs[2],  # p1
            intrinsics.coeffs[3],  # p2
            intrinsics.coeffs[4]   # k3
        ]
        
        # Rectification matrix R (identity for monocular camera)
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # Projection matrix P
        camera_info.p = [
            intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
            0.0, intrinsics.fy, intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        camera_info.distortion_model = 'plumb_bob'
        
        return camera_info
    
    def timer_callback(self):
        """Timer callback to capture and publish frames."""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            if self.enable_depth and self.align:
                # Align depth to color
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None
            
            if not color_frame:
                self.get_logger().warning('No color frame received')
                return
            
            # Get timestamp
            stamp = self.get_clock().now().to_msg()
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = 'camera_color_optical_frame'
            self.color_pub.publish(color_msg)
            
            # Publish color camera info
            color_info_msg = self.create_camera_info_msg(self.color_intrinsics, stamp)
            self.color_info_pub.publish(color_info_msg)
            
            # Publish depth if enabled
            if self.enable_depth and depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = 'camera_depth_optical_frame'
                self.depth_pub.publish(depth_msg)
                
                # Publish depth camera info
                depth_info_msg = self.create_camera_info_msg(self.depth_intrinsics, stamp)
                self.depth_info_pub.publish(depth_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
    
    def destroy_node(self):
        """Cleanup when node is destroyed."""
        if self.pipeline:
            self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        camera_node = RealsenseCameraNode()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'camera_node' in locals():
            camera_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
