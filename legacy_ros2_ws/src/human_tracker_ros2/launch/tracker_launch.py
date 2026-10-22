"""
Launch file for human tracker system
Starts camera node and tracker node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for human tracker system."""
    
    # Declare launch arguments
    frame_rate_arg = DeclareLaunchArgument(
        'frame_rate',
        default_value='30',
        description='Camera frame rate'
    )
    
    rgb_width_arg = DeclareLaunchArgument(
        'rgb_width',
        default_value='640',
        description='RGB image width'
    )
    
    rgb_height_arg = DeclareLaunchArgument(
        'rgb_height',
        default_value='480',
        description='RGB image height'
    )
    
    target_distance_arg = DeclareLaunchArgument(
        'target_follow_distance',
        default_value='1.5',
        description='Desired following distance in meters'
    )
    
    # Camera node
    camera_node = Node(
        package='human_tracker_ros2',
        executable='camera_node',
        name='realsense_camera',
        output='screen',
        parameters=[{
            'frame_rate': LaunchConfiguration('frame_rate'),
            'rgb_width': LaunchConfiguration('rgb_width'),
            'rgb_height': LaunchConfiguration('rgb_height'),
            'depth_width': LaunchConfiguration('rgb_width'),
            'depth_height': LaunchConfiguration('rgb_height'),
            'enable_depth': True
        }],
        emulate_tty=True
    )
    
    # Tracker node
    tracker_node = Node(
        package='human_tracker_ros2',
        executable='tracker_node',
        name='human_tracker',
        output='screen',
        parameters=[{
            'publish_annotated_image': True,
            'publish_control_commands': True,
            'target_follow_distance': LaunchConfiguration('target_follow_distance'),
            'max_linear_speed': 0.5,
            'max_angular_speed': 1.0
        }],
        emulate_tty=True
    )
    
    return LaunchDescription([
        frame_rate_arg,
        rgb_width_arg,
        rgb_height_arg,
        target_distance_arg,
        camera_node,
        tracker_node
    ])
