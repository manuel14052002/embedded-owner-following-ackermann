#!/usr/bin/env python3
"""
robot_base.launch.py — a2r_bot_bringup
======================================
Everything the robot needs before it does anything in particular.

    ros2 launch a2r_bot_bringup robot_base.launch.py

This is the layer both follow mode and SLAM mode sit on top of:

    robot_state_publisher   the URDF
    lidar driver            rplidar A2M12 (default) | ydlidar | none
    serial_bridge_node      the ONLY owner of /dev/ttyARDUINO
    firmware_monitor_node   one Bool answering "is the hardware healthy?"
    imu_filter_node         /imu/data_raw -> /imu/data, with orientation
    ekf_state_node          -> /odometry/filtered and odom -> base_footprint
    scan_fusion_node        lidar + both sonars -> /scan/fused

Start it on its own when bringing up hardware. Everything published here is
inspectable without any planner running, which is the right way to find out
whether a sensor is wired correctly:

    ros2 topic echo /battery
    ros2 topic echo /arduino/health_text
    ros2 topic echo /imu/rpy
    ros2 run rviz2 rviz2 -d $(ros2 pkg prefix a2r_bot_description)/share/a2r_bot_description/rviz/model.rviz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc = FindPackageShare('a2r_bot_description')
    arduino = FindPackageShare('a2r_bot_arduino')
    fusion = FindPackageShare('a2r_bot_sensor_fusion')

    args = [
        DeclareLaunchArgument('lidar', default_value='rplidar',
                              description='rplidar (A2M12, default) | ydlidar | none'),
        DeclareLaunchArgument('arduino', default_value='false'),
        DeclareLaunchArgument('fusion', default_value='true'),
        DeclareLaunchArgument('use_mag', default_value='true',
                              description='Turn off if the GY-91 sits near '
                                          'the motor or the ESC leads.'),
        DeclareLaunchArgument('port', default_value='/dev/ttyARDUINO'),
        DeclareLaunchArgument('styling', default_value='true'),
        DeclareLaunchArgument('joint_source', default_value='cmd'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [desc, 'launch', 'description.launch.py'])),
        launch_arguments={
            'joint_source': LaunchConfiguration('joint_source'),
            'styling': LaunchConfiguration('styling'),
        }.items(),
    )

    arduino_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [arduino, 'launch', 'arduino_bringup.launch.py'])),
        launch_arguments={
            'port': LaunchConfiguration('port'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('arduino')),
    )

    fusion_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [fusion, 'launch', 'sensor_fusion.launch.py'])),
        launch_arguments={
            'use_mag': LaunchConfiguration('use_mag'),
            'publish_tf': 'true',
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('fusion')),
    )

    ydlidar = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_node', output='screen',
        parameters=[{'port': '/dev/ttyLIDAR', 'frame_id': 'lidar_link',
                     'baudrate': 115200, 'lidar_type': 1, 'device_type': 0,
                     'sample_rate': 3, 'frequency': 7.0,
                     'angle_min': -180.0, 'angle_max': 180.0,
                     'range_min': 0.10, 'range_max': 8.0,
                     'inverted': False, 'auto_reconnect': True,
                     'isSingleChannel': True}],
        condition=LaunchConfigurationEquals('lidar', 'ydlidar'))

    rplidar = Node(
        package='sllidar_ros2', executable='sllidar_node',
        name='rplidar_node', output='screen',
        parameters=[{
            'serial_port': '/dev/ttyLIDAR',
            # A2M12 is 256000, NOT the A2M8's 115200. At 115200 the driver
            # opens the port, reports no error, and never publishes a scan.
            'serial_baudrate': 256000,
            'frame_id': 'lidar_link',
            'angle_compensate': True,
            'scan_mode': 'Standard',   # ~4k samples/s: 4x less CPU than Boost
            'inverted': False,
        }],
        condition=LaunchConfigurationEquals('lidar', 'rplidar'))

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                output='screen',
                arguments=['-d', PathJoinSubstitution(
                    [desc, 'rviz', 'model.rviz'])],
                condition=IfCondition(LaunchConfiguration('use_rviz')))

    # ---- lidar driver availability (fix_sllidar_launch.py) ---
    # The robot's lidar is the RPLIDAR A2M12 on sllidar_ros2. Humble resolves
    # a Node's package while the launch runs, so a driver that is not
    # installed must not be in the description at all, even behind a
    # condition. Missing drivers become a log line instead of a crash.
    from launch.actions import LogInfo
    from ament_index_python.packages import get_package_prefix, PackageNotFoundError

    def _have(pkg):
        try:
            get_package_prefix(pkg)
            return True
        except PackageNotFoundError:
            return False

    if not _have('ydlidar_ros2_driver'):
        ydlidar = LogInfo(
            msg='lidar:=ydlidar ignored - ydlidar_ros2_driver is not installed '
                '(this robot uses the RPLIDAR A2M12: lidar:=rplidar)',
            condition=LaunchConfigurationEquals('lidar', 'ydlidar'))
    if not _have('sllidar_ros2'):
        rplidar = LogInfo(
            msg='!! sllidar_ros2 (RPLIDAR A2M12 driver) is NOT installed - there will be '
                'NO /scan. Fix: cd <workspace>/src && git clone '
                'https://github.com/Slamtec/sllidar_ros2 && cd .. && '
                'colcon build --packages-select sllidar_ros2',
            condition=LaunchConfigurationEquals('lidar', 'rplidar'))

    return LaunchDescription(args + [
        description, arduino_stack, fusion_stack, ydlidar, rplidar, rviz,
    ])
