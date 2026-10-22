#!/usr/bin/env python3
"""
sensor_fusion.launch.py
=======================
Starts the three fusion nodes:

    imu_filter_node   /imu/data_raw + /imu/mag  ->  /imu/data
    ekf_state_node    /imu/data + command       ->  /odometry/filtered + TF
    scan_fusion_node  /scan + sonars            ->  /scan/fused

Usage
-----
    ros2 launch a2r_bot_sensor_fusion sensor_fusion.launch.py
    ros2 launch a2r_bot_sensor_fusion sensor_fusion.launch.py use_mag:=false
    ros2 launch a2r_bot_sensor_fusion sensor_fusion.launch.py publish_tf:=false

`publish_tf:=false` is what you want when the SLAM node is running, because
SLAM publishes map -> odom and expects someone else to own odom -> base.  By
default that someone is this EKF, which is correct; set it false only if you
have another odometry source.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('a2r_bot_sensor_fusion')
    default_params = PathJoinSubstitution([pkg, 'config', 'fusion.yaml'])

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument(
            'use_mag', default_value='true',
            description='Use the AK8963 for absolute yaw. Turn off if the '
                        'magnetometer sits near the motor.'),
        DeclareLaunchArgument(
            'publish_tf', default_value='true',
            description='EKF publishes odom -> base_footprint.'),
        DeclareLaunchArgument(
            'command_topic', default_value='/local_planner/cmd_out'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('output_scan', default_value='/scan/fused'),
        DeclareLaunchArgument(
            'imu_filter', default_value='true',
            description='Set false if your IMU already reports orientation.'),
        DeclareLaunchArgument('ekf', default_value='true'),
        DeclareLaunchArgument('scan_fusion', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    log = ['--ros-args', '--log-level', LaunchConfiguration('log_level')]

    imu = Node(
        package='a2r_bot_sensor_fusion',
        executable='imu_filter_node',
        name='imu_filter_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_mag': LaunchConfiguration('use_mag')},
        ],
        arguments=log,
        condition=IfCondition(LaunchConfiguration('imu_filter')),
    )

    ekf = Node(
        package='a2r_bot_sensor_fusion',
        executable='ekf_state_node',
        name='ekf_state_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'publish_tf': LaunchConfiguration('publish_tf'),
                'command_topic': LaunchConfiguration('command_topic'),
            },
        ],
        arguments=log,
        condition=IfCondition(LaunchConfiguration('ekf')),
    )

    scan = Node(
        package='a2r_bot_sensor_fusion',
        executable='scan_fusion_node',
        name='scan_fusion_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'scan_topic': LaunchConfiguration('scan_topic'),
                'output_topic': LaunchConfiguration('output_scan'),
            },
        ],
        arguments=log,
        condition=IfCondition(LaunchConfiguration('scan_fusion')),
    )

    return LaunchDescription(args + [imu, ekf, scan])
