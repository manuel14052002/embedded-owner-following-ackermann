#!/usr/bin/env python3
"""
slam.launch.py — a2r_bot_bringup
================================
Mapping mode, from the bringup package.

    ros2 launch a2r_bot_bringup slam.launch.py
    ros2 launch a2r_bot_bringup slam.launch.py teleop:=true

Drive the robot; it builds a map. Save it with:

    ros2 topic pub --once /map_save std_msgs/msg/String "{data: ~/maps/lab}"

A thin forward to a2r_bot_slam's slam.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam = FindPackageShare('a2r_bot_slam')

    args = [
        DeclareLaunchArgument('lidar', default_value='rplidar'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('teleop', default_value='false'),
        DeclareLaunchArgument('planner', default_value='pure_pursuit'),
        DeclareLaunchArgument('local_planner', default_value='true'),
        DeclareLaunchArgument('arduino', default_value='false'),
        DeclareLaunchArgument('fusion', default_value='true'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [slam, 'launch', 'slam.launch.py'])),
        launch_arguments={
            'lidar': LaunchConfiguration('lidar'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'teleop': LaunchConfiguration('teleop'),
            'planner': LaunchConfiguration('planner'),
            'local_planner': LaunchConfiguration('local_planner'),
            'arduino': LaunchConfiguration('arduino'),
            'fusion': LaunchConfiguration('fusion'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
    )
    return LaunchDescription(args + [inner])
