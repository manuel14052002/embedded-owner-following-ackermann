#!/usr/bin/env python3
"""
follow.launch.py — a2r_bot_bringup
==================================
Owner-following mode, from the bringup package.

    ros2 launch a2r_bot_bringup follow.launch.py
    ros2 launch a2r_bot_bringup follow.launch.py planner:=rrt pet:=true

This is a thin forward to a2r_bot_follow's own follow.launch.py, which is the
file the original workspace was built around and which keeps its name and its
job. It exists so that every mode can be started the same way:

    ros2 launch a2r_bot_bringup follow.launch.py
    ros2 launch a2r_bot_bringup slam.launch.py
    ros2 launch a2r_bot_bringup navigation.launch.py
    ros2 launch a2r_bot_bringup pet.launch.py

Use either. `ros2 launch a2r_bot_follow follow.launch.py` does the same thing.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    follow = FindPackageShare('a2r_bot_follow')

    args = [
        DeclareLaunchArgument('planner', default_value='pure_pursuit',
                              description='dwa | apf | teb | vfh | '
                                          'pure_pursuit | fgm | lattice | rrt'),
        DeclareLaunchArgument('lidar', default_value='rplidar'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('mode', default_value='indoor'),          # a2r_step3: indoor | outdoor
        DeclareLaunchArgument('pet', default_value='false'),
        DeclareLaunchArgument('fusion', default_value='true'),
        DeclareLaunchArgument('arduino', default_value='false'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [follow, 'launch', 'follow.launch.py'])),
        launch_arguments={
            'planner': LaunchConfiguration('planner'),
            'lidar': LaunchConfiguration('lidar'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'mode': LaunchConfiguration('mode'),
            'pet': LaunchConfiguration('pet'),
            'fusion': LaunchConfiguration('fusion'),
            'arduino': LaunchConfiguration('arduino'),
            'scan_topic': LaunchConfiguration('scan_topic'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
    )
    return LaunchDescription(args + [inner])
