#!/usr/bin/env python3
"""
pet.launch.py — a2r_bot_bringup
===============================
The animated face on its own.

    ros2 launch a2r_bot_bringup pet.launch.py
    ros2 launch a2r_bot_bringup pet.launch.py fullscreen:=false

Normally you would start the face as part of follow mode:

    ros2 launch a2r_bot_bringup follow.launch.py pet:=true

This file is for working on the face by itself — checking the LCD, testing
expressions, or demoing it with no Jetson attached (auto_sequence:=true).

A thin forward to a2r_bot_pet's pet.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pet = FindPackageShare('a2r_bot_pet')

    args = [
        DeclareLaunchArgument('fullscreen', default_value='true'),
        DeclareLaunchArgument('auto_sequence', default_value='false'),
        DeclareLaunchArgument('fps', default_value='60'),
        DeclareLaunchArgument('start_expression', default_value='idle'),
        DeclareLaunchArgument('bridge', default_value='true'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [pet, 'launch', 'pet.launch.py'])),
        launch_arguments={
            'fullscreen': LaunchConfiguration('fullscreen'),
            'auto_sequence': LaunchConfiguration('auto_sequence'),
            'fps': LaunchConfiguration('fps'),
            'start_expression': LaunchConfiguration('start_expression'),
            'bridge': LaunchConfiguration('bridge'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
    )
    return LaunchDescription(args + [inner])
