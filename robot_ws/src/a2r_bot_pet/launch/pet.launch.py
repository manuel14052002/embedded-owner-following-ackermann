#!/usr/bin/env python3
"""
pet.launch.py — a2r_bot_pet
===========================
The animated face, and the bridge that decides what it shows.

    ros2 launch a2r_bot_pet pet.launch.py
    ros2 launch a2r_bot_pet pet.launch.py fullscreen:=false     # on a laptop
    ros2 launch a2r_bot_pet pet.launch.py auto_sequence:=true   # demo, no Jetson

Two nodes:

    state_bridge_node   /pet/state (from the Jetson)  ->  /pet/face
    face_node           /pet/face                     ->  pixels

They are split so the face can be driven directly for testing without
inventing fake Jetson states:

    ros2 topic pub --once /pet/expression std_msgs/msg/String "{data: happy}"

and so that the mapping can be reloaded without restarting pygame, which on a
framebuffer LCD is slow and occasionally leaves the console in a strange
state.

There is no state machine in this package. The Jetson owns that; these nodes
only translate and render.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('a2r_bot_pet')
    params = PathJoinSubstitution([pkg, 'config', 'pet.yaml'])

    args = [
        DeclareLaunchArgument('params_file', default_value=params),
        DeclareLaunchArgument('fullscreen', default_value='true'),
        DeclareLaunchArgument('auto_sequence', default_value='false'),
        DeclareLaunchArgument('fps', default_value='60'),
        DeclareLaunchArgument('start_expression', default_value='idle'),
        DeclareLaunchArgument('bridge', default_value='true',
                              description='Run the state->expression bridge.'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    pf = LaunchConfiguration('params_file')
    log = ['--ros-args', '--log-level', LaunchConfiguration('log_level')]

    bridge = Node(
        package='a2r_bot_pet', executable='state_bridge_node',
        name='state_bridge_node', output='screen', emulate_tty=True,
        parameters=[pf], arguments=log,
        condition=IfCondition(LaunchConfiguration('bridge')))

    face = Node(
        package='a2r_bot_pet', executable='face_node',
        name='face_node', output='screen', emulate_tty=True,
        parameters=[pf, {
            'fullscreen': LaunchConfiguration('fullscreen'),
            'auto_sequence': LaunchConfiguration('auto_sequence'),
            'fps': LaunchConfiguration('fps'),
            'start_expression': LaunchConfiguration('start_expression'),
        }],
        arguments=log,
        # The face is cosmetic. If it dies, the robot keeps working, so it
        # must never take the launch file down with it.
        respawn=True, respawn_delay=3.0)

    return LaunchDescription(args + [bridge, face])
