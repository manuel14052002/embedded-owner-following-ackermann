#!/usr/bin/env python3
"""
navigation.launch.py — a2r_bot_bringup
======================================
Autonomous navigation on a map, from the bringup package.

    ros2 launch a2r_bot_bringup navigation.launch.py
    ros2 launch a2r_bot_bringup navigation.launch.py algorithm:=rrt_star planner:=fgm

Two independent planner choices, which is the point:

    algorithm:=  astar | dijkstra | rrt | rrt_star        the GLOBAL route
    planner:=    dwa | apf | teb | vfh | pure_pursuit |
                 fgm | lattice | rrt                      the LOCAL avoidance

The local set is the same eight follow mode uses, running the same node,
because navigation_node publishes /cmd_vel exactly as the Jetson's tracker
does. Both can be changed while driving.

A thin forward to a2r_bot_slam's navigation.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam = FindPackageShare('a2r_bot_slam')

    args = [
        DeclareLaunchArgument('algorithm', default_value='astar',
                              description='GLOBAL: astar | dijkstra | rrt | '
                                          'rrt_star'),
        DeclareLaunchArgument('planner', default_value='pure_pursuit',
                              description='LOCAL: dwa | apf | teb | vfh | '
                                          'pure_pursuit | fgm | lattice | rrt'),
        DeclareLaunchArgument('lidar', default_value='rplidar'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('slam', default_value='true',
                              description='Keep mapping while navigating.'),
        DeclareLaunchArgument('map', default_value='',
                              description='Saved map .yaml, when slam:=false.'),
        DeclareLaunchArgument('unknown_is_free', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [slam, 'launch', 'navigation.launch.py'])),
        launch_arguments={
            'algorithm': LaunchConfiguration('algorithm'),
            'planner': LaunchConfiguration('planner'),
            'lidar': LaunchConfiguration('lidar'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'slam': LaunchConfiguration('slam'),
            'map': LaunchConfiguration('map'),
            'unknown_is_free': LaunchConfiguration('unknown_is_free'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
    )
    return LaunchDescription(args + [inner])
