"""
description.launch.py — robot_state_publisher + TF, nothing else.
═══════════════════════════════════════════════════════════════════════════════
Included by every other launch file in the workspace (follow, slam, pet), so
the model is described in exactly ONE place. Never start robot_state_publisher
anywhere else.

Args
────
  styling:=true|false      cosmetic bodywork (false is lighter on a Pi)
  use_meshes:=true|false   load meshes/body.dae instead of the primitives
  use_sim_time:=false
  joint_source:=cmd|gui|none
        cmd   wheel_state_node — joints follow the REAL commanded steering
        gui   joint_state_publisher_gui — sliders, for inspecting the model
        none  something else publishes /joint_states (e.g. Gazebo)
═══════════════════════════════════════════════════════════════════════════════
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('a2r_bot_description')
    xacro_file = PathJoinSubstitution([pkg, 'urdf', 'a2r_bot.urdf.xacro'])

    styling      = LaunchConfiguration('styling')
    use_meshes   = LaunchConfiguration('use_meshes')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # `xacro` is run at launch time, so editing the model needs no rebuild of
    # anything except a colcon install of this package's share/ directory.
    #
    # ParameterValue(..., value_type=str) is NOT optional on ROS 2 Humble.
    #
    # A Command() substitution hands launch a plain string, and launch then
    # tries to infer the parameter's type by parsing it as YAML. A URDF is
    # XML, and the first thing in it is `<?xml version="1.0"?>`, which YAML
    # rejects. The error you get names the parameter but not the cause:
    #
    #     Unable to parse the value of parameter robot_description as yaml.
    #
    # Declaring the type explicitly stops launch guessing. Humble is stricter
    # about this than Foxy was, which is why plenty of older tutorials omit it
    # and then fail on exactly this line.
    # description.yaml existed but was never loaded, so robot_state_publisher
    # ran at its default publish_frequency of 20 Hz. That cap throttled the
    # 30 Hz /joint_states to an irregular ~15 Hz, and RViz drew every link
    # under a movable joint (wheels, rims, spokes, steer knuckles) at that
    # older stamp while the chassis followed the 50 Hz EKF: parts lagging
    # behind the body and snapping forward.
    params_yaml = PathJoinSubstitution([pkg, 'config', 'description.yaml'])

    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' styling:=', styling,
            ' use_meshes:=', use_meshes,
        ]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument('styling',      default_value='true'),
        DeclareLaunchArgument('use_meshes',   default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('joint_source', default_value='cmd',
                              description='cmd | gui | none'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher', output='screen',
             parameters=[params_yaml,
                         {'robot_description': robot_description,
                          'use_sim_time': use_sim_time}]),

        Node(package='a2r_bot_description', executable='wheel_state_node',
             name='wheel_state_node', output='screen',
             parameters=[params_yaml, {'use_sim_time': use_sim_time}],
             condition=LaunchConfigurationEquals('joint_source', 'cmd')),

        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             name='joint_state_publisher_gui', output='screen',
             condition=LaunchConfigurationEquals('joint_source', 'gui')),
    ])
