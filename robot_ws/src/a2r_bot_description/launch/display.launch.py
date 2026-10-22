"""
display.launch.py — look at the robot, on its own, with sliders.
═══════════════════════════════════════════════════════════════════════════════
    ros2 launch a2r_bot_description display.launch.py
    ros2 launch a2r_bot_description display.launch.py styling:=false
    ros2 launch a2r_bot_description display.launch.py joint_source:=gui

Use this while editing the URDF. It runs no hardware, no planner and no
serial port — just the model, TF and RViz. With joint_source:=gui you get a
slider per joint, which is the fastest way to confirm the steering limits and
the Ackermann split look right.
═══════════════════════════════════════════════════════════════════════════════
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('a2r_bot_description')

    return LaunchDescription([
        DeclareLaunchArgument('styling',      default_value='true'),
        DeclareLaunchArgument('use_meshes',   default_value='false'),
        DeclareLaunchArgument('joint_source', default_value='gui'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([pkg, 'launch', 'description.launch.py'])]),
            launch_arguments={
                'styling':      LaunchConfiguration('styling'),
                'use_meshes':   LaunchConfiguration('use_meshes'),
                'joint_source': LaunchConfiguration('joint_source'),
            }.items()),

        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'model.rviz'])]),
    ])
