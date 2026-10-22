#!/usr/bin/env python3
"""
arduino_bringup.launch.py
=========================
Brings up the low-level hardware layer: the serial bridge that owns
/dev/ttyARDUINO, plus the health monitor.

    ros2 launch a2r_bot_arduino arduino_bringup.launch.py
    ros2 launch a2r_bot_arduino arduino_bringup.launch.py port:=/dev/ttyUSB0
    ros2 launch a2r_bot_arduino arduino_bringup.launch.py drive_topic:=/cmd_vel \\
                                                         normalised:=false

Nothing else in the workspace should open the serial port.  If you want to
drive the car from a node that speaks geometry_msgs/Twist in real units, point
drive_topic at it and set normalised:=false — the bridge will do the Ackermann
conversion for you.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('a2r_bot_arduino')
    default_params = PathJoinSubstitution([pkg, 'config', 'arduino.yaml'])

    args = [
        DeclareLaunchArgument(
            'port', default_value='/dev/ttyTHS1',
            description='Serial device, or "auto" to probe USB VID/PID.'),
        DeclareLaunchArgument(
            'baud', default_value='921600',
            description='Must match SERIAL_BAUD in firmware Config.h.'),
        DeclareLaunchArgument(
            'drive_topic', default_value='/local_planner/cmd_out',
            description='Topic the bridge reads drive commands from.'),
        DeclareLaunchArgument(
            'normalised', default_value='true',
            description='true: -1..1 throttle/steer. false: m/s and rad/s.'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Override the whole parameter file.'),
        DeclareLaunchArgument(
            'monitor', default_value='true',
            description='Also start the firmware health monitor.'),
        DeclareLaunchArgument(
            'log_level', default_value='info'),
    ]

    bridge = Node(
        package='a2r_bot_arduino',
        executable='serial_bridge_node',
        name='serial_bridge_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'port': LaunchConfiguration('port'),
                'baud': LaunchConfiguration('baud'),
                'drive_topic': LaunchConfiguration('drive_topic'),
                'input_is_normalised': LaunchConfiguration('normalised'),
            },
        ],
        arguments=['--ros-args', '--log-level',
                   LaunchConfiguration('log_level')],
        respawn=True,
        respawn_delay=2.0,
    )

    monitor = Node(
        package='a2r_bot_arduino',
        executable='firmware_monitor_node',
        name='firmware_monitor_node',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
        condition=IfCondition(LaunchConfiguration('monitor')),
    )

    status = Node(
        package='a2r_bot_arduino',
        executable='robot_status_node',
        name='robot_status_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'usb_device': LaunchConfiguration('port')}],
        respawn=True,
        respawn_delay=2.0,
    )
    return LaunchDescription(args + [bridge, monitor, status])
