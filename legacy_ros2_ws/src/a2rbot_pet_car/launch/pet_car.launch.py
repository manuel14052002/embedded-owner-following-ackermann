#!/usr/bin/env python3
"""
pet_car.launch.py — starts camera, tracker, and serial bridge.

Usage:
  ros2 launch a2rbot_pet_car pet_car.launch.py
  ros2 launch a2rbot_pet_car pet_car.launch.py dry_run:=true      # no motors
  ros2 launch a2rbot_pet_car pet_car.launch.py max_linear:=0.3    # slower
  ros2 launch a2rbot_pet_car pet_car.launch.py port:=/dev/ttyACM0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    dry_run    = LaunchConfiguration("dry_run")
    port       = LaunchConfiguration("port")
    max_linear = LaunchConfiguration("max_linear")
    max_speed_mps = LaunchConfiguration("max_speed_mps")
    wheelbase  = LaunchConfiguration("wheelbase_m")
    model_path = LaunchConfiguration("model_path")

    return LaunchDescription([
        # ── Launch arguments (safe defaults) ───────────────────────────
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="Bridge logs frames instead of writing serial"),
        DeclareLaunchArgument("port", default_value="auto",
                              description="ESP32 serial port ('auto' = detect)"),
        DeclareLaunchArgument("max_linear", default_value="0.5",
                              description="Tracker max linear m/s (START LOW)"),
        DeclareLaunchArgument("max_speed_mps", default_value="1.2",
                              description="Robot physical speed at throttle v=1.0"),
        DeclareLaunchArgument("wheelbase_m", default_value="0.30",
                              description="Front-to-rear axle distance (m)"),
        DeclareLaunchArgument("model_path", default_value="/home/a2r/yolov8n.engine",
                              description="YOLO model — use TensorRT .engine on Jetson"),

        # ── Camera: DISABLED — voice system owns the D435, we subscribe ──
        # (camera_node removed to avoid D435 device conflict)

        DeclareLaunchArgument("bridge", default_value="false",
                              description="OLD bridge (old firmware). Keep false: a2r_bot_arduino owns the UART"),

        # ── Tracker ────────────────────────────────────────────────────
        Node(
            package="a2rbot_pet_car", executable="tracker_node",
            name="tracker_node", output="screen",
            parameters=[{
                "model_path": model_path,
                "target_dist_m": 1.2,
                "max_linear": max_linear,
                "max_angular": 0.8,
                "confidence": 0.45,
                "lost_timeout_s": 1.0,
                "auto_start": False,
            }],
        ),

        # ── Serial bridge ──────────────────────────────────────────────
        Node(
            package="a2rbot_pet_car", executable="serial_bridge_node",
            name="serial_bridge_node", output="screen",
            condition=IfCondition(LaunchConfiguration("bridge")),
            parameters=[{
                "port": port,
                "baud": 115200,
                "dry_run": dry_run,
                "max_speed_mps": max_speed_mps,
                "max_speed_cmd": 0.7,        # 70% throttle cap
                "wheelbase_m": wheelbase,
                "max_steer_deg": 30.0,
                "cmd_rate_hz": 20.0,
                "cmd_timeout_s": 0.3,
            }],
        ),
    ])
