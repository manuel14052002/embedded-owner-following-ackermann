"""RealSense (aligned depth) + FollowMe node.

    ros2 launch followme_ros followme.launch.py followme_root:=$HOME/owner_tracker \
        followme_config:=$HOME/owner_tracker/config/followme_a2rbot.yaml enable_navigation:=false

Start Nav2 / localisation separately (they must provide TF map -> camera optical frame).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("followme_root", default_value="", description="owner_tracker repo (if not pip-installed)"),
        DeclareLaunchArgument("followme_config", default_value=""),
        DeclareLaunchArgument("target_model", default_value="models/target.npz"),
        DeclareLaunchArgument("enable_navigation", default_value="false"),
        DeclareLaunchArgument("control_mode", default_value="cmd_vel"),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("global_frame", default_value="map"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        SetEnvironmentVariable("FOLLOWME_ROOT", arg("followme_root")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])),
            launch_arguments={"align_depth.enable": "true", "enable_color": "true", "enable_depth": "true",
                              "rgb_camera.color_profile": "640,480,30",
                              "depth_module.depth_profile": "640,480,30"}.items(),
            condition=IfCondition(arg("start_camera"))),
        Node(
            package="followme_ros", executable="followme_node", name="followme", output="screen",
            parameters=[PathJoinSubstitution([FindPackageShare("followme_ros"), "config", "followme_ros.yaml"]),
                        {"config_path": arg("followme_config"), "target_model_path": arg("target_model"),
                         "enable_navigation": arg("enable_navigation"), "control_mode": arg("control_mode"),
                         "global_frame": arg("global_frame"),
                         "base_frame": arg("base_frame")}]),
    ])
