#!/usr/bin/env python3
"""
slam.launch.py — a2r_bot_slam
=============================
Mapping mode.  Drive the robot around; it builds a map.

    ros2 launch a2r_bot_slam slam.launch.py
    ros2 launch a2r_bot_slam slam.launch.py lidar:=rplidar
    ros2 launch a2r_bot_slam slam.launch.py teleop:=true

What starts
-----------
    robot_state_publisher   the URDF
    lidar driver            rplidar A2M12 (default) | ydlidar | none
    a2r_bot_arduino         serial bridge — the only owner of /dev/ttyARDUINO
    a2r_bot_sensor_fusion   IMU filter, EKF pose, lidar+sonar scan fusion
    slam_node               occupancy mapping + scan matching
    global_costmap_node     inflates /map by the robot radius
    map_saver_node          waits for /map_save
    rviz2                   with the SLAM config

Note what is NOT here: no global planner and no navigation.  This launch file
only builds a map.  Once you have one, save it and run navigation.launch.py.

Saving the map
--------------
    ros2 topic pub --once /map_save std_msgs/msg/String "{data: ~/maps/lab}"

which writes ~/maps/lab.pgm and ~/maps/lab.yaml.

Driving it
----------
Nothing in this file moves the robot.  Either run your own teleop and publish
to /cmd_vel, or pass teleop:=true to start teleop_twist_keyboard — note that
its output still goes through smart_planner_node, so the local planner will
refuse to let you drive into a wall even under manual control.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_share = FindPackageShare('a2r_bot_slam')
    desc_share = FindPackageShare('a2r_bot_description')
    fusion_share = FindPackageShare('a2r_bot_sensor_fusion')
    arduino_share = FindPackageShare('a2r_bot_arduino')
    follow_share = FindPackageShare('a2r_bot_follow')

    params = PathJoinSubstitution([slam_share, 'config', 'slam.yaml'])
    rviz_cfg = PathJoinSubstitution([desc_share, 'rviz', 'slam.rviz'])

    args = [
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('lidar', default_value='rplidar',
                              description='rplidar (A2M12, default) | ydlidar | none'),
        DeclareLaunchArgument('arduino', default_value='false'),
        DeclareLaunchArgument('fusion', default_value='true'),
        DeclareLaunchArgument('local_planner', default_value='true',
                              description='Run smart_planner_node so manual '
                                          'driving is still collision-aware.'),
        DeclareLaunchArgument('planner', default_value='pure_pursuit'),
        DeclareLaunchArgument('teleop', default_value='false'),
        DeclareLaunchArgument('params_file', default_value=params),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    pf = LaunchConfiguration('params_file')
    log = ['--ros-args', '--log-level', LaunchConfiguration('log_level')]

    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [desc_share, 'launch', 'description.launch.py'])),
        launch_arguments={'joint_source': 'cmd', 'styling': 'true'}.items(),
    )

    arduino = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [arduino_share, 'launch', 'arduino_bringup.launch.py'])),
        condition=IfCondition(LaunchConfiguration('arduino')),
    )

    fusion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [fusion_share, 'launch', 'sensor_fusion.launch.py'])),
        launch_arguments={'publish_tf': 'true'}.items(),
        condition=IfCondition(LaunchConfiguration('fusion')),
    )

    ydlidar = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_node', output='screen',
        parameters=[{'port': '/dev/ttyLIDAR', 'frame_id': 'lidar_link',
                     'baudrate': 115200, 'lidar_type': 1, 'device_type': 0,
                     'sample_rate': 3, 'frequency': 7.0,
                     'angle_min': -180.0, 'angle_max': 180.0,
                     'range_min': 0.10, 'range_max': 8.0,
                     'inverted': False, 'auto_reconnect': True,
                     'isSingleChannel': True}],
        condition=LaunchConfigurationEquals('lidar', 'ydlidar'))

    rplidar = Node(
        package='sllidar_ros2', executable='sllidar_node',
        name='rplidar_node', output='screen',
        parameters=[{
            'serial_port': '/dev/ttyLIDAR',
            # A2M12 is 256000, NOT the A2M8's 115200. At 115200 the driver
            # opens the port, reports no error, and never publishes a scan.
            'serial_baudrate': 256000,
            'frame_id': 'lidar_link',
            'angle_compensate': True,
            'scan_mode': 'Boost',      # ~16k samples/s vs Standard's ~4k
            'inverted': False,
        }],
        condition=LaunchConfigurationEquals('lidar', 'rplidar'))

    slam = Node(package='a2r_bot_slam', executable='slam_node',
                name='slam_node', output='screen', emulate_tty=True,
                parameters=[pf], arguments=log)

    costmap = Node(package='a2r_bot_slam', executable='global_costmap_node',
                   name='global_costmap_node', output='screen',
                   parameters=[pf], arguments=log)

    saver = Node(package='a2r_bot_slam', executable='map_saver_node',
                 name='map_saver_node', output='screen', parameters=[pf])

    # The local planner runs during mapping too, so that whatever is driving
    # — teleop, a joystick, a script — still cannot put the car into a wall.
    local = Node(
        package='a2r_bot_follow', executable='smart_planner_node',
        name='local_planner_bridge', output='screen', emulate_tty=True,
        parameters=[
            PathJoinSubstitution([follow_share, 'config', 'params.yaml']),
            {'planner': LaunchConfiguration('planner'),
             'output_mode': 'topic',
             'scan_topic': '/scan'},
        ],
        arguments=log,
        condition=IfCondition(LaunchConfiguration('local_planner')))

    teleop = Node(
        package='teleop_twist_keyboard', executable='teleop_twist_keyboard',
        name='teleop', output='screen', prefix='xterm -e',
        condition=IfCondition(LaunchConfiguration('teleop')))

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_cfg], output='screen',
                condition=IfCondition(LaunchConfiguration('use_rviz')))

    # ---- lidar driver availability (fix_sllidar_launch.py) ---
    # The robot's lidar is the RPLIDAR A2M12 on sllidar_ros2. Humble resolves
    # a Node's package while the launch runs, so a driver that is not
    # installed must not be in the description at all, even behind a
    # condition. Missing drivers become a log line instead of a crash.
    from launch.actions import LogInfo
    from ament_index_python.packages import get_package_prefix, PackageNotFoundError

    def _have(pkg):
        try:
            get_package_prefix(pkg)
            return True
        except PackageNotFoundError:
            return False

    if not _have('ydlidar_ros2_driver'):
        ydlidar = LogInfo(
            msg='lidar:=ydlidar ignored - ydlidar_ros2_driver is not installed '
                '(this robot uses the RPLIDAR A2M12: lidar:=rplidar)',
            condition=LaunchConfigurationEquals('lidar', 'ydlidar'))
    if not _have('sllidar_ros2'):
        rplidar = LogInfo(
            msg='!! sllidar_ros2 (RPLIDAR A2M12 driver) is NOT installed - there will be '
                'NO /scan. Fix: cd <workspace>/src && git clone '
                'https://github.com/Slamtec/sllidar_ros2 && cd .. && '
                'colcon build --packages-select sllidar_ros2',
            condition=LaunchConfigurationEquals('lidar', 'rplidar'))

    return LaunchDescription(args + [
        description, arduino, fusion, ydlidar, rplidar,
        slam, costmap, saver, local, teleop, rviz,
    ])
