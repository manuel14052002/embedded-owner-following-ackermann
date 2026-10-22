#!/usr/bin/env python3
"""
follow.launch.py — a2r_bot_follow
=================================
Owner-following mode.  This is the launch file the original workspace was
built around; it keeps the same name and the same job.  What changed is that
the pieces it starts now live in their own packages.

    ros2 launch a2r_bot_follow follow.launch.py
    ros2 launch a2r_bot_follow follow.launch.py planner:=rrt
    ros2 launch a2r_bot_follow follow.launch.py lidar:=rplidar use_rviz:=false
    ros2 launch a2r_bot_follow follow.launch.py fusion:=false scan_topic:=/scan

What starts
-----------
    robot_state_publisher   the URDF, from a2r_bot_description
    lidar driver            rplidar A2M12 (default) | ydlidar | none
    a2r_bot_arduino         serial bridge — the ONLY owner of /dev/ttyARDUINO
    a2r_bot_sensor_fusion   IMU filter, EKF pose, lidar+sonar scan fusion
    smart_planner_node      the eight local planners
    a2r_bot_pet             the animated face, if pet:=true
    rviz2                   if use_rviz:=true

The data path
-------------
    Jetson --/cmd_vel--> smart_planner_node --/local_planner/cmd_out-->
    serial_bridge_node --serial--> Arduino --> ESC (ch1) + servo (ch0)

    lidar --/scan----\
                      >-- scan_fusion --/scan/fused--> smart_planner_node
    sonars ----------/

    IMU --> imu_filter --/imu/data--> ekf_state --/odometry/filtered-->

odometry_node from the original package is NOT started by default.  Its job —
dead reckoning from IMU yaw and commanded velocity — is now done by
ekf_state_node, which also folds in the accelerometer, a zero-velocity update
and scan matching.  Pass legacy_odom:=true to run the old one instead; it is
worth doing once, with both on screen in RViz, to see what the EKF is buying.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('a2r_bot_follow')
    params = os.path.join(pkg, 'config', 'params.yaml')

    desc_share = FindPackageShare('a2r_bot_description')
    fusion_share = FindPackageShare('a2r_bot_sensor_fusion')
    arduino_share = FindPackageShare('a2r_bot_arduino')
    pet_share = FindPackageShare('a2r_bot_pet')

    use_rviz = LaunchConfiguration('use_rviz')
    planner = LaunchConfiguration('planner')
    fusion = LaunchConfiguration('fusion')
    arduino = LaunchConfiguration('arduino')
    pet = LaunchConfiguration('pet')
    scan_topic = LaunchConfiguration('scan_topic')
    legacy_odom = LaunchConfiguration('legacy_odom')

    args = [
        DeclareLaunchArgument('mode', default_value='indoor',                      # a2r_step3
                              description='indoor | outdoor (outdoor = also track the owner by GPS)'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='a2r_merge: RViz on the Pi costs most of a core - run it on a laptop'),
        DeclareLaunchArgument(
            'lidar', default_value='rplidar',
            description='rplidar (A2M12, default) | ydlidar | none'),
        DeclareLaunchArgument(
            'lidar_mode', default_value='Boost',
            description='RPLIDAR A2M12 scan mode: Sensitivity | Boost (16 kHz) | Standard'),
        DeclareLaunchArgument(
            'lidar_hz', default_value='7.0',
            description='Lidar rotation rate, 5-15 Hz (needs the A2R-patched sllidar_ros2). '
                        'Lower = finer angular resolution, slower updates.'),
        DeclareLaunchArgument(
            'planner', default_value='dwa',
            description='dwa | apf | teb | vfh | pure_pursuit | fgm | '
                        'lattice | rrt'),
        DeclareLaunchArgument(
            'fusion', default_value='true',
            description='Run the IMU filter, EKF and scan fusion.'),
        DeclareLaunchArgument(
            'arduino', default_value='false',
            description='Run the serial bridge. Set false only if it is '
                        'already running elsewhere — never run two.'),
        DeclareLaunchArgument('pet', default_value='false',
                              description='Also start the animated face.'),
        DeclareLaunchArgument(
            'scan_topic', default_value='/scan/planner',
            description='Set to /scan to bypass sonar fusion.'),
        DeclareLaunchArgument(
            'legacy_odom', default_value='false',
            description='Run the original odometry_node instead of the EKF.'),
        DeclareLaunchArgument('use_mag', default_value='true'),
        DeclareLaunchArgument('port', default_value='/dev/ttyARDUINO'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('gps', default_value='false',            # a2r_v5
                              description='false = no gps_owner node (indoors, saves CPU)'),
        DeclareLaunchArgument(
            'follow_pipeline', default_value='true',
            description='owner_fusion + follow_manager + twist_mux + collision_monitor. '
                        'false = the planner drives the bridge directly (old wiring).'),
    ]

    # ---- Robot model -----------------------------------------------------
    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [desc_share, 'launch', 'description.launch.py'])),
        launch_arguments={
            'joint_source': 'cmd',      # animate the wheels from the command
            'styling': 'true',
        }.items(),
    )

    # ---- Lidar -----------------------------------------------------------
    ydlidar_node = Node(
        package='ydlidar_ros2_driver',
        executable='ydlidar_ros2_driver_node',
        name='ydlidar_node',
        parameters=[{
            'port': '/dev/ttyLIDAR', 'frame_id': 'lidar_link',
            'baudrate': 115200, 'lidar_type': 1, 'device_type': 0,
            'sample_rate': 3, 'frequency': 7.0,
            'angle_min': -180.0, 'angle_max': 180.0,
            'range_min': 0.10, 'range_max': 8.0,
            'inverted': False, 'auto_reconnect': True,
            'isSingleChannel': True,
        }],
        condition=LaunchConfigurationEquals('lidar', 'ydlidar'),
        output='screen')

    rplidar_node = Node(

        prefix='nice -n -5',  # a2r_v11
        package='sllidar_ros2', executable='sllidar_node', name='rplidar_node',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyLIDAR',
            # ---- A2M12: 256000, NOT 115200 -----------------------------
            # The older A2M8 runs at 115200 and that number is all over the
            # internet. The A2M12 does not: it enumerates at 256000 and at
            # 115200 you get a driver that opens the port, reports no error,
            # and never publishes a single scan. Silent, and it looks exactly
            # like a dead lidar.
            'serial_baudrate': 256000,
            'frame_id': 'lidar_link',
            'angle_compensate': True,
            # ---- v2.8: higher resolution -------------------------------
            # Angular resolution = 360 deg x rotation rate / sample rate.
            #   Standard @ 10 Hz : ~1 deg/beam  - a 2.5 cm chair leg at
            #                      1.5 m is ~1 beam, so scan_leg_fill never
            #                      saw it and the planner drove between legs
            #   Sensitivity (16 kHz) @ 7 Hz : ~0.16 deg, ~2300 points/rev -
            #                      the same leg is 5-6 beams, visible to ~4 m
            # 'Sensitivity' is Slamtec's own default for the A2M12; 'Boost'
            # is the other 16 kHz mode. A mode the unit does not have makes
            # the driver print the supported list.
            'scan_mode': LaunchConfiguration('lidar_mode'),
            # Slower spin = more points per turn. scan_frequency only sets
            # the motor speed with the A2R-patched driver (motor_control);
            # the stock driver ignores motor_control and stays at 10 Hz.
            # float() so lidar_hz:=7 works too (the driver declares a float)
            'scan_frequency': PythonExpression(
                ["float('", LaunchConfiguration('lidar_hz'), "')"]),
            'motor_control': True,
            'inverted': False,
        }],
        condition=LaunchConfigurationEquals('lidar', 'rplidar'),
    )

    # ---- Arduino serial bridge -------------------------------------------
    # The ONLY owner of /dev/ttyARDUINO, and the only source of
    # /ultrasonic/*, /imu/data_raw and /battery. Without it the sonar never
    # reaches scan_fusion and the planner drives on lidar alone.
    arduino_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [arduino_share, 'launch', 'arduino_bringup.launch.py'])),
        launch_arguments={
            'port': LaunchConfiguration('port'),
            'drive_topic': '/local_planner/cmd_out',
            'normalised': 'true',
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
        condition=IfCondition(arduino),
    )

    # ---- Fusion ----------------------------------------------------------
    sensor_fusion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [fusion_share, 'launch', 'sensor_fusion.launch.py'])),
        launch_arguments={
            'use_mag': LaunchConfiguration('use_mag'),
            'publish_tf': 'true',
            'command_topic': '/local_planner/cmd_out',
        }.items(),
        condition=IfCondition(fusion),
    )

    # ---- The original dead-reckoning node, on request --------------------
    legacy_odom_node = Node(
        package='a2r_bot_follow', executable='odometry_node',
        name='odometry_node', parameters=[params], output='screen',
        condition=IfCondition(legacy_odom))

    # ---- The planner -----------------------------------------------------
    planner_node = Node(
        prefix='nice -n -5',  # a2r_v11
        package='a2r_bot_follow',
        executable='smart_planner_node',
        name='local_planner_bridge',
        parameters=[
            params,
            {
                'planner': planner,
                'scan_topic': scan_topic,
                'output_mode': 'topic',
                # pipeline on : planner -> /cmd_vel_nav -> twist_mux -> collision_monitor
                # pipeline off: planner -> /local_planner/cmd_out -> bridge (old wiring)
                'cmd_out_topic': PythonExpression(["'/cmd_vel_nav' if '",
                    LaunchConfiguration('follow_pipeline'),
                    "'.lower() == 'true' else '/local_planner/cmd_out'"]),
            },
        ],
        arguments=['--ros-args', '--log-level',
                   LaunchConfiguration('log_level')],
        output='screen',
        emulate_tty=True)

    # ---- Face ------------------------------------------------------------
    # v2.8: a second `arduino_bringup = IncludeLaunchDescription(...)` used to
    # sit here. It replaced the one above and dropped drive_topic /
    # normalised, so arduino:=true started the bridge on the WRONG command
    # topic (it bypassed collision_monitor). Removed.

    pet_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [pet_share, 'launch', 'pet.launch.py'])),
        condition=IfCondition(pet),
    )

    # ---- Follow pipeline (fix_follow_pipeline.py) ------------------------
    #  owner_fusion   camera owner point + lidar -> tracked owner (base_link)
    #  follow_manager tracked owner -> goal (stand-off behind the owner)
    #  twist_mux      teleop > escape > navigation, e-stop lock on top
    #  collision_monitor  raw-lidar last line of defence -> serial bridge
    follow_pipeline = LaunchConfiguration('follow_pipeline')
    twist_mux_yaml = os.path.join(pkg, 'config', 'twist_mux.yaml')
    owner_fusion_node = Node(
        package='a2r_bot_follow', executable='owner_fusion_node',
        name='owner_fusion', parameters=[params], output='screen',
        condition=IfCondition(follow_pipeline))
    follow_manager_node = Node(
        package='a2r_bot_follow', executable='follow_manager_node',
        name='follow_manager', parameters=[params, {'mode': LaunchConfiguration('mode')}], output='screen',
        condition=IfCondition(follow_pipeline))
    # Built-in priority mux (cmd_mux_node) - the apt twist_mux died at launch
    # on this robot. Same topics / priorities: see params.yaml -> cmd_mux.
    twist_mux_node = Node(
        prefix='nice -n -5',  # a2r_v11
        package='a2r_bot_follow', executable='cmd_mux_node', name='cmd_mux',
        parameters=[params],
        output='screen', condition=IfCondition(follow_pipeline))
    collision_monitor_node = Node(
        prefix='nice -n -5',  # a2r_v11
        package='a2r_bot_follow', executable='collision_monitor_node',
        name='collision_monitor', parameters=[params], output='screen',
        condition=IfCondition(follow_pipeline))

    # ---- Chairs + owner for the planner (a2r_pi_follow_obstacles.sh) ----
    #  /scan -> scan_leg_fill -> /scan/planner (owner removed) -> planner
    #                         -> /scan/merged  (owner kept)    -> collision_monitor
    leg_fill_node = Node(
        package='a2r_bot_follow', executable='scan_leg_fill_node',
        name='scan_leg_fill', parameters=[params], output='screen')

    # ---- a2r_step3: GPS owner tracking (outdoor mode; idles indoors) -------
    gps_owner_node = Node(
        package='a2r_bot_follow', executable='gps_owner_node', name='gps_owner',  # a2r_v5: gps:=false skips it
        parameters=[params, {'mode': LaunchConfiguration('mode')}], output='screen',
        condition=IfCondition(PythonExpression(["'", follow_pipeline, "'.lower() == 'true' and '",
                                                LaunchConfiguration('gps'), "'.lower() == 'true'"])))

    # ---- RViz ------------------------------------------------------------
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', PathJoinSubstitution(
            [desc_share, 'rviz', 'follow.rviz'])],
        condition=IfCondition(use_rviz), output='screen')

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
        ydlidar_node = LogInfo(
            msg='lidar:=ydlidar ignored - ydlidar_ros2_driver is not installed '
                '(this robot uses the RPLIDAR A2M12: lidar:=rplidar)',
            condition=LaunchConfigurationEquals('lidar', 'ydlidar'))
    if not _have('sllidar_ros2'):
        rplidar_node = LogInfo(
            msg='!! sllidar_ros2 (RPLIDAR A2M12 driver) is NOT installed - there will be '
                'NO /scan. Fix: cd <workspace>/src && git clone '
                'https://github.com/Slamtec/sllidar_ros2 && cd .. && '
                'colcon build --packages-select sllidar_ros2',
            condition=LaunchConfigurationEquals('lidar', 'rplidar'))

    return LaunchDescription(args + [
        description,
        ydlidar_node, rplidar_node,
        arduino_bringup,
        sensor_fusion,
        legacy_odom_node,
        planner_node,
        owner_fusion_node, follow_manager_node,
        twist_mux_node, collision_monitor_node,
        leg_fill_node,
        gps_owner_node,
        pet_launch,
        rviz,
    ])
