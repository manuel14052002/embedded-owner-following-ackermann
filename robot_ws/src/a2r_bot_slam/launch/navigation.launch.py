#!/usr/bin/env python3
"""
navigation.launch.py — a2r_bot_slam
===================================
Autonomous navigation on a map.

    ros2 launch a2r_bot_slam navigation.launch.py
    ros2 launch a2r_bot_slam navigation.launch.py algorithm:=dijkstra
    ros2 launch a2r_bot_slam navigation.launch.py planner:=rrt
    ros2 launch a2r_bot_slam navigation.launch.py slam:=false map:=~/maps/lab.yaml

Give it a goal with RViz's "2D Goal Pose" button, or:

    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \\
      "{header: {frame_id: map}, pose: {position: {x: 3.0, y: 1.5}}}"

The two planner arguments are independent, and that is the point:

    algorithm:=  astar | dijkstra | rrt | rrt_star      the GLOBAL route
    planner:=    dwa | apf | teb | vfh | pure_pursuit |
                 fgm | lattice | rrt                    the LOCAL avoidance

The local one is the same set follow mode uses, running the same node, because
navigation_node publishes to /cmd_vel exactly as the Jetson's tracker does.
Both can be changed while driving:

    ros2 param set /global_planner_node algorithm rrt_star
    ros2 param set /local_planner_bridge planner fgm

Running with slam:=true (the default) keeps mapping while navigating, so the
map improves as you drive and the robot localises against what it has just
seen. Set slam:=false and pass a saved map when you want the map frozen —
which is what you want once a map is good, because then a moving person
walking past cannot carve a corridor through your wall.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_share = FindPackageShare('a2r_bot_slam')
    desc_share = FindPackageShare('a2r_bot_description')
    follow_share = FindPackageShare('a2r_bot_follow')

    params = PathJoinSubstitution([slam_share, 'config', 'slam.yaml'])
    rviz_cfg = PathJoinSubstitution([desc_share, 'rviz', 'slam.rviz'])

    args = [
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('lidar', default_value='rplidar'),
        DeclareLaunchArgument('slam', default_value='true',
                              description='Keep mapping while navigating.'),
        DeclareLaunchArgument('map', default_value='',
                              description='Saved map .yaml, when slam:=false.'),
        DeclareLaunchArgument(
            'algorithm', default_value='astar',
            description='GLOBAL planner: astar | dijkstra | rrt | rrt_star'),
        DeclareLaunchArgument(
            'planner', default_value='pure_pursuit',
            description='LOCAL planner: dwa | apf | teb | vfh | '
                        'pure_pursuit | fgm | lattice | rrt'),
        DeclareLaunchArgument('unknown_is_free', default_value='false'),
        DeclareLaunchArgument('params_file', default_value=params),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    pf = LaunchConfiguration('params_file')
    log = ['--ros-args', '--log-level', LaunchConfiguration('log_level')]

    # Mapping stack, minus its RViz and minus its local planner, since we
    # start our own below with the navigation planner argument.
    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [slam_share, 'launch', 'slam.launch.py'])),
        launch_arguments={
            'use_rviz': 'false',
            'local_planner': 'false',
            'lidar': LaunchConfiguration('lidar'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('slam')),
    )

    # Frozen map instead: map_server publishes the saved grid and we localise
    # against it with the scan matcher in localisation-only mode.
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': LaunchConfiguration('map')}],
        condition=UnlessCondition(LaunchConfiguration('slam')))

    costmap = Node(
        package='a2r_bot_slam', executable='global_costmap_node',
        name='global_costmap_node', output='screen', parameters=[pf],
        arguments=log,
        condition=UnlessCondition(LaunchConfiguration('slam')))

    global_planner = Node(
        package='a2r_bot_slam', executable='global_planner_node',
        name='global_planner_node', output='screen', emulate_tty=True,
        parameters=[pf, {
            'algorithm': LaunchConfiguration('algorithm'),
            'unknown_is_free': LaunchConfiguration('unknown_is_free'),
        }],
        arguments=log)

    navigation = Node(
        package='a2r_bot_slam', executable='navigation_node',
        name='navigation_node', output='screen', emulate_tty=True,
        parameters=[pf], arguments=log)

    # The same node, the same config file and the same eight planners as
    # follow mode. Only its input changed: navigation_node is publishing
    # /cmd_vel instead of the Jetson.
    local_planner = Node(
        package='a2r_bot_follow', executable='smart_planner_node',
        name='local_planner_bridge', output='screen', emulate_tty=True,
        parameters=[
            PathJoinSubstitution([follow_share, 'config', 'params.yaml']),
            {'planner': LaunchConfiguration('planner'),
             'output_mode': 'topic',
             'scan_topic': '/scan'},
        ],
        arguments=log)

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_cfg], output='screen',
                condition=IfCondition(LaunchConfiguration('use_rviz')))

    return LaunchDescription(args + [
        mapping, map_server, costmap,
        global_planner, navigation, local_planner, rviz,
    ])
