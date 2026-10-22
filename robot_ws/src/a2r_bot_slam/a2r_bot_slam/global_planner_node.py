#!/usr/bin/env python3
"""
global_planner_node.py
======================
Wraps the planners in a2r_bot_slam.planners behind a ROS interface.

Four algorithms, selected by the `algorithm` parameter exactly as the local
planners are selected by `planner` in a2r_bot_follow:

    astar       A* with an octile heuristic.  The default.
    dijkstra    A* with the heuristic switched off.
    rrt         Sampling, with line-of-sight shortcutting.
    rrt_star    Sampling with rewiring.

A note on that list, because the measurements are worth knowing before you
choose.  scripts/test_global_planners.py runs all four over four maps:

  * A* and Dijkstra return **identical costs** on every map, which is the
    check that the heuristic is admissible.  A* expands a small fraction of
    the nodes — 181 against 39,693 on an open field — and is 12 to 118 times
    faster.  Dijkstra is kept because it is the right tool when you have no
    heuristic (multi-goal queries, cost fields), and because when a path looks
    wrong, "does Dijkstra agree?" is the fastest way to find out whether the
    bug is in the heuristic or the map.

  * RRT* genuinely does what it claims: its raw trees are 6 to 11 percent
    shorter than plain RRT's.  But line-of-sight shortcutting recovers almost
    all of that for plain RRT too, so **after smoothing the two are within
    1.1 percent of each other** — while RRT* costs about 38 times the time.
    On a 2-D grid where A* is available, optimal and deterministic, neither
    sampling planner is the right default.  They are here because the request
    asked for RRT, and because they are the right answer if you later move to
    a state space A* cannot search — one that includes heading, or velocity.

  * Both sampling planners come within 1 to 10 percent of A*'s optimal cost.

Subscribes
----------
/map or /global_costmap   nav_msgs/OccupancyGrid
/goal_pose                geometry_msgs/PoseStamped   (RViz "2D Goal Pose")
/slam/pose                geometry_msgs/PoseStamped
/odometry/filtered        nav_msgs/Odometry           (fallback for the start)

Publishes
---------
/plan                     nav_msgs/Path
/global_planner/status    std_msgs/String
/global_planner/markers   visualization_msgs/MarkerArray
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray

from a2r_bot_slam.planners import grid as gridlib
from a2r_bot_slam.planners import astar as astarlib
from a2r_bot_slam.planners import rrt as rrtlib


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GlobalPlannerNode(Node):
    """Plan a route across the map, on demand or on a timer."""

    def __init__(self):
        super().__init__('global_planner_node')

        self.declare_parameter('algorithm', 'astar')
        self.declare_parameter('map_topic', '/global_costmap')
        self.declare_parameter('map_fallback_topic', '/map')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('pose_topic', '/slam/pose')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('map_frame', 'map')

        self.declare_parameter('lethal_threshold', 65)
        self.declare_parameter('unknown_is_free', False)
        self.declare_parameter('obstacle_weight', 0.6)
        self.declare_parameter('allow_diagonal', True)
        self.declare_parameter('heuristic_weight', 1.0)
        self.declare_parameter('max_expansions', 400000)

        self.declare_parameter('rrt_iterations', 6000)
        self.declare_parameter('rrt_step_m', 0.30)
        self.declare_parameter('rrt_goal_bias', 0.08)
        self.declare_parameter('rrt_goal_tolerance_m', 0.25)
        self.declare_parameter('rrt_smooth_passes', 120)

        self.declare_parameter('simplify_path', True)
        self.declare_parameter('densify_spacing_m', 0.25)
        self.declare_parameter('snap_endpoints', True)
        self.declare_parameter('snap_radius_cells', 40)
        self.declare_parameter('replan_period_s', 2.0)
        self.declare_parameter('replan_on_map_update', True)
        self.declare_parameter('goal_reached_tolerance_m', 0.30)

        g = self.get_parameter
        self._algo = str(g('algorithm').value).strip().lower()
        self._map_frame = str(g('map_frame').value)
        self._lethal = int(g('lethal_threshold').value)
        self._unknown_free = bool(g('unknown_is_free').value)
        self._obs_w = float(g('obstacle_weight').value)
        self._diag = bool(g('allow_diagonal').value)
        self._hw = float(g('heuristic_weight').value)
        self._max_exp = int(g('max_expansions').value)
        self._simplify = bool(g('simplify_path').value)
        self._densify = float(g('densify_spacing_m').value)
        self._snap = bool(g('snap_endpoints').value)
        self._snap_r = int(g('snap_radius_cells').value)
        self._goal_tol = float(g('goal_reached_tolerance_m').value)

        self._grid = None
        self._grid_stamp = 0.0
        self._goal = None
        self._pose = (0.0, 0.0, 0.0)
        self._have_pose = False
        self._last_plan_t = 0.0
        self._planning = False

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._plan_pub = self.create_publisher(
            Path, str(g('plan_topic').value), latched)
        self._status_pub = self.create_publisher(
            String, '/global_planner/status', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/global_planner/markers', 1)

        self.create_subscription(OccupancyGrid, str(g('map_topic').value),
                                 self._map_cb, latched)
        if str(g('map_fallback_topic').value) != str(g('map_topic').value):
            self.create_subscription(
                OccupancyGrid, str(g('map_fallback_topic').value),
                self._map_fallback_cb, latched)
        self.create_subscription(PoseStamped, str(g('goal_topic').value),
                                 self._goal_cb, 10)
        self.create_subscription(PoseStamped, str(g('pose_topic').value),
                                 self._pose_cb, 10)
        self.create_subscription(Odometry, str(g('odom_topic').value),
                                 self._odom_cb, 10)
        self.create_subscription(Bool, '/global_planner/cancel',
                                 self._cancel_cb, 10)

        self.add_on_set_parameters_callback(self._on_param)
        self.create_timer(float(g('replan_period_s').value), self._tick)

        self.get_logger().info(
            'global_planner_node up — algorithm=%s, unknown space is %s'
            % (self._algo, 'traversable' if self._unknown_free else 'blocked'))

    # ==================================================================
    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'algorithm':
                new = str(p.value).strip().lower()
                if new not in ('astar', 'dijkstra', 'rrt', 'rrt_star'):
                    return SetParametersResult(
                        successful=False,
                        reason='algorithm must be astar, dijkstra, rrt or '
                               'rrt_star')
                self._algo = new
                self.get_logger().info('algorithm -> %s' % new)
            elif p.name == 'unknown_is_free':
                self._unknown_free = bool(p.value)
            elif p.name == 'obstacle_weight':
                self._obs_w = float(p.value)
        return SetParametersResult(successful=True)

    def _map_cb(self, msg: OccupancyGrid):
        self._store_map(msg, primary=True)

    def _map_fallback_cb(self, msg: OccupancyGrid):
        # Only used until the costmap node starts publishing. Planning against
        # the raw map means planning for a point robot, which will happily
        # route through a gap narrower than the car.
        if self._grid is None or (time.monotonic() - self._grid_stamp) > 5.0:
            self._store_map(msg, primary=False)

    def _store_map(self, msg, primary):
        self._grid = gridlib.Grid(
            msg.data, msg.info.width, msg.info.height, msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y,
            lethal_threshold=self._lethal,
            unknown_is_free=self._unknown_free)
        self._grid_stamp = time.monotonic()
        if not primary:
            self.get_logger().warn(
                'planning against the raw map — no /global_costmap yet, so '
                'obstacles are NOT inflated by the robot radius',
                throttle_duration_sec=20.0)

    def _goal_cb(self, msg: PoseStamped):
        self._goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.get_logger().info('goal: (%.2f, %.2f)' % self._goal)
        self._plan_now()

    def _cancel_cb(self, msg: Bool):
        if msg.data:
            self._goal = None
            self._publish_path([])
            self.get_logger().info('goal cancelled')

    def _pose_cb(self, msg: PoseStamped):
        self._pose = (float(msg.pose.position.x), float(msg.pose.position.y),
                      _yaw_of(msg.pose.orientation))
        self._have_pose = True

    def _odom_cb(self, msg: Odometry):
        # Only a fallback: odometry is in the odom frame, and the map is not.
        # Using it as the planning start is correct only before SLAM has
        # published a pose, when the two frames are still coincident.
        if not self._have_pose:
            self._pose = (float(msg.pose.pose.position.x),
                          float(msg.pose.pose.position.y),
                          _yaw_of(msg.pose.pose.orientation))

    # ==================================================================
    def _tick(self):
        if self._goal is None or self._grid is None:
            return
        gx, gy = self._goal
        if math.hypot(gx - self._pose[0], gy - self._pose[1]) < self._goal_tol:
            self.get_logger().info('goal reached')
            self._goal = None
            self._publish_path([])
            return
        self._plan_now()

    def _plan_now(self):
        if self._planning or self._goal is None or self._grid is None:
            return
        self._planning = True
        try:
            self._plan()
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error('planner raised: %s: %s'
                                    % (type(exc).__name__, exc))
        finally:
            self._planning = False

    def _plan(self):
        grid = self._grid
        grid.unknown_is_free = self._unknown_free
        sx, sy, _ = self._pose
        gx, gy = self._goal

        t0 = time.perf_counter()

        if self._algo in ('rrt', 'rrt_star'):
            fn = (rrtlib.plan_rrt if self._algo == 'rrt'
                  else rrtlib.plan_rrt_star)
            p = self.get_parameter
            path_world, stats = fn(
                grid, (sx, sy), (gx, gy),
                max_iterations=int(p('rrt_iterations').value),
                step_m=float(p('rrt_step_m').value),
                goal_bias=float(p('rrt_goal_bias').value),
                goal_tolerance_m=float(p('rrt_goal_tolerance_m').value),
                smooth_passes=int(p('rrt_smooth_passes').value))
            if path_world is None:
                self._fail(stats.get('reason', 'no path'), t0)
                return
        else:
            start_cell = grid.world_to_cell(sx, sy)
            goal_cell = grid.world_to_cell(gx, gy)

            if self._snap:
                snapped_start = grid.nearest_free(*start_cell,
                                                  max_radius_cells=self._snap_r)
                snapped_goal = grid.nearest_free(*goal_cell,
                                                 max_radius_cells=self._snap_r)
                if snapped_start is None:
                    self._fail('start is walled in', t0)
                    return
                if snapped_goal is None:
                    self._fail('goal is unreachable — no free cell nearby',
                               t0)
                    return
                if snapped_start != start_cell:
                    self.get_logger().warn(
                        'start cell was blocked — snapped by %d cells'
                        % max(abs(snapped_start[0] - start_cell[0]),
                              abs(snapped_start[1] - start_cell[1])))
                if snapped_goal != goal_cell:
                    self.get_logger().warn('goal cell was blocked — snapped')
                start_cell, goal_cell = snapped_start, snapped_goal

            fn = (astarlib.plan_dijkstra if self._algo == 'dijkstra'
                  else astarlib.plan_astar)
            cells, stats = fn(
                grid, start_cell, goal_cell,
                heuristic_weight=self._hw,
                allow_diagonal=self._diag,
                obstacle_weight=self._obs_w,
                max_expansions=self._max_exp)
            if cells is None:
                self._fail(stats.get('reason', 'no path'), t0)
                return
            if self._simplify:
                cells = gridlib.simplify(cells, grid)
            path_world = [grid.cell_to_world(c, r) for (c, r) in cells]

        if self._densify > 0.0:
            path_world = gridlib.densify(path_world, self._densify)

        dt = (time.perf_counter() - t0) * 1000.0
        length = gridlib.path_length(path_world)
        self._publish_path(path_world)
        self._publish_markers(path_world)

        msg = String()
        msg.data = ('%s: %d points, %.2f m, %.0f ms'
                    % (self._algo, len(path_world), length, dt))
        self._status_pub.publish(msg)
        self.get_logger().info(msg.data, throttle_duration_sec=2.0)

    def _fail(self, reason, t0):
        dt = (time.perf_counter() - t0) * 1000.0
        msg = String()
        msg.data = '%s FAILED: %s (%.0f ms)' % (self._algo, reason, dt)
        self._status_pub.publish(msg)
        self.get_logger().warn(msg.data, throttle_duration_sec=3.0)
        self._publish_path([])

    # ==================================================================
    def _publish_path(self, pts):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._map_frame
        for i, (x, y) in enumerate(pts):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            # Give each pose the heading of the segment leaving it, so RViz
            # arrows point along the route and the follower can use it.
            if i + 1 < len(pts):
                yaw = math.atan2(pts[i + 1][1] - y, pts[i + 1][0] - x)
            elif i > 0:
                yaw = math.atan2(y - pts[i - 1][1], x - pts[i - 1][0])
            else:
                yaw = 0.0
            ps.pose.orientation.w = math.cos(yaw * 0.5)
            ps.pose.orientation.z = math.sin(yaw * 0.5)
            path.poses.append(ps)
        self._plan_pub.publish(path)

    def _publish_markers(self, pts):
        arr = MarkerArray()
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self._map_frame
        m.ns = 'global_plan'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD if pts else Marker.DELETE
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 1.0, 0.9
        m.pose.orientation.w = 1.0
        for (x, y) in pts:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), 0.03
            m.points.append(p)
        arr.markers.append(m)
        self._marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
