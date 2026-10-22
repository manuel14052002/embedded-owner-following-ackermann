#!/usr/bin/env python3
"""
navigation_node.py
==================
Turns a global path into the same /cmd_vel the Jetson produces in follow mode.

-----------------------------------------------------------------------------
This node is the answer to "make the two interchangeable"
-----------------------------------------------------------------------------
The request was that SLAM navigation should use the same local path planning
as follow mode, and that the two should be swappable.  The way to get that is
not to share code between two path-following stacks.  It is to make sure there
is only one.

In follow mode the chain is:

    Jetson tracker --/cmd_vel--> smart_planner_node --/local_planner/cmd_out-->
        serial_bridge_node --> Arduino

In SLAM navigation mode it is:

    global_planner --/plan--> navigation_node --/cmd_vel--> smart_planner_node
        --/local_planner/cmd_out--> serial_bridge_node --> Arduino

Only the first box changes.  `smart_planner_node` cannot tell the difference
and does not need to: it receives a desired velocity and heading, and it is
responsible for getting there without hitting anything, exactly as before.
Every one of the eight local planners, the trap detector, the three-point
escape manoeuvre and the speed law apply unchanged in both modes.

That is what "interchangeable" has to mean to be worth anything.  It also
means the local planner can be swapped mid-mission — `ros2 param set
/local_planner_bridge planner rrt` works identically whether the robot is
following a person or driving to a mapped waypoint.

-----------------------------------------------------------------------------
What this node is therefore NOT allowed to do
-----------------------------------------------------------------------------
It must not avoid obstacles.  That is the local planner's job, and doing it
in two places produces two controllers fighting over the steering — the
classic symptom being a robot that oscillates in doorways because one layer
is pushing left while the other pushes right.

This node's entire job is: look ahead along the path, work out which way that
is, and ask for it.  If an obstacle is in the way, the local planner will
deal with it and the path follower will simply notice, next cycle, that it has
drifted off the path and aim back at it.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String, Bool, Float32
from visualization_msgs.msg import Marker


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class NavigationNode(Node):
    """Pure-pursuit path follower that emits a desired Twist."""

    def __init__(self):
        super().__init__('navigation_node')

        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('pose_topic', '/slam/pose')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('rate_hz', 20.0)

        self.declare_parameter('max_speed_mps', 0.9)
        self.declare_parameter('min_speed_mps', 0.28)
        self.declare_parameter('wheelbase_m', 0.280)
        self.declare_parameter('max_steer_deg', 30.0)

        # Lookahead scales with speed: a fixed lookahead is either too short
        # to be stable at speed or too long to track a tight turn slowly.
        self.declare_parameter('lookahead_min_m', 0.45)
        self.declare_parameter('lookahead_max_m', 1.20)
        self.declare_parameter('lookahead_gain', 0.9)

        self.declare_parameter('goal_tolerance_m', 0.25)
        self.declare_parameter('slow_radius_m', 1.20)
        self.declare_parameter('curvature_slow_gain', 0.55)
        self.declare_parameter('path_timeout_s', 5.0)
        self.declare_parameter('pose_timeout_s', 1.0)
        self.declare_parameter('off_path_warn_m', 1.5)
        self.declare_parameter('publish_markers', True)

        g = self.get_parameter
        self._rate = float(g('rate_hz').value)
        self._vmax = float(g('max_speed_mps').value)
        self._vmin = float(g('min_speed_mps').value)
        self._L = float(g('wheelbase_m').value)
        self._max_steer = math.radians(float(g('max_steer_deg').value))
        self._ld_min = float(g('lookahead_min_m').value)
        self._ld_max = float(g('lookahead_max_m').value)
        self._ld_gain = float(g('lookahead_gain').value)
        self._goal_tol = float(g('goal_tolerance_m').value)
        self._slow_r = float(g('slow_radius_m').value)
        self._curv_gain = float(g('curvature_slow_gain').value)
        self._path_timeout = float(g('path_timeout_s').value)
        self._pose_timeout = float(g('pose_timeout_s').value)
        self._off_warn = float(g('off_path_warn_m').value)
        self._markers = bool(g('publish_markers').value)

        self._path = []
        self._path_t = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._pose_t = 0.0
        self._speed = 0.0
        self._idx = 0
        self._enabled = True
        self._arrived = False
        self._last_warn = 0.0

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._cmd_pub = self.create_publisher(
            Twist, str(g('cmd_topic').value), 10)
        self._status_pub = self.create_publisher(String, '/navigation/status',
                                                 10)
        self._done_pub = self.create_publisher(Bool, '/navigation/goal_reached',
                                               latched)
        self._xtrack_pub = self.create_publisher(
            Float32, '/navigation/cross_track_error', 10)
        self._marker_pub = self.create_publisher(
            Marker, '/navigation/lookahead', 1)

        self.create_subscription(Path, str(g('path_topic').value),
                                 self._path_cb, latched)
        self.create_subscription(PoseStamped, str(g('pose_topic').value),
                                 self._pose_cb, 10)
        self.create_subscription(Odometry, str(g('odom_topic').value),
                                 self._odom_cb, 10)
        self.create_subscription(Bool, '/navigation/enable',
                                 self._enable_cb, 10)

        self.create_timer(1.0 / self._rate, self._tick)
        self.get_logger().info(
            'navigation_node up — publishing desired velocity on %s, which '
            'the local planner then makes safe'
            % str(g('cmd_topic').value))

    # ==================================================================
    def _path_cb(self, msg: Path):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not pts:
            if self._path:
                self.get_logger().info('path cleared')
            self._path = []
            return
        # Keep the index roughly where it was if the new path starts near the
        # old one. Resetting to zero on every replan makes the robot briefly
        # aim at a point behind itself, which on an Ackermann car means a
        # full-lock twitch every time the global planner ticks.
        if self._path and self._idx > 0:
            ox, oy = self._path[min(self._idx, len(self._path) - 1)]
            best, bd = 0, float('inf')
            for i, (x, y) in enumerate(pts):
                d = (x - ox) ** 2 + (y - oy) ** 2
                if d < bd:
                    bd, best = d, i
            self._idx = best if bd < 1.0 else 0
        else:
            self._idx = 0
        self._path = pts
        self._path_t = time.monotonic()
        self._arrived = False

    def _pose_cb(self, msg: PoseStamped):
        self._pose = (float(msg.pose.position.x), float(msg.pose.position.y),
                      _yaw_of(msg.pose.orientation))
        self._pose_t = time.monotonic()

    def _odom_cb(self, msg: Odometry):
        self._speed = float(msg.twist.twist.linear.x)
        if self._pose_t == 0.0:
            self._pose = (float(msg.pose.pose.position.x),
                          float(msg.pose.pose.position.y),
                          _yaw_of(msg.pose.pose.orientation))

    def _enable_cb(self, msg: Bool):
        self._enabled = bool(msg.data)
        if not self._enabled:
            self._stop()
        self.get_logger().info('navigation %s'
                               % ('enabled' if self._enabled else 'disabled'))

    # ==================================================================
    def _tick(self):
        now = time.monotonic()

        if not self._enabled or not self._path:
            return

        if (now - self._path_t) > self._path_timeout:
            if self._path:
                self.get_logger().warn(
                    'no path for %.1f s — stopping' % (now - self._path_t))
                self._path = []
                self._stop()
            return

        if self._pose_t == 0.0 or (now - self._pose_t) > self._pose_timeout:
            # Following a path without knowing where you are is worse than
            # not moving.
            self._stop()
            if now - self._last_warn > 3.0:
                self._last_warn = now
                self.get_logger().warn('no pose — holding')
            return

        px, py, pth = self._pose

        # ---- have we arrived? ----------------------------------------
        gx, gy = self._path[-1]
        dist_to_goal = math.hypot(gx - px, gy - py)
        if dist_to_goal < self._goal_tol:
            if not self._arrived:
                self._arrived = True
                self.get_logger().info('arrived (%.2f m from goal)'
                                       % dist_to_goal)
                b = Bool()
                b.data = True
                self._done_pub.publish(b)
            self._stop()
            return

        # ---- advance the index ---------------------------------------
        self._idx = self._closest_index(px, py)

        # ---- lookahead ------------------------------------------------
        speed_now = max(abs(self._speed), self._vmin)
        ld = min(self._ld_max,
                 max(self._ld_min, self._ld_gain * speed_now))
        # Never look further ahead than the remaining path, or the robot aims
        # past the goal and circles it.
        ld = min(ld, max(dist_to_goal, self._ld_min * 0.5))

        target = self._lookahead_point(px, py, ld)
        if target is None:
            self._stop()
            return
        tx, ty = target

        # ---- pure pursuit --------------------------------------------
        dx, dy = tx - px, ty - py
        # Into the body frame.
        c, s = math.cos(-pth), math.sin(-pth)
        lx = dx * c - dy * s
        ly = dx * s + dy * c

        dist = math.hypot(lx, ly)
        if dist < 1e-3:
            self._stop()
            return

        curvature = 2.0 * ly / (dist * dist)
        steer = math.atan(curvature * self._L)
        steer = max(-self._max_steer, min(self._max_steer, steer))

        # ---- speed ----------------------------------------------------
        v = self._vmax
        # Slow into the goal.
        if dist_to_goal < self._slow_r:
            v *= max(0.35, dist_to_goal / self._slow_r)
        # Slow for curvature: a car that takes a hairpin at full speed
        # understeers out of it, and the local planner then has to rescue a
        # situation the follower created.
        v *= 1.0 - self._curv_gain * min(1.0,
                                         abs(steer) / self._max_steer)
        v = max(self._vmin, min(self._vmax, v))

        # ---- cross-track error, for diagnosis ------------------------
        xte = self._cross_track(px, py)
        f = Float32()
        f.data = float(xte)
        self._xtrack_pub.publish(f)
        if abs(xte) > self._off_warn and (now - self._last_warn) > 3.0:
            self._last_warn = now
            self.get_logger().warn(
                '%.2f m off the path — the local planner is probably routing '
                'around something' % xte)

        # ---- emit -----------------------------------------------------
        # Twist in real units: linear.x in m/s, angular.z in rad/s. This is
        # exactly the contract the Jetson's tracker uses in follow mode, so
        # smart_planner_node needs no mode switch.
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(v * math.tan(steer) / self._L)
        self._cmd_pub.publish(cmd)

        if self._markers:
            self._publish_marker(tx, ty)

        st = String()
        st.data = ('idx %d/%d  goal %.2f m  ld %.2f  steer %+.0f deg  '
                   'v %.2f  xte %+.2f'
                   % (self._idx, len(self._path), dist_to_goal, ld,
                      math.degrees(steer), v, xte))
        self._status_pub.publish(st)

    # ==================================================================
    def _closest_index(self, px, py):
        """Nearest path point, searched forward only.

        Searching the whole path would let the robot latch onto an earlier
        part of a route that doubles back near itself — a figure-of-eight or
        a corridor walked twice — and drive the loop forever. Searching only
        forward from the current index makes progress monotonic.
        """
        best, bd = self._idx, float('inf')
        end = min(len(self._path), self._idx + 60)
        for i in range(self._idx, end):
            x, y = self._path[i]
            d = (x - px) ** 2 + (y - py) ** 2
            if d < bd:
                bd, best = d, i
        return best

    def _lookahead_point(self, px, py, ld):
        """First path point at least `ld` ahead of the closest one."""
        for i in range(self._idx, len(self._path)):
            x, y = self._path[i]
            if math.hypot(x - px, y - py) >= ld:
                return x, y
        return self._path[-1] if self._path else None

    def _cross_track(self, px, py):
        """Signed perpendicular distance to the current path segment."""
        i = self._idx
        if i + 1 >= len(self._path):
            i = max(0, len(self._path) - 2)
        if i + 1 >= len(self._path):
            return 0.0
        x0, y0 = self._path[i]
        x1, y1 = self._path[i + 1]
        sx, sy = x1 - x0, y1 - y0
        n = math.hypot(sx, sy)
        if n < 1e-9:
            return 0.0
        return ((px - x0) * sy - (py - y0) * sx) / n

    def _stop(self):
        self._cmd_pub.publish(Twist())

    def _publish_marker(self, x, y):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'lookahead'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.1, 0.95
        self._marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
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
