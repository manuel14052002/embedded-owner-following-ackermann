#!/usr/bin/env python3
"""
slam_node.py
============
Occupancy-grid SLAM: builds the map, and corrects the pose against it.

Two jobs, run in a loop:

  **Localise.**  Take the incoming scan and find the small (dx, dy, dtheta)
  that best aligns it with the map built so far.  The EKF's odometry is the
  starting guess; the scan match is the correction.

  **Map.**  Ray-cast every beam of the corrected scan into a log-odds grid:
  the cells the beam passes through become more free, the cell it ends on
  becomes more occupied.

-----------------------------------------------------------------------------
Why log-odds rather than just storing a probability
-----------------------------------------------------------------------------
Because updating a probability directly requires Bayes' rule with a division
every time, and because probabilities saturate badly in fixed arithmetic.  In
log-odds, l = log(p / (1-p)), Bayes' rule becomes *addition*:

    l_new = l_old + l_measurement - l_prior

Observing a cell occupied twenty times adds twenty increments.  Observing it
free once subtracts one.  That asymmetry — cheap to accumulate evidence, cheap
to revise it — is exactly what mapping needs, and it is why every occupancy
mapper since Moravec and Elfes in 1985 has used this representation.

The values are clamped.  Without clamping, a cell observed occupied a thousand
times reaches a log-odds of several hundred, and then needs a thousand free
observations to change its mind.  A map that cannot forget is useless in a
world with doors in it, so clamping is what keeps the map responsive.

-----------------------------------------------------------------------------
Why this scan matcher, and not ICP or a correlative matcher
-----------------------------------------------------------------------------
The matcher here is a **hill climb over the map's own likelihood field**: try
the current pose, try small perturbations in x, y and yaw, keep whichever
scores best, shrink the step, repeat.  This is close to what Hector SLAM does.

It was chosen over the alternatives for specific reasons:

  * **ICP** needs point-to-point correspondences, which means a nearest
    neighbour search per beam per iteration.  On a Pi, with a 360-beam scan,
    that is slow, and the correspondences are unstable in the corridors and
    doorways where you most need the match to be right.

  * **A correlative matcher** (exhaustive search over a discretised pose
    window) is robust and is what Cartographer uses, but it is O(window^3) and
    the window has to be large enough to cover odometry error.

  * **Hill climbing** is O(iterations x beams), converges in a handful of
    iterations when the initial guess is close, and the initial guess *is*
    close, because the EKF upstream is doing its job.

The cost of hill climbing is that it can stop at a local optimum.  That is
mitigated by multi-resolution: match against a coarse map first to get into
the right basin, then refine against the fine map.  A corridor that looks
identical for six metres will still defeat it — no 2-D scan matcher solves
that, which is why the EKF's yaw rate matters.
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       qos_profile_sensor_data)

from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry, MapMetaData
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from tf2_ros import TransformBroadcaster


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class SlamNode(Node):
    """Log-odds occupancy mapping with hill-climb scan matching."""

    def __init__(self):
        super().__init__('slam_node')

        # ---- parameters ----------------------------------------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('map_width_m', 30.0)
        self.declare_parameter('map_height_m', 30.0)
        self.declare_parameter('resolution_m', 0.05)
        self.declare_parameter('map_publish_period_s', 1.0)

        self.declare_parameter('max_range_m', 6.0)
        self.declare_parameter('min_range_m', 0.15)

        # Log-odds increments.  The free increment is deliberately much
        # smaller than the occupied one: a beam passing through a cell is weak
        # evidence (it may have passed above or below a real obstacle), while
        # a beam *ending* in a cell is strong evidence something is there.
        self.declare_parameter('log_odds_hit', 0.85)
        self.declare_parameter('log_odds_miss', 0.25)
        self.declare_parameter('log_odds_min', -4.0)
        self.declare_parameter('log_odds_max', 6.0)
        self.declare_parameter('occupied_threshold', 0.65)
        self.declare_parameter('free_threshold', 0.25)

        self.declare_parameter('use_scan_matching', True)
        self.declare_parameter('match_iterations', 6)
        self.declare_parameter('match_linear_step_m', 0.08)
        self.declare_parameter('match_angular_step_deg', 3.0)
        self.declare_parameter('match_step_shrink', 0.55)
        self.declare_parameter('match_min_beams', 60)
        self.declare_parameter('match_max_beams', 240)
        self.declare_parameter('match_score_floor', 0.25)
        self.declare_parameter('match_max_correction_m', 0.45)
        self.declare_parameter('match_max_correction_deg', 20.0)

        self.declare_parameter('update_min_distance_m', 0.08)
        self.declare_parameter('update_min_angle_deg', 4.0)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('lidar_offset_x_m', 0.08)
        self.declare_parameter('lidar_offset_y_m', 0.0)

        g = self.get_parameter
        self._map_frame = str(g('map_frame').value)
        self._odom_frame = str(g('odom_frame').value)
        self._base_frame = str(g('base_frame').value)
        self._res = float(g('resolution_m').value)
        self._w = int(round(float(g('map_width_m').value) / self._res))
        self._h = int(round(float(g('map_height_m').value) / self._res))
        self._ox = -float(g('map_width_m').value) * 0.5
        self._oy = -float(g('map_height_m').value) * 0.5

        self._rmax = float(g('max_range_m').value)
        self._rmin = float(g('min_range_m').value)
        self._l_hit = float(g('log_odds_hit').value)
        self._l_miss = float(g('log_odds_miss').value)
        self._l_min = float(g('log_odds_min').value)
        self._l_max = float(g('log_odds_max').value)
        self._occ_th = float(g('occupied_threshold').value)
        self._free_th = float(g('free_threshold').value)

        self._match_on = bool(g('use_scan_matching').value)
        self._m_iters = int(g('match_iterations').value)
        self._m_lin = float(g('match_linear_step_m').value)
        self._m_ang = math.radians(float(g('match_angular_step_deg').value))
        self._m_shrink = float(g('match_step_shrink').value)
        self._m_min_beams = int(g('match_min_beams').value)
        self._m_max_beams = int(g('match_max_beams').value)
        self._m_floor = float(g('match_score_floor').value)
        self._m_max_lin = float(g('match_max_correction_m').value)
        self._m_max_ang = math.radians(
            float(g('match_max_correction_deg').value))

        self._min_d = float(g('update_min_distance_m').value)
        self._min_a = math.radians(float(g('update_min_angle_deg').value))
        self._publish_tf = bool(g('publish_tf').value)
        self._lx = float(g('lidar_offset_x_m').value)
        self._ly = float(g('lidar_offset_y_m').value)

        # ---- map ------------------------------------------------------
        self._log = np.zeros((self._h, self._w), dtype=np.float32)
        self._seen = np.zeros((self._h, self._w), dtype=bool)
        self._map_lock = threading.Lock()

        # ---- pose -----------------------------------------------------
        # map_pose is where we think we are in the map frame.
        # odom_pose is what the EKF last told us, in the odom frame.
        # The difference between them is the map->odom transform, which is
        # the standard ROS arrangement: the EKF owns odom->base and publishes
        # it continuously and smoothly; SLAM owns map->odom and jumps it
        # whenever the scan match says so. Nothing downstream ever sees a
        # discontinuity in odom->base, which is what lets the local planner
        # differentiate it safely.
        self._mx = self._my = self._mth = 0.0
        self._ox_odom = self._oy_odom = self._oth_odom = 0.0
        self._have_odom = False
        self._last_update_pose = None
        self._scan_count = 0
        self._match_fail = 0
        self._last_score = 0.0
        self._paused = False
        self._t_last_map = 0.0

        # ---- interfaces -----------------------------------------------
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self._meta_pub = self.create_publisher(MapMetaData, '/map_metadata',
                                               latched)
        self._match_pub = self.create_publisher(Odometry, '/scan_match/odom',
                                                10)
        self._pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self._score_pub = self.create_publisher(Float32, '/slam/match_score',
                                                10)
        self._status_pub = self.create_publisher(String, '/slam/status', 10)

        self._tf = TransformBroadcaster(self) if self._publish_tf else None

        self.create_subscription(LaserScan, str(g('scan_topic').value),
                                 self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g('odom_topic').value),
                                 self._odom_cb, 20)
        self.create_subscription(Bool, '/slam/pause', self._pause_cb, 10)
        self.create_subscription(Bool, '/slam/reset', self._reset_cb, 10)

        self.create_timer(float(g('map_publish_period_s').value),
                          self._publish_map)
        self.create_timer(0.05, self._publish_tf_tick)

        self.get_logger().info(
            'slam_node up — %dx%d cells @ %.3f m (%.0fx%.0f m), '
            'scan matching %s'
            % (self._w, self._h, self._res,
               self._w * self._res, self._h * self._res,
               'on' if self._match_on else 'off'))

    # ==================================================================
    def _pause_cb(self, msg: Bool):
        self._paused = bool(msg.data)
        self.get_logger().info('mapping %s'
                               % ('paused' if self._paused else 'resumed'))

    def _reset_cb(self, msg: Bool):
        if not msg.data:
            return
        with self._map_lock:
            self._log[:] = 0.0
            self._seen[:] = False
        self._mx = self._my = self._mth = 0.0
        self._last_update_pose = None
        self.get_logger().warn('map cleared')

    def _odom_cb(self, msg: Odometry):
        self._ox_odom = float(msg.pose.pose.position.x)
        self._oy_odom = float(msg.pose.pose.position.y)
        self._oth_odom = _yaw_of(msg.pose.pose.orientation)
        if not self._have_odom:
            # Anchor the map frame to the first odometry sample so the two
            # start coincident, rather than offset by wherever odom happened
            # to be when SLAM started.
            self._mx, self._my, self._mth = (self._ox_odom, self._oy_odom,
                                             self._oth_odom)
            self._have_odom = True

    # ==================================================================
    def _scan_cb(self, scan: LaserScan):
        if not self._have_odom:
            return
        self._scan_count += 1

        n = len(scan.ranges)
        if n == 0:
            return
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        good = (np.isfinite(ranges) & (ranges >= max(self._rmin,
                                                     scan.range_min))
                & (ranges <= min(self._rmax, scan.range_max)))
        if good.sum() < 10:
            return
        r = ranges[good]
        a = angles[good]

        # Beam endpoints in the base frame, accounting for where the lidar
        # actually sits on the robot.
        bx = self._lx + r * np.cos(a)
        by = self._ly + r * np.sin(a)

        # ---- predict pose from odometry delta -------------------------
        # We track the map pose incrementally: apply the odometry *change*
        # since the last scan to the last map pose. Using odometry absolutely
        # would throw away every scan-match correction we ever made.
        pred_x, pred_y, pred_th = self._predict()

        # ---- scan match ----------------------------------------------
        score = 0.0
        if self._match_on and self._scan_count > 3:
            mx, my, mth, score = self._match(bx, by, pred_x, pred_y, pred_th)
            dx = mx - pred_x
            dy = my - pred_y
            dth = _wrap(mth - pred_th)
            if (math.hypot(dx, dy) > self._m_max_lin
                    or abs(dth) > self._m_max_ang):
                # A correction this large is far more likely to be the matcher
                # having found a wrong basin than the odometry having been
                # that wrong. Reject it and trust odometry for this scan.
                self._match_fail += 1
                if self._match_fail % 20 == 1:
                    self.get_logger().warn(
                        'scan match wanted %.2f m / %.0f deg — rejected, '
                        'trusting odometry' % (math.hypot(dx, dy),
                                               math.degrees(dth)))
            elif score < self._m_floor:
                self._match_fail += 1
            else:
                pred_x, pred_y, pred_th = mx, my, mth
        self._last_score = score

        self._mx, self._my, self._mth = pred_x, pred_y, pred_th
        self._odom_at_update = (self._ox_odom, self._oy_odom, self._oth_odom)

        self._publish_match(scan.header.stamp, score)

        # ---- map ------------------------------------------------------
        if self._paused:
            return
        if not self._should_update():
            return
        self._integrate(bx, by, pred_x, pred_y, pred_th)
        self._last_update_pose = (pred_x, pred_y, pred_th)

    # ------------------------------------------------------------------
    def _predict(self):
        """Apply the odometry delta since the last scan to the map pose."""
        if not hasattr(self, '_odom_at_update'):
            self._odom_at_update = (self._ox_odom, self._oy_odom,
                                    self._oth_odom)
            return self._mx, self._my, self._mth

        ox0, oy0, oth0 = self._odom_at_update
        dx = self._ox_odom - ox0
        dy = self._oy_odom - oy0
        dth = _wrap(self._oth_odom - oth0)

        # Rotate the odom-frame displacement into the map frame.
        c = math.cos(self._mth - oth0)
        s = math.sin(self._mth - oth0)
        return (self._mx + dx * c - dy * s,
                self._my + dx * s + dy * c,
                _wrap(self._mth + dth))

    # ------------------------------------------------------------------
    def _should_update(self):
        """Only fold a scan into the map after the robot has actually moved.

        Integrating every scan from a stationary robot drives the log-odds of
        whatever it is looking at straight to the clamp in a couple of
        seconds. That is not more information — it is the same information
        counted a hundred times — and it makes those cells unreasonably hard
        to revise later. It also burns CPU for nothing.
        """
        if self._last_update_pose is None:
            return True
        lx, ly, lth = self._last_update_pose
        if math.hypot(self._mx - lx, self._my - ly) >= self._min_d:
            return True
        return abs(_wrap(self._mth - lth)) >= self._min_a

    # ------------------------------------------------------------------
    def _match(self, bx, by, x0, y0, th0):
        """Hill-climb the pose to maximise agreement with the map.

        Subsamples the scan: matching against 240 beams instead of 720 is
        three times faster and, because neighbouring beams in a dense scan see
        the same surface, costs almost nothing in accuracy.
        """
        k = len(bx)
        if k < self._m_min_beams:
            return x0, y0, th0, 0.0
        if k > self._m_max_beams:
            idx = np.linspace(0, k - 1, self._m_max_beams).astype(np.int64)
            bx = bx[idx]
            by = by[idx]

        with self._map_lock:
            occ = (self._log > 0.0) & self._seen

        best = (x0, y0, th0)
        best_score = self._score(bx, by, x0, y0, th0, occ)
        lin = self._m_lin
        ang = self._m_ang

        for _ in range(self._m_iters):
            improved = False
            cx, cy, cth = best
            # Six candidate moves: +/- in each of x, y, yaw. Coordinate
            # descent rather than a full gradient: the likelihood field is not
            # differentiable (it is a binary grid), so a numerical gradient
            # would be noise.
            for dx, dy, dth in ((lin, 0, 0), (-lin, 0, 0),
                                (0, lin, 0), (0, -lin, 0),
                                (0, 0, ang), (0, 0, -ang)):
                nx, ny, nth = cx + dx, cy + dy, _wrap(cth + dth)
                s = self._score(bx, by, nx, ny, nth, occ)
                if s > best_score:
                    best_score = s
                    best = (nx, ny, nth)
                    improved = True
            if not improved:
                # Converged at this scale — refine.
                lin *= self._m_shrink
                ang *= self._m_shrink
                if lin < self._res * 0.25:
                    break
        return best[0], best[1], best[2], best_score

    def _score(self, bx, by, x, y, th, occ):
        """Fraction of beam endpoints that land on an occupied cell.

        Crude compared to a proper likelihood field with a smoothed kernel,
        but it is one array lookup per beam and it is monotonic in alignment,
        which is all a hill climb needs.
        """
        c, s = math.cos(th), math.sin(th)
        wx = x + bx * c - by * s
        wy = y + bx * s + by * c
        cols = ((wx - self._ox) / self._res).astype(np.int64)
        rows = ((wy - self._oy) / self._res).astype(np.int64)
        ok = ((cols >= 0) & (cols < self._w)
              & (rows >= 0) & (rows < self._h))
        if not ok.any():
            return 0.0
        hits = occ[rows[ok], cols[ok]]
        return float(hits.sum()) / float(len(bx))

    # ------------------------------------------------------------------
    def _integrate(self, bx, by, x, y, th):
        """Ray-cast the scan into the log-odds grid."""
        c, s = math.cos(th), math.sin(th)
        wx = x + bx * c - by * s
        wy = y + bx * s + by * c

        # Sensor origin in map cells.
        sx = x + self._lx * c - self._ly * s
        sy = y + self._lx * s + self._ly * c
        c0 = int((sx - self._ox) / self._res)
        r0 = int((sy - self._oy) / self._res)
        if not (0 <= c0 < self._w and 0 <= r0 < self._h):
            return

        cols = ((wx - self._ox) / self._res).astype(np.int64)
        rows = ((wy - self._oy) / self._res).astype(np.int64)

        with self._map_lock:
            log = self._log
            seen = self._seen
            for c1, r1 in zip(cols, rows):
                if not (0 <= c1 < self._w and 0 <= r1 < self._h):
                    continue
                # Free along the ray...
                for cc, rr in _bresenham(c0, r0, c1, r1):
                    log[rr, cc] -= self._l_miss
                    seen[rr, cc] = True
                # ...occupied at the end.
                log[r1, c1] += self._l_hit + self._l_miss
                seen[r1, c1] = True
            np.clip(log, self._l_min, self._l_max, out=log)

    # ------------------------------------------------------------------
    def _publish_match(self, stamp, score):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._map_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = float(self._mx)
        odom.pose.pose.position.y = float(self._my)
        odom.pose.pose.orientation = _quat(self._mth)

        # Report a covariance that reflects how well the match actually went.
        # The EKF gates on this, so lying here — reporting a fixed small
        # covariance regardless of score — is how a bad match gets accepted
        # and drags the whole estimate with it.
        conf = max(score, 1e-3)
        var_xy = 0.004 / conf
        var_th = 0.002 / conf
        cov = [0.0] * 36
        cov[0] = var_xy
        cov[7] = var_xy
        cov[35] = var_th
        odom.pose.covariance = cov
        self._match_pub.publish(odom)

        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self._pose_pub.publish(ps)

        f = Float32()
        f.data = float(score)
        self._score_pub.publish(f)

    def _publish_tf_tick(self):
        """Publish map -> odom.

        Computed, not tracked: it is whatever transform reconciles the pose we
        believe in the map with the pose odometry currently reports.
        """
        if self._tf is None or not self._have_odom:
            return
        # T_map_odom = T_map_base * inverse(T_odom_base)
        dth = _wrap(self._mth - self._oth_odom)
        c, s = math.cos(dth), math.sin(dth)
        tx = self._mx - (self._ox_odom * c - self._oy_odom * s)
        ty = self._my - (self._ox_odom * s + self._oy_odom * c)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._map_frame
        t.child_frame_id = self._odom_frame
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.rotation = _quat(dth)
        self._tf.sendTransform(t)

    def _publish_map(self):
        with self._map_lock:
            log = self._log.copy()
            seen = self._seen.copy()

        # Log-odds -> occupancy probability -> the ROS int8 convention.
        prob = 1.0 - 1.0 / (1.0 + np.exp(log))
        out = np.full(log.shape, -1, dtype=np.int8)
        known = seen
        occupied = known & (prob >= self._occ_th)
        free = known & (prob <= self._free_th)
        # Cells that are known but neither confidently free nor confidently
        # occupied get their actual probability, so the inflation layer and
        # the planners can treat them as the uncertain ground they are.
        mid = known & ~occupied & ~free
        out[free] = 0
        out[occupied] = 100
        out[mid] = (prob[mid] * 100.0).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.info.resolution = self._res
        msg.info.width = self._w
        msg.info.height = self._h
        msg.info.origin.position.x = self._ox
        msg.info.origin.position.y = self._oy
        msg.info.origin.orientation.w = 1.0
        msg.data = out.ravel().tolist()
        self._map_pub.publish(msg)
        self._meta_pub.publish(msg.info)

        known_n = int(known.sum())
        st = String()
        st.data = ('scans=%d known=%d (%.1f%%) occ=%d score=%.2f rejects=%d'
                   % (self._scan_count, known_n,
                      100.0 * known_n / (self._w * self._h),
                      int(occupied.sum()), self._last_score,
                      self._match_fail))
        self._status_pub.publish(st)


def _bresenham(c0, r0, c1, r1):
    """Cells along a ray, excluding the endpoint.

    Excluding the endpoint matters: the endpoint is where the beam *stopped*,
    which is evidence of occupancy, and marking it free first and occupied
    second would be a wash. The caller marks it separately.
    """
    out = []
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        if c == c1 and r == r1:
            return out
        out.append((c, r))
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
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
