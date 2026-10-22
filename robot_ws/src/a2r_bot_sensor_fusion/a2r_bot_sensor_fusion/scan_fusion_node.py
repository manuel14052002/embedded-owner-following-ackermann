#!/usr/bin/env python3
"""
scan_fusion_node.py
===================
Merges the lidar scan with the two DYP-A21 ultrasonic sensors into a single
/scan/fused that the local planners consume unchanged.

-----------------------------------------------------------------------------
Why bother?  The lidar already sees everything.
-----------------------------------------------------------------------------
It does not, and the gaps are exactly the ones that hurt:

  * **Glass, gloss and dark fabric.** A 2-D lidar fires an infrared beam and
    waits for a return.  A glass door returns almost nothing; a matte black
    trouser leg returns very little.  Both read as "infinite range" — which
    the planner interprets as free space, and drives into.  Ultrasound does
    not care about colour or infrared reflectivity at all.  It cares about
    acoustic impedance, and glass is an excellent acoustic reflector.

  * **The blind cone below the scan plane.** The lidar sits on the top deck
    and sweeps a horizontal plane.  A kerb, a doorstep, a cat, or the foot of
    a table are all below that plane and are simply invisible.  The front
    sonar is mounted low on the nose and looks slightly down.

  * **Behind.** The lidar is 360-degree, but it is occluded to the rear by the
    battery and the Jetson.  The rear sonar covers the reversing arc, which
    matters a great deal because the escape manoeuvre in the local planner
    reverses.

Ultrasound has its own failure modes — a 60-degree beam that cannot tell you
*where* in that cone the obstacle is, specular reflection off angled surfaces,
and crosstalk between the two units.  So neither sensor is trusted on its own.

-----------------------------------------------------------------------------
How the fusion works
-----------------------------------------------------------------------------
The scan is divided into angular sectors.  Each sector carries its own 1-D
Kalman filter over "range to the nearest obstacle in this sector":

    state       r        (metres)
    predict     r' = r - v_r dt,  where v_r is the closing speed implied by
                                  the robot's own motion into that sector
    update      lidar minimum within the sector, and, if the sector falls
                inside a sonar's cone, the sonar reading

Two things make this work rather than just averaging:

1. **The measurement models differ.**  The lidar measures range at a precise
   bearing; the sonar measures the range to the nearest thing anywhere in a
   60-degree cone.  So a sonar reading is folded into *every* sector in its
   cone, but only as an upper bound on that sector's range — never to push a
   sector's range further out.  A sonar that says "something at 0.8 m
   somewhere ahead" is strong evidence that no sector ahead is clear beyond
   0.8 m, and no evidence at all about which sector it is.

2. **Disagreement is resolved in favour of the pessimist.**  When the lidar
   says 5 m and the sonar says 0.9 m, the honest reading is 0.9 m — that is
   the glass door case, and it is the whole reason this node exists.  When the
   lidar says 0.9 m and the sonar says 5 m, the honest reading is still 0.9 m,
   because a sonar missing a thin table leg is routine.  The filter encodes
   this asymmetry explicitly rather than pretending both sensors are symmetric
   Gaussian observers, which they are not.

The prediction step matters more than it looks.  Between scans at 10 Hz the
car travelling at 1.2 m/s covers 12 cm.  Propagating each sector's range
forward by the robot's own motion means a sector that is briefly occluded or
dropped does not snap back to "clear"; it decays towards clear at the rate the
geometry actually allows.

Subscribes
----------
/scan                sensor_msgs/LaserScan
/ultrasonic/front    sensor_msgs/Range
/ultrasonic/rear     sensor_msgs/Range
/odometry/filtered   nav_msgs/Odometry   (for the motion prediction)

Publishes
---------
/scan/fused          sensor_msgs/LaserScan   — drop-in replacement for /scan
/scan/sonar_overlay  sensor_msgs/LaserScan   — sonar contribution alone, for
                                               debugging in RViz
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Range

from a2r_bot_sensor_fusion.kalman import wrap_angle


class SonarState:
    """Latest reading from one ultrasonic sensor, plus its geometry."""

    def __init__(self, centre_rad, fov_rad, min_r, max_r):
        self.centre = centre_rad
        self.fov = fov_rad
        self.min_r = min_r
        self.max_r = max_r
        self.range = float('inf')
        self.stamp = 0.0
        self.valid = False


class ScanFusionNode(Node):
    """Per-sector Kalman fusion of lidar and ultrasonic range data."""

    def __init__(self):
        super().__init__('scan_fusion_node')

        # ---- parameters ----------------------------------------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/scan/fused')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('frame_id', '')          # '' = keep the lidar's
        self.declare_parameter('sector_count', 72)      # 5 degrees each
        self.declare_parameter('publish_overlay', True)

        self.declare_parameter('front_centre_deg', 0.0)
        self.declare_parameter('rear_centre_deg', 180.0)
        self.declare_parameter('sonar_fov_deg', 60.0)
        self.declare_parameter('sonar_min_range_m', 0.28)
        self.declare_parameter('sonar_max_range_m', 5.0)
        self.declare_parameter('sonar_timeout_s', 0.6)

        # Where the sonars sit relative to base_footprint.  A reading of 0.9 m
        # from a sensor mounted 20 cm ahead of the origin is 1.1 m from the
        # origin, and the planner reasons about the origin.
        self.declare_parameter('front_offset_x_m', 0.215)
        self.declare_parameter('rear_offset_x_m', -0.210)
        # ...but the fused scan is published in the LIDAR's own frame (the
        # header is copied from the incoming scan), and every consumer then
        # applies its own lidar->base_link offset. Converting the sonar to
        # base_link here therefore double-counted that offset and pushed
        # sonar-only obstacles ~8 cm further away than they are - the wrong
        # direction for glass doors and kerbs, which only the sonar sees.
        # Express the sonar relative to the SCAN FRAME instead. Keep this
        # equal to lidar_offset_x in the planner / slam configs.
        self.declare_parameter('scan_frame_offset_x_m', -0.140)

        self.declare_parameter('lidar_variance', 9.0e-4)     # (30 mm)^2
        self.declare_parameter('sonar_variance', 2.5e-3)     # (50 mm)^2
        self.declare_parameter('process_variance', 4.0e-3)
        self.declare_parameter('gate_sigma', 3.5)

        self.declare_parameter('sonar_only_shortens', True)
        self.declare_parameter('sonar_disagreement_m', 0.35)
        self.declare_parameter('sonar_confirm_count', 2)
        self.declare_parameter('max_prediction_dt_s', 0.5)
        self.declare_parameter('range_max_clamp_m', 0.0)  # 0 = use scan's own

        g = self.get_parameter
        self._out_topic = str(g('output_topic').value)
        self._frame_override = str(g('frame_id').value)
        self._nsec = int(g('sector_count').value)
        self._overlay = bool(g('publish_overlay').value)
        self._sonar_timeout = float(g('sonar_timeout_s').value)
        self._var_lidar = float(g('lidar_variance').value)
        self._var_sonar = float(g('sonar_variance').value)
        self._var_proc = float(g('process_variance').value)
        self._gate = float(g('gate_sigma').value)
        self._only_shortens = bool(g('sonar_only_shortens').value)
        self._disagree = float(g('sonar_disagreement_m').value)
        self._confirm_n = int(g('sonar_confirm_count').value)
        self._max_dt = float(g('max_prediction_dt_s').value)
        self._clamp = float(g('range_max_clamp_m').value)

        fov = math.radians(float(g('sonar_fov_deg').value))
        smin = float(g('sonar_min_range_m').value)
        smax = float(g('sonar_max_range_m').value)

        self._front = SonarState(
            math.radians(float(g('front_centre_deg').value)), fov, smin, smax)
        self._rear = SonarState(
            wrap_angle(math.radians(float(g('rear_centre_deg').value))),
            fov, smin, smax)
        _sfx = float(g('scan_frame_offset_x_m').value)
        self._front.offset = float(g('front_offset_x_m').value) - _sfx
        self._rear.offset = float(g('rear_offset_x_m').value) - _sfx
        self._front.confirm = 0
        self._rear.confirm = 0
        self._front.last_raw = float('inf')
        self._rear.last_raw = float('inf')

        # ---- per-sector filter state ----------------------------------
        # Kept as flat numpy arrays rather than 72 KalmanFilter objects: the
        # maths is one-dimensional and identical per sector, so vectorising it
        # turns 72 Python-level filter updates into four array operations.
        # On a Pi 4 at 10 Hz that is the difference between 0.4 ms and 14 ms.
        self._sec_r = np.full(self._nsec, np.inf)
        self._sec_P = np.full(self._nsec, 1.0)
        self._sec_t = 0.0

        self._v = 0.0
        self._w = 0.0
        self._scan_count = 0
        self._last_log = time.monotonic()
        self._n_sonar_saves = 0

        # ---- interfaces -----------------------------------------------
        self._pub = self.create_publisher(LaserScan, self._out_topic,
                                          qos_profile_sensor_data)
        self._overlay_pub = (
            self.create_publisher(LaserScan, '/scan/sonar_overlay',
                                  qos_profile_sensor_data)
            if self._overlay else None)

        self.create_subscription(LaserScan, str(g('scan_topic').value),
                                 self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Range, '/ultrasonic/front',
                                 self._front_cb, qos_profile_sensor_data)
        self.create_subscription(Range, '/ultrasonic/rear',
                                 self._rear_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g('odom_topic').value),
                                 self._odom_cb, 10)

        self.get_logger().info(
            'scan_fusion_node up — %d sectors, sonar FOV %.0f deg, '
            'publishing %s'
            % (self._nsec, math.degrees(fov), self._out_topic))

    # ==================================================================
    def _odom_cb(self, msg: Odometry):
        self._v = float(msg.twist.twist.linear.x)
        self._w = float(msg.twist.twist.angular.z)

    def _sonar_cb(self, state: SonarState, msg: Range):
        r = float(msg.range)

        # REP-117: +inf means "nothing in range", which is information, not a
        # fault.  -inf would mean "something too close to measure", which we
        # treat as the minimum range — the pessimistic reading.
        if math.isinf(r) and r > 0:
            state.range = float('inf')
            state.valid = True
            state.stamp = time.monotonic()
            state.confirm = 0
            return
        if math.isinf(r) and r < 0:
            r = state.min_r

        if not (state.min_r <= r <= state.max_r) or math.isnan(r):
            state.valid = False
            return

        # Require a reading to repeat before we believe a sudden large change.
        # A single spurious short echo — off the ground, off a wheel arch, off
        # the other sonar — would otherwise slam the brakes on.
        if abs(r - state.last_raw) > self._disagree:
            state.confirm += 1
            if state.confirm < self._confirm_n:
                state.last_raw = r
                return
        state.confirm = 0
        state.last_raw = r

        # Shift from the sensor's mounting point to the SCAN FRAME origin,
        # which is what the published header says these ranges are measured
        # from. A negative shift is possible (a sensor behind the lidar), so
        # clamp rather than take a blind absolute value.
        state.range = max(0.0, r + getattr(state, 'offset', 0.0))
        state.valid = True
        state.stamp = time.monotonic()

    def _front_cb(self, msg: Range):
        self._sonar_cb(self._front, msg)

    def _rear_cb(self, msg: Range):
        self._sonar_cb(self._rear, msg)

    # ==================================================================
    def _scan_cb(self, scan: LaserScan):
        now = time.monotonic()
        n = len(scan.ranges)
        if n == 0:
            return
        self._scan_count += 1

        ranges = np.asarray(scan.ranges, dtype=float)
        rmax = self._clamp if self._clamp > 0.0 else float(scan.range_max)
        rmin = float(scan.range_min)

        # Anything non-finite, zero, or beyond the sensor's rated range is
        # "no return".  Keep it as inf rather than range_max: the distinction
        # between "definitely clear to 12 m" and "no information" is one the
        # planner's inflation layer genuinely uses.
        bad = ~np.isfinite(ranges) | (ranges <= rmin) | (ranges > rmax)
        ranges = np.where(bad, np.inf, ranges)

        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        # ---- bucket the beams into sectors ---------------------------
        sec_idx = self._sector_of(angles)
        lidar_sec = np.full(self._nsec, np.inf)
        # Minimum range per sector.  np.minimum.at is the unbuffered scatter
        # reduction — a plain fancy-index assignment would keep only the last
        # beam per sector, not the nearest.
        np.minimum.at(lidar_sec, sec_idx, ranges)

        # ---- predict -------------------------------------------------
        dt = 0.0 if self._sec_t == 0.0 else (now - self._sec_t)
        self._sec_t = now
        if dt <= 0.0 or dt > self._max_dt:
            # Long gap: throw away the history rather than propagate a stale
            # world forward by an unknown amount.
            self._sec_r[:] = np.inf
            self._sec_P[:] = 1.0
        else:
            centres = self._sector_centres()
            # Closing speed into each sector: forward motion closes the
            # sectors ahead and opens those behind, scaled by cos(bearing).
            closing = self._v * np.cos(centres)
            finite = np.isfinite(self._sec_r)
            self._sec_r[finite] -= closing[finite] * dt
            np.maximum(self._sec_r, 0.0, out=self._sec_r)
            # Rotation shuffles which sector sees what.  Rather than
            # resampling, we inflate the covariance in proportion to how far
            # the sectors rotated — honest about the loss of correspondence
            # without pretending to track it.
            rot_frac = abs(self._w) * dt / (2.0 * math.pi / self._nsec)
            self._sec_P += self._var_proc * dt + 0.02 * rot_frac

        # ---- update with lidar ---------------------------------------
        self._sector_update(lidar_sec, self._var_lidar, gate=self._gate)

        # ---- update with sonar ---------------------------------------
        overlay = np.full(self._nsec, np.inf)
        for state in (self._front, self._rear):
            if not state.valid or (now - state.stamp) > self._sonar_timeout:
                continue
            if not math.isfinite(state.range):
                continue
            mask = self._cone_mask(state)
            overlay[mask] = state.range

            if self._only_shortens:
                # A sonar return is an upper bound on every sector in the
                # cone, and nothing more.  Only apply it where it actually
                # shortens the current estimate.
                apply = mask & (state.range < self._sec_r - 1e-6)
            else:
                apply = mask

            if not np.any(apply):
                continue

            z = np.full(self._nsec, np.inf)
            z[apply] = state.range
            saved = np.sum(apply & ~np.isfinite(lidar_sec))
            self._n_sonar_saves += int(saved)
            # No gate here: the whole point is to accept a reading the lidar
            # disagrees with.  The `only_shortens` rule is the safety
            # argument, and it is a conservative one.
            self._sector_update(z, self._var_sonar, gate=None)

        # ---- write the fused ranges back onto the beam array ---------
        fused = ranges.copy()
        sec_val = self._sec_r[sec_idx]
        # Only ever pull a beam inwards.  The per-sector estimate is a
        # minimum over the sector, so pushing an individual beam out to it
        # would invent free space that no sensor observed.
        improve = np.isfinite(sec_val) & (
            ~np.isfinite(fused) | (sec_val < fused))
        fused[improve] = sec_val[improve]

        out = LaserScan()
        out.header = scan.header
        if self._frame_override:
            out.header.frame_id = self._frame_override
        out.angle_min = scan.angle_min
        out.angle_max = scan.angle_max
        out.angle_increment = scan.angle_increment
        out.time_increment = scan.time_increment
        out.scan_time = scan.scan_time
        out.range_min = scan.range_min
        out.range_max = max(float(scan.range_max), 0.01)
        out.ranges = [float(v) if math.isfinite(v) else float('inf')
                      for v in fused]
        out.intensities = list(scan.intensities)
        self._pub.publish(out)

        if self._overlay_pub is not None:
            ov = LaserScan()
            ov.header = out.header
            ov.angle_min = scan.angle_min
            ov.angle_max = scan.angle_max
            ov.angle_increment = scan.angle_increment
            ov.range_min = scan.range_min
            ov.range_max = out.range_max
            ov_sec = overlay[sec_idx]
            ov.ranges = [float(v) if math.isfinite(v) else float('inf')
                         for v in ov_sec]
            self._overlay_pub.publish(ov)

        # ---- occasional note about how much the sonar earned its keep --
        if now - self._last_log > 60.0:
            self._last_log = now
            if self._n_sonar_saves:
                self.get_logger().info(
                    'sonar supplied range in %d sector-samples the lidar '
                    'saw nothing in (last 60 s)' % self._n_sonar_saves)
            self._n_sonar_saves = 0

    # ==================================================================
    def _sector_update(self, z, var, gate=None):
        """
        Vectorised scalar Kalman update across all sectors at once.

        Sectors whose measurement is infinite are skipped: "no return" is not
        a measurement of range, so folding it in as a large number would be a
        lie.  Sectors whose prior is infinite take the measurement directly —
        there is nothing to blend with.
        """
        meas = np.isfinite(z)
        if not np.any(meas):
            return

        fresh = meas & ~np.isfinite(self._sec_r)
        self._sec_r[fresh] = z[fresh]
        self._sec_P[fresh] = var

        both = meas & np.isfinite(self._sec_r) & ~fresh
        if not np.any(both):
            return

        if gate is not None:
            with np.errstate(invalid='ignore'):
                innov = np.abs(z - self._sec_r)
            S = self._sec_P + var
            with np.errstate(invalid='ignore', divide='ignore'):
                d = innov / np.sqrt(np.maximum(S, 1e-12))
            # Reject only measurements that are *further* than expected.  A
            # surprisingly close reading is the one case where being wrong is
            # cheap and being right is everything.
            both = both & ~((d > gate) & (z > self._sec_r))

        if not np.any(both):
            return

        P = self._sec_P[both]
        K = P / (P + var)
        self._sec_r[both] = self._sec_r[both] + K * (z[both] - self._sec_r[both])
        self._sec_P[both] = (1.0 - K) * P

    # ------------------------------------------------------------------
    def _sector_of(self, angles):
        """Map beam bearings to sector indices, wrapping at +/-pi."""
        a = np.mod(angles + math.pi, 2.0 * math.pi)
        idx = (a / (2.0 * math.pi) * self._nsec).astype(np.int64)
        return np.clip(idx, 0, self._nsec - 1)

    def _sector_centres(self):
        step = 2.0 * math.pi / self._nsec
        return (np.arange(self._nsec) + 0.5) * step - math.pi

    def _cone_mask(self, state: SonarState):
        centres = self._sector_centres()
        delta = np.arctan2(np.sin(centres - state.centre),
                           np.cos(centres - state.centre))
        return np.abs(delta) <= (state.fov * 0.5)


def main(args=None):
    rclpy.init(args=args)
    node = ScanFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # rclpy raises this from take_message() while Ctrl-C tears the context
        # down. Noise at shutdown, not a fault - re-raise it any other time.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
