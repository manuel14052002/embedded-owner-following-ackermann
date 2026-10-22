#!/usr/bin/env python3
"""
ekf_state_node.py
=================
The planar state estimator.  Everything that knows where the robot is gets it
from here.

State vector (6):

    x = [ px, py, theta, v, omega, a ]^T

      px, py   position in the odom frame, metres
      theta    heading, radians
      v        body-forward linear velocity, m/s
      omega    yaw rate, rad/s
      a        body-forward linear acceleration, m/s^2

Why carry acceleration in the state?  Because the car has no wheel encoders on
the drive shaft that we trust, so forward speed is only weakly observed.  The
accelerometer observes its derivative directly.  Including `a` lets the filter
integrate the accelerometer into the speed estimate instead of throwing it
away, which is the difference between knowing your speed to 0.3 m/s and
knowing it to 0.05 m/s.  It also gives the filter something sensible to do
during the half-second after you command a stop, when the commanded velocity
says zero and the car is very much still moving.

Motion model (nonlinear, hence the EKF):

    px'    = px + v cos(theta) dt
    py'    = py + v sin(theta) dt
    theta' = theta + omega dt
    v'     = v + a dt
    omega' = omega            (random walk)
    a'     = a * decay        (see below)

The acceleration decay is deliberate.  A constant-acceleration model applied to
a robot that has stopped accelerating will happily predict it accelerating
forever.  Bleeding `a` towards zero with a time constant encodes "accelerations
are transient", which for a following robot is true.

Measurement sources, each folded in separately as it arrives:

    IMU yaw          from /imu/data    (absolute if mag is on, else relative)
    IMU yaw rate     from /imu/data    gyro z, the most reliable signal we have
    IMU accel x      from /imu/data    gravity-compensated body-forward accel
    command velocity from the drive command — a weak pseudo-measurement
    scan match       from /scan_match/odom if the SLAM node is running

Publishes
---------
/odometry/filtered   nav_msgs/Odometry
odom -> base_footprint TF (optional, on by default)

Subscribes
----------
/imu/data, /local_planner/cmd_out (or /cmd_vel), /scan_match/odom,
/ekf/reset (std_msgs/Bool)
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from tf2_ros import TransformBroadcaster

from a2r_bot_sensor_fusion.kalman import (ExtendedKalmanFilter, euler_to_quat,
                                          wrap_angle, yaw_from_quat_msg)

PX, PY, TH, V, W, A = range(6)


class EkfStateNode(Node):
    """Six-state extended Kalman filter for the planar pose."""

    def __init__(self):
        super().__init__('ekf_state_node')

        # ---- parameters ----------------------------------------------
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('command_topic', '/local_planner/cmd_out')
        self.declare_parameter('command_is_normalised', True)
        self.declare_parameter('wheelbase_m', 0.280)
        self.declare_parameter('max_speed_mps', 1.2)
        self.declare_parameter('max_steer_deg', 30.0)
        self.declare_parameter('steer_sign', 1.0)    # a2r_yaw_fix: -1 = command steering sign inverted

        self.declare_parameter('use_imu_yaw', True)
        self.declare_parameter('use_imu_yaw_rate', True)
        self.declare_parameter('use_imu_accel', True)
        self.declare_parameter('use_command', True)
        self.declare_parameter('use_scan_match', True)

        self.declare_parameter('accel_decay_tau_s', 0.35)
        self.declare_parameter('command_lag_s', 0.12)

        # Process noise, per second.  These are the numbers you actually tune.
        self.declare_parameter('q_position', 1.0e-4)
        self.declare_parameter('q_heading', 2.0e-4)
        self.declare_parameter('q_velocity', 6.0e-3)
        self.declare_parameter('q_yaw_rate', 3.0e-2)
        self.declare_parameter('q_accel', 5.0e-1)

        # Measurement noise.
        self.declare_parameter('r_imu_yaw', 2.5e-3)
        self.declare_parameter('r_imu_yaw_rate', 4.0e-4)
        self.declare_parameter('r_imu_accel', 2.0e-1)
        self.declare_parameter('r_command_v', 4.0e-2)
        self.declare_parameter('r_command_w', 8.0e-2)
        self.declare_parameter('r_scan_xy', 4.0e-3)
        self.declare_parameter('r_scan_yaw', 2.0e-3)

        self.declare_parameter('gate_sigma', 4.0)
        self.declare_parameter('imu_timeout_s', 0.5)
        self.declare_parameter('zero_velocity_snap', True)
        self.declare_parameter('zero_velocity_speed_eps', 0.02)

        g = self.get_parameter
        self._rate = float(g('rate_hz').value)
        self._odom_frame = str(g('odom_frame').value)
        self._base_frame = str(g('base_frame').value)
        self._publish_tf = bool(g('publish_tf').value)
        self._cmd_normalised = bool(g('command_is_normalised').value)
        self._L = float(g('wheelbase_m').value)
        self._vmax = float(g('max_speed_mps').value)
        self._smax = math.radians(float(g('max_steer_deg').value))

        self._use_imu_yaw = bool(g('use_imu_yaw').value)
        self._use_imu_rate = bool(g('use_imu_yaw_rate').value)
        self._use_imu_acc = bool(g('use_imu_accel').value)
        self._use_cmd = bool(g('use_command').value)
        self._use_scan = bool(g('use_scan_match').value)

        self._tau = max(1e-3, float(g('accel_decay_tau_s').value))
        self._cmd_lag = float(g('command_lag_s').value)
        self._gate = float(g('gate_sigma').value)
        self._imu_timeout = float(g('imu_timeout_s').value)
        self._zvu = bool(g('zero_velocity_snap').value)
        self._zvu_eps = float(g('zero_velocity_speed_eps').value)

        self._r_yaw = float(g('r_imu_yaw').value)
        self._r_rate = float(g('r_imu_yaw_rate').value)
        self._r_acc = float(g('r_imu_accel').value)
        self._r_cv = float(g('r_command_v').value)
        self._r_cw = float(g('r_command_w').value)
        self._r_sxy = float(g('r_scan_xy').value)
        self._r_syaw = float(g('r_scan_yaw').value)

        self._q_diag = np.array([
            float(g('q_position').value),
            float(g('q_position').value),
            float(g('q_heading').value),
            float(g('q_velocity').value),
            float(g('q_yaw_rate').value),
            float(g('q_accel').value),
        ])

        # ---- filter ---------------------------------------------------
        self._ekf = ExtendedKalmanFilter(6, self._fx, self._Fx)
        self._ekf.P = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-1])

        # ---- state ----------------------------------------------------
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._cmd_t = 0.0
        self._imu_t = 0.0
        self._last_predict = time.monotonic()
        self._yaw_offset = None      # first IMU yaw becomes our zero
        self._imu_yaw = 0.0
        self._imu_rate = 0.0
        self._imu_accel = 0.0
        self._have_imu = False
        self._rejects = 0
        self._last_report = time.monotonic()

        # ---- interfaces -----------------------------------------------
        self._odom_pub = self.create_publisher(Odometry,
                                               '/odometry/filtered', 20)
        self._tf = TransformBroadcaster(self) if self._publish_tf else None

        self.create_subscription(Imu, '/imu/data', self._imu_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Twist, str(g('command_topic').value),
                                 self._cmd_cb, 10)
        self.create_subscription(Odometry, '/scan_match/odom',
                                 self._scan_cb, 10)
        self.create_subscription(Bool, '/ekf/reset', self._reset_cb, 10)

        self.create_timer(1.0 / self._rate, self._tick)
        self.get_logger().info(
            'ekf_state_node up — %.0f Hz, sources:%s%s%s%s%s'
            % (self._rate,
               ' imu_yaw' if self._use_imu_yaw else '',
               ' imu_rate' if self._use_imu_rate else '',
               ' imu_accel' if self._use_imu_acc else '',
               ' command' if self._use_cmd else '',
               ' scan_match' if self._use_scan else ''))

    # ==================================================================
    #  Motion model
    # ==================================================================
    def _fx(self, x, u, dt):
        px, py, th, v, w, a = x.ravel()
        decay = math.exp(-dt / self._tau)
        return np.array([
            [px + v * math.cos(th) * dt],
            [py + v * math.sin(th) * dt],
            [wrap_angle(th + w * dt)],
            [v + a * dt],
            [w],
            [a * decay],
        ])

    def _Fx(self, x, u, dt):
        _, _, th, v, _, _ = x.ravel()
        decay = math.exp(-dt / self._tau)
        F = np.eye(6)
        F[PX, TH] = -v * math.sin(th) * dt
        F[PX, V] = math.cos(th) * dt
        F[PY, TH] = v * math.cos(th) * dt
        F[PY, V] = math.sin(th) * dt
        F[TH, W] = dt
        F[V, A] = dt
        F[A, A] = decay
        return F

    # ==================================================================
    #  Callbacks
    # ==================================================================
    def _reset_cb(self, msg: Bool):
        if not msg.data:
            return
        self._ekf.x = np.zeros((6, 1))
        self._ekf.P = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-1])
        self._yaw_offset = None
        self.get_logger().info('EKF reset to origin')

    def _imu_cb(self, msg: Imu):
        self._imu_t = time.monotonic()
        self._have_imu = True

        yaw = yaw_from_quat_msg(msg.orientation)
        if self._yaw_offset is None:
            # Anchor the odom frame to wherever the robot was pointing at
            # startup, the way wheel odometry would.  Publishing raw magnetic
            # heading as odom yaw would make the odom frame jump every time
            # you power-cycle, which breaks every saved map.
            self._yaw_offset = yaw
            self._ekf.x[TH, 0] = 0.0
        self._imu_yaw = wrap_angle(yaw - self._yaw_offset)

        self._imu_rate = float(msg.angular_velocity.z)
        self._imu_new = True        # a2r_step2: fuse each gyro sample once, never a stale one again

        # Body-forward acceleration with gravity removed.  On a flat floor
        # linear_acceleration.x is already the forward accel, but the car
        # pitches under braking and on ramps, and then gravity leaks into x
        # at g*sin(pitch) — 1.7 m/s^2 on a 10 degree slope, which is more
        # than the car can actually produce.  So we subtract it.
        q = msg.orientation
        # pitch from the quaternion, clamped for the near-vertical case
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        self._imu_accel = float(msg.linear_acceleration.x) \
            + 9.80665 * math.sin(pitch)

    def _cmd_cb(self, msg: Twist):
        self._cmd_t = time.monotonic()
        if self._cmd_normalised:
            # Normalised throttle/steer from the local planner.
            v = float(msg.linear.x) * self._vmax
            steer = float(msg.angular.z) * self._smax
        else:
            v = float(msg.linear.x)
            # Twist.angular.z in real units is a yaw rate; convert to the
            # steering angle the Ackermann model needs.
            steer = (math.atan2(float(msg.angular.z) * self._L, abs(v))
                     if abs(v) > 1e-3 else 0.0)
        steer *= float(self.get_parameter('steer_sign').value)   # a2r_yaw_fix
        self._cmd_v = v
        self._cmd_w = (v / self._L) * math.tan(steer) if abs(v) > 1e-6 else 0.0

    def _scan_cb(self, msg: Odometry):
        if not self._use_scan:
            return
        z = np.array([[msg.pose.pose.position.x],
                      [msg.pose.pose.position.y],
                      [yaw_from_quat_msg(msg.pose.pose.orientation)]])

        # Trust the covariance the scan matcher actually reports, if it
        # bothered to fill it in; fall back to the parameter otherwise.
        cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        rxx = cov[0, 0] if cov[0, 0] > 1e-9 else self._r_sxy
        ryy = cov[1, 1] if cov[1, 1] > 1e-9 else self._r_sxy
        rtt = cov[5, 5] if cov[5, 5] > 1e-9 else self._r_syaw
        R = np.diag([rxx, ryy, rtt])

        def h(x):
            return np.array([[x[PX, 0]], [x[PY, 0]], [x[TH, 0]]])

        def H(_x):
            m = np.zeros((3, 6))
            m[0, PX] = 1.0
            m[1, PY] = 1.0
            m[2, TH] = 1.0
            return m

        ok, d = self._ekf.update(z, h, H, R, gate=self._gate,
                                 angle_indices=(2,))
        if not ok:
            self._rejects += 1

    # ==================================================================
    #  Main loop
    # ==================================================================
    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_predict
        self._last_predict = now
        if dt <= 0.0 or dt > 0.5:
            return

        # ---- predict --------------------------------------------------
        Q = np.diag(self._q_diag * dt)
        self._ekf.predict(None, dt, Q=Q)
        self._ekf.x[TH, 0] = wrap_angle(self._ekf.x[TH, 0])
        # v2.7: physical bounds - this car cannot do 50 m/s (what the odom
        # believed), so the estimate may never leave a sane range
        self._ekf.x[V, 0] = max(-1.5, min(1.5, self._ekf.x[V, 0]))
        self._ekf.x[A, 0] = max(-4.0, min(4.0, self._ekf.x[A, 0]))

        imu_fresh = self._have_imu and (now - self._imu_t) < self._imu_timeout

        # ---- IMU yaw --------------------------------------------------
        if self._use_imu_yaw and imu_fresh:
            self._scalar_update(self._imu_yaw, TH, self._r_yaw, angle=True)

        # ---- IMU yaw rate ---------------------------------------------
        # This is the single most valuable measurement on the robot.  A MEMS
        # gyro's yaw rate is accurate to a fraction of a degree per second,
        # which is far better than anything the steering command can tell us —
        # the command says what we asked for, the gyro says what happened.
        if self._use_imu_rate and imu_fresh and getattr(self, '_imu_new', True):
            self._imu_new = False   # a2r_step2
            self._gyro_update()

        # ---- IMU forward acceleration ---------------------------------
        if self._use_imu_acc and imu_fresh:
            self._scalar_update(self._imu_accel, A, self._r_acc)

        # ---- commanded velocity as a weak pseudo-measurement ----------
        # The command is not a measurement of anything; it is an intention.
        # But on a robot with no drive encoder it is the only thing that
        # observes forward speed at all, so we fold it in with a deliberately
        # large R.  The accelerometer then corrects it.
        if self._use_cmd:
            age = now - self._cmd_t
            if age < 0.5:
                # Account for actuator lag: what the car is doing now is what
                # we asked for roughly `command_lag_s` ago.  Ignoring this
                # biases the speed estimate high during acceleration.
                fade = 1.0 if age > self._cmd_lag else 0.5
                # v2.7: the command is the ONLY speed observation (no encoders).
                # Gating it let one disturbance lock the speed out forever -
                # log: "speed 1462" rejections/30 s, odom ran 90 m in 1.9 s.
                self._scalar_update(self._cmd_v, V, self._r_cv / fade, gated=False)
                # a2r_step2: the command yaw rate assumes the wheels are ALREADY at the commanded lock;
                # in a K-turn they are not, and it pulled the estimate the wrong way. Gyro present -> skip.
                if not (self._use_imu_rate and imu_fresh):
                    self._scalar_update(self._cmd_w, W, self._r_cw / fade)
            else:
                # No command for half a second: the failsafe has stopped us.
                self._scalar_update(0.0, V, self._r_cv, gated=False)

        # ---- zero-velocity update -------------------------------------
        # When the gyro is quiet and no command is active, the car is
        # stationary.  Saying so explicitly collapses the velocity variance
        # and stops the position from wandering while parked — the single
        # cheapest improvement you can make to a dead-reckoning filter.
        if self._zvu:
            quiet = (abs(self._imu_rate) < 0.02
                     and abs(self._cmd_v) < self._zvu_eps
                     and (now - self._cmd_t) > 0.3)
            if quiet:
                self._scalar_update(0.0, V, 1e-4, gated=False)
                self._scalar_update(0.0, W, 1e-4)
                self._scalar_update(0.0, A, 1e-3)

        self._publish(now)

        # ---- periodic health note -------------------------------------
        if now - self._last_report > 30.0:
            self._last_report = now
            if self._rejects:
                self.get_logger().info(
                    'EKF gated out %d measurement(s) in the last 30 s (%s)'
                    % (self._rejects, ', '.join(
                        '%s %d' % (('x', 'y', 'yaw', 'speed', 'yaw_rate', 'accel')[i], k)
                        for i, k in sorted(self.__dict__.get('_rej_by', {}).items())) or 'scan match'))
            self._rejects = 0
            self.__dict__['_rej_by'] = {}
            if not imu_fresh:
                self.get_logger().warn(
                    'no IMU for %.1f s — running on commands alone'
                    % (now - self._imu_t))

    def _gyro_update(self):
        """a2r_step2: gyro yaw rate with lock-out recovery. A K-turn changes the yaw rate faster than
        the old process noise allowed, the gyro was gated out, the stale state drifted further from
        it and every following sample was gated too (logs: 41-75 yaw_rate rejections per 30 s).
        After 3 rejections in a row the gyro is believed again, with R widened just enough to sit
        on the gate. A single spike is still rejected."""
        if self._scalar_update(self._imu_rate, W, self._r_rate):
            self._w_rej = 0
            return
        self._w_rej = getattr(self, '_w_rej', 0) + 1
        if self._w_rej >= 3:
            y = self._imu_rate - float(self._ekf.x[W, 0])
            r = max(self._r_rate, y * y / (self._gate * self._gate) - float(self._ekf.P[W, W]))
            self._scalar_update(self._imu_rate, W, r, gated=False)
            self._w_rej = 0
            self._w_rescued = getattr(self, '_w_rescued', 0) + 1

    # ------------------------------------------------------------------
    def _scalar_update(self, z, index, r, angle=False, gated=True):
        """Fold in a single scalar that observes one state directly."""
        def h(x):
            return np.array([[x[index, 0]]])

        def H(_x):
            m = np.zeros((1, 6))
            m[0, index] = 1.0
            return m

        ok, _ = self._ekf.update(
            np.array([[float(z)]]), h, H, np.array([[float(r)]]),
            gate=self._gate if gated else None, angle_indices=(0,) if angle else ())
        if not ok:
            self._rejects += 1
            rb = self.__dict__.setdefault("_rej_by", {})     # v2.5: which state is rejected
            rb[index] = rb.get(index, 0) + 1
        return ok                   # a2r_step2

    # ------------------------------------------------------------------
    def _publish(self, _now):
        x = self._ekf.x.ravel()
        P = self._ekf.P
        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame

        odom.pose.pose.position.x = float(x[PX])
        odom.pose.pose.position.y = float(x[PY])
        odom.pose.pose.position.z = 0.0
        q = euler_to_quat(0.0, 0.0, float(x[TH]))
        odom.pose.pose.orientation.w = float(q[0])
        odom.pose.pose.orientation.x = float(q[1])
        odom.pose.pose.orientation.y = float(q[2])
        odom.pose.pose.orientation.z = float(q[3])

        odom.twist.twist.linear.x = float(x[V])
        odom.twist.twist.angular.z = float(x[W])

        # Map the 6x6 filter covariance into the 6x6 ROS pose covariance,
        # which is ordered [x y z roll pitch yaw].  Everything the filter does
        # not estimate gets a large number rather than zero: zero means
        # "perfectly known", and claiming perfect knowledge of z is how you
        # get downstream consumers doing strange things.
        pc = np.zeros((6, 6))
        pc[0, 0] = P[PX, PX]
        pc[0, 1] = P[PX, PY]
        pc[1, 0] = P[PY, PX]
        pc[1, 1] = P[PY, PY]
        pc[0, 5] = P[PX, TH]
        pc[5, 0] = P[TH, PX]
        pc[1, 5] = P[PY, TH]
        pc[5, 1] = P[TH, PY]
        pc[5, 5] = P[TH, TH]
        pc[2, 2] = 1e6
        pc[3, 3] = 1e6
        pc[4, 4] = 1e6
        odom.pose.covariance = [float(v) for v in pc.ravel()]

        tc = np.zeros((6, 6))
        tc[0, 0] = P[V, V]
        tc[5, 5] = P[W, W]
        tc[0, 5] = P[V, W]
        tc[5, 0] = P[W, V]
        tc[1, 1] = 1e6
        tc[2, 2] = 1e6
        tc[3, 3] = 1e6
        tc[4, 4] = 1e6
        odom.twist.covariance = [float(v) for v in tc.ravel()]

        self._odom_pub.publish(odom)

        if self._tf is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self._odom_frame
            t.child_frame_id = self._base_frame
            t.transform.translation.x = float(x[PX])
            t.transform.translation.y = float(x[PY])
            t.transform.translation.z = 0.0
            t.transform.rotation = odom.pose.pose.orientation
            self._tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = EkfStateNode()
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
