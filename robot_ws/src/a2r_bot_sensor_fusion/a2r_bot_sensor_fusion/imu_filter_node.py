#!/usr/bin/env python3
"""
imu_filter_node.py
==================
Turns the raw GY-91 stream into an oriented IMU message.

The Arduino publishes /imu/data_raw with angular velocity and linear
acceleration but an all-zero orientation quaternion, because the firmware
deliberately does no attitude estimation — an 8-bit-era microcontroller
budget is better spent on deterministic PWM timing than on quaternion
algebra, and the Pi has floating point to spare.

So the attitude solution lives here, and /imu/data comes out with a real
orientation, ready for the EKF and for RViz.

Subscribes
----------
/imu/data_raw   sensor_msgs/Imu    gyro + accel, orientation unset
/imu/mag        sensor_msgs/MagneticField
/imu/calibrate  std_msgs/Bool      trigger a bias re-estimate

Publishes
---------
/imu/data       sensor_msgs/Imu    with orientation and covariances
/imu/yaw        std_msgs/Float32    convenience, degrees, for debugging
/imu/rpy        geometry_msgs/Vector3Stamped  roll/pitch/yaw radians
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool, Float32

from a2r_bot_sensor_fusion.kalman import AttitudeFilter, wrap_angle


class ImuFilterNode(Node):
    """Madgwick/Mahony attitude estimation with optional magnetometer."""

    def __init__(self):
        super().__init__('imu_filter_node')

        # ---- parameters ----------------------------------------------
        self.declare_parameter('algorithm', 'madgwick')
        self.declare_parameter('beta', 0.06)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.05)
        self.declare_parameter('use_mag', True)
        self.declare_parameter('mag_declination_deg', 0.0)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate_hz', 0.0)
        self.declare_parameter('gyro_bias_samples', 300)
        self.declare_parameter('auto_calibrate_on_start', True)
        self.declare_parameter('stationary_accel_tol', 0.35)
        self.declare_parameter('stationary_gyro_tol', 0.05)
        self.declare_parameter('mag_hard_iron', [0.0, 0.0, 0.0])
        self.declare_parameter('mag_soft_iron_scale', [1.0, 1.0, 1.0])
        self.declare_parameter('orientation_variance', 2.5e-3)
        self.declare_parameter('gyro_variance', 4.0e-4)
        self.declare_parameter('accel_variance', 8.0e-3)
        self.declare_parameter('max_dt_s', 0.2)
        self.declare_parameter('gyro_z_sign', 1.0)   # a2r_yaw_fix: -1 = gyro z mounted/sent inverted

        g = self.get_parameter
        self._algo = str(g('algorithm').value).lower()
        self._use_mag = bool(g('use_mag').value)
        self._decl = math.radians(float(g('mag_declination_deg').value))
        self._frame = str(g('frame_id').value)
        self._bias_n = int(g('gyro_bias_samples').value)
        self._auto_cal = bool(g('auto_calibrate_on_start').value)
        self._acc_tol = float(g('stationary_accel_tol').value)
        self._gyr_tol = float(g('stationary_gyro_tol').value)
        self._hard = np.array(list(g('mag_hard_iron').value), dtype=float)
        self._soft = np.array(list(g('mag_soft_iron_scale').value),
                              dtype=float)
        self._ori_var = float(g('orientation_variance').value)
        self._gyr_var = float(g('gyro_variance').value)
        self._acc_var = float(g('accel_variance').value)
        self._max_dt = float(g('max_dt_s').value)

        self._filter = AttitudeFilter(
            algorithm=self._algo,
            beta=float(g('beta').value),
            kp=float(g('kp').value),
            ki=float(g('ki').value),
        )

        # ---- state ----------------------------------------------------
        self._mag = None
        self._mag_stamp = 0.0
        self._last_t = None
        self._bias = np.zeros(3)
        self._bias_acc = []
        self._calibrating = bool(self._auto_cal)
        self._seeded = False
        self._n_msgs = 0
        self._last_warn = 0.0

        # ---- interfaces -----------------------------------------------
        self._imu_pub = self.create_publisher(Imu, '/imu/data',
                                              10)   # reliable: any subscriber can receive it
        self._rpy_pub = self.create_publisher(Vector3Stamped, '/imu/rpy', 10)
        self._yaw_pub = self.create_publisher(Float32, '/imu/yaw', 10)

        self.create_subscription(Imu, '/imu/data_raw', self._imu_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(MagneticField, '/imu/mag', self._mag_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Bool, '/imu/calibrate', self._cal_cb, 10)

        self.get_logger().info(
            'imu_filter_node up — %s, mag %s, %s'
            % (self._algo,
               'on' if self._use_mag else 'off',
               'calibrating gyro bias' if self._calibrating else 'no cal'))

    # ------------------------------------------------------------------
    def _cal_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info('gyro bias re-calibration requested')
            self._bias_acc = []
            self._calibrating = True

    def _mag_cb(self, msg: MagneticField):
        m = np.array([msg.magnetic_field.x,
                      msg.magnetic_field.y,
                      msg.magnetic_field.z], dtype=float)
        # Hard-iron offset first (translation), then soft-iron scale.  Doing
        # it the other way round is a common and subtle error: the scale must
        # act on the centred ellipsoid, not the offset one.
        self._mag = (m - self._hard) * self._soft
        self._mag_stamp = time.monotonic()

    # ------------------------------------------------------------------
    def _imu_cb(self, msg: Imu):
        now = time.monotonic()
        self._n_msgs += 1

        gyro = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z * float(self.get_parameter('gyro_z_sign').value)],
                        dtype=float)   # a2r_yaw_fix
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z], dtype=float)

        # ---- dt ------------------------------------------------------
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return
        if dt > self._max_dt:
            # The IMU stream stalled.  Integrating the gap would rotate the
            # estimate by however far the robot turned while we were blind,
            # in one step, with full confidence.  Better to skip and let the
            # accelerometer pull us back over the next few samples.
            if now - self._last_warn > 5.0:
                self._last_warn = now
                self.get_logger().warn(
                    'IMU gap of %.0f ms — skipping integration' % (dt * 1e3))
            return

        # ---- gyro bias estimation ------------------------------------
        if self._calibrating:
            still = (abs(np.linalg.norm(accel) - 9.80665) < self._acc_tol
                     and np.linalg.norm(gyro) < self._gyr_tol)
            if still:
                self._bias_acc.append(gyro)
                if len(self._bias_acc) >= self._bias_n:
                    self._bias = np.mean(self._bias_acc, axis=0)
                    self._calibrating = False
                    self._bias_acc = []
                    self.get_logger().info(
                        'gyro bias: [%.5f %.5f %.5f] rad/s'
                        % tuple(self._bias))
            else:
                # Motion partway through invalidates the average; start over
                # rather than bake the motion into the bias.
                if self._bias_acc:
                    self._bias_acc = []
            return

        gyro = gyro - self._bias

        # ---- seed attitude from gravity on the first good sample ------
        # Without this the filter starts level and spends several seconds
        # swinging to the true attitude, which looks alarming in RViz and
        # briefly poisons the EKF.
        if not self._seeded and np.linalg.norm(accel) > 1.0:
            ax, ay, az = accel / np.linalg.norm(accel)
            roll = math.atan2(ay, az)
            pitch = math.atan2(-ax, math.hypot(ay, az))
            yaw = 0.0
            if self._use_mag and self._mag is not None:
                yaw = self._tilt_compensated_yaw(roll, pitch, self._mag)
            self._filter.set_from_euler(roll, pitch, yaw)
            self._seeded = True
            self.get_logger().info(
                'attitude seeded: roll %.1f deg, pitch %.1f deg, yaw %.1f deg'
                % (math.degrees(roll), math.degrees(pitch),
                   math.degrees(yaw)))

        # ---- run the filter ------------------------------------------
        mag = None
        if self._use_mag and self._mag is not None:
            if (now - self._mag_stamp) < 0.5:
                mag = self._mag

        self._filter.update(gyro, accel, mag, dt)
        roll, pitch, yaw = self._filter.euler

        # Magnetic declination converts magnetic north to true north.  Only
        # meaningful when the magnetometer is actually in use.
        if mag is not None and self._decl != 0.0:
            yaw = wrap_angle(yaw + self._decl)

        self._publish(msg, gyro, accel, roll, pitch, yaw)

    # ------------------------------------------------------------------
    @staticmethod
    def _tilt_compensated_yaw(roll, pitch, mag):
        """
        Heading from a magnetometer, corrected for the fact that the sensor
        is not level.  Using atan2(my, mx) directly is only right when the
        robot is perfectly flat; on a 10-degree slope it is wrong by tens of
        degrees, which is exactly when you most want a heading.
        """
        mx, my, mz = mag / (np.linalg.norm(mag) + 1e-12)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        xh = mx * cp + mz * sp
        yh = mx * sr * sp + my * cr - mz * sr * cp
        return math.atan2(-yh, xh)

    # ------------------------------------------------------------------
    def _publish(self, src, gyro, accel, roll, pitch, yaw):
        q = self._filter.q
        out = Imu()
        out.header.stamp = (src.header.stamp
                            if src.header.stamp.sec
                            else self.get_clock().now().to_msg())
        out.header.frame_id = self._frame

        out.orientation.w = float(q[0])
        out.orientation.x = float(q[1])
        out.orientation.y = float(q[2])
        out.orientation.z = float(q[3])

        out.angular_velocity.x = float(gyro[0])
        out.angular_velocity.y = float(gyro[1])
        out.angular_velocity.z = float(gyro[2])
        out.linear_acceleration.x = float(accel[0])
        out.linear_acceleration.y = float(accel[1])
        out.linear_acceleration.z = float(accel[2])

        # Yaw is the least trustworthy axis: without a magnetometer it is
        # unobservable and drifts, so we advertise a much larger variance on
        # it.  The EKF reads these numbers and will weight yaw accordingly —
        # lying here is how you get a filter that confidently follows a
        # drifting gyro.
        yaw_var = self._ori_var if (self._use_mag and self._mag is not None) \
            else self._ori_var * 50.0
        out.orientation_covariance = [
            self._ori_var, 0.0, 0.0,
            0.0, self._ori_var, 0.0,
            0.0, 0.0, yaw_var,
        ]
        out.angular_velocity_covariance = [
            self._gyr_var, 0.0, 0.0,
            0.0, self._gyr_var, 0.0,
            0.0, 0.0, self._gyr_var,
        ]
        out.linear_acceleration_covariance = [
            self._acc_var, 0.0, 0.0,
            0.0, self._acc_var, 0.0,
            0.0, 0.0, self._acc_var,
        ]
        self._imu_pub.publish(out)

        rpy = Vector3Stamped()
        rpy.header = out.header
        rpy.vector.x = float(roll)
        rpy.vector.y = float(pitch)
        rpy.vector.z = float(yaw)
        self._rpy_pub.publish(rpy)

        f = Float32()
        f.data = float(math.degrees(yaw))
        self._yaw_pub.publish(f)


def main(args=None):
    rclpy.init(args=args)
    node = ImuFilterNode()
    try:
        import threading     # v2.3: tf2 listener thread - ignore the Ctrl-C shutdown race
        _hook = threading.excepthook
        threading.excepthook = lambda a, _h=_hook: None if a.exc_type.__name__ == "ExternalShutdownException" else _h(a)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:        # v2.3: Ctrl-C race - context closed while a message was taken
        if rclpy.ok():
            raise            # a real error while running is still reported
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
