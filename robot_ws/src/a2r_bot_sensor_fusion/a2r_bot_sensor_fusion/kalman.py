#!/usr/bin/env python3
"""
kalman.py
=========
Plain-numpy Kalman filtering.  No filterpy, no scipy — partly to keep the
dependency list on a Raspberry Pi honest, and partly because a filter you
cannot read is a filter you cannot debug at 2 a.m. when the robot is driving
into a wall.

Three things live here:

  KalmanFilter          the linear case, used per-sector in scan fusion
  ExtendedKalmanFilter  the nonlinear case, used for the planar pose
  AttitudeFilter        Madgwick/Mahony complementary attitude estimation

The two Kalman classes are deliberately generic: they know nothing about
robots.  All the robot-specific modelling (what the state means, what the
motion model is) lives in the nodes that use them.

-----------------------------------------------------------------------------
A note on the maths, because the notation varies between textbooks
-----------------------------------------------------------------------------
State-space form used throughout:

    predict:   x⁻ = f(x, u)          P⁻ = F P Fᵀ + Q
    update:    y  = z - h(x⁻)        S  = H P⁻ Hᵀ + R
               K  = P⁻ Hᵀ S⁻¹
               x  = x⁻ + K y         P  = (I - K H) P⁻ (I - K H)ᵀ + K R Kᵀ

The covariance update is written in **Joseph form**.  The textbook shortcut
P = (I - KH)P⁻ is algebraically identical but loses symmetry and positive-
definiteness after a few thousand iterations of float arithmetic, and a filter
whose covariance has gone indefinite produces confident nonsense.  Joseph form
costs one extra matrix multiply and is numerically stable indefinitely.  On a
6-state filter at 50 Hz that is free.
"""

import math

import numpy as np


# =============================================================================
#  Linear Kalman filter
# =============================================================================
class KalmanFilter:
    """
    Standard discrete linear Kalman filter.

    Parameters
    ----------
    dim_x : int   state dimension
    dim_z : int   measurement dimension
    dim_u : int   control dimension (0 for none)
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.dim_u = int(dim_u)

        self.x = np.zeros((self.dim_x, 1))          # state
        self.P = np.eye(self.dim_x)                 # state covariance
        self.Q = np.eye(self.dim_x) * 0.01          # process noise
        self.R = np.eye(self.dim_z)                 # measurement noise
        self.F = np.eye(self.dim_x)                 # state transition
        self.H = np.zeros((self.dim_z, self.dim_x))  # measurement model
        self.B = (np.zeros((self.dim_x, self.dim_u))
                  if self.dim_u else None)          # control model

        self._I = np.eye(self.dim_x)

        # Diagnostics, cheap to keep and invaluable when tuning.
        self.y = np.zeros((self.dim_z, 1))          # last innovation
        self.S = np.eye(self.dim_z)                 # last innovation cov
        self.K = np.zeros((self.dim_x, self.dim_z))  # last gain
        self.mahalanobis = 0.0

    # -- prediction --------------------------------------------------------
    def predict(self, u=None, F=None, Q=None):
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q

        if u is not None and self.B is not None:
            u = np.asarray(u, dtype=float).reshape(self.dim_u, 1)
            self.x = F @ self.x + self.B @ u
        else:
            self.x = F @ self.x

        self.P = F @ self.P @ F.T + Q
        self._symmetrise()
        return self.x

    # -- correction --------------------------------------------------------
    def update(self, z, R=None, H=None):
        """
        Fold in a measurement.  Returns the Mahalanobis distance of the
        innovation, which is what you gate outliers on.
        """
        if z is None:
            return None

        H = self.H if H is None else H
        R = self.R if R is None else R
        z = np.asarray(z, dtype=float).reshape(self.dim_z, 1)

        self.y = z - H @ self.x
        PHT = self.P @ H.T
        self.S = H @ PHT + R

        try:
            S_inv = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            # Singular innovation covariance means R was set to zero
            # somewhere.  Regularise rather than crash the robot.
            S_inv = np.linalg.pinv(self.S + np.eye(self.dim_z) * 1e-9)

        self.K = PHT @ S_inv
        self.x = self.x + self.K @ self.y

        # Joseph form — see module docstring.
        IKH = self._I - self.K @ H
        self.P = IKH @ self.P @ IKH.T + self.K @ R @ self.K.T
        self._symmetrise()

        d2 = float((self.y.T @ S_inv @ self.y).item())
        self.mahalanobis = math.sqrt(max(d2, 0.0))
        return self.mahalanobis

    # -- gated correction --------------------------------------------------
    def update_gated(self, z, gate=3.0, R=None, H=None):
        """
        Chi-square gated update.  Computes what the innovation *would* be,
        rejects the measurement if it is more than `gate` sigma away, and only
        then commits.

        This is the single most valuable line of defence in a real robot.  A
        lidar ray that clips a passing leg, a sonar echo off the ceiling, a
        magnetometer sample taken as the motor kicks in — all produce
        measurements that are individually plausible and collectively
        poisonous.  Gating throws them away without you having to enumerate
        every failure mode in advance.

        Returns (accepted: bool, distance: float).
        """
        H = self.H if H is None else H
        R = self.R if R is None else R
        z = np.asarray(z, dtype=float).reshape(self.dim_z, 1)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, float('inf')

        d = math.sqrt(max(float((y.T @ S_inv @ y).item()), 0.0))
        if d > gate:
            return False, d

        self.update(z, R=R, H=H)
        return True, d

    # -- helpers -----------------------------------------------------------
    def _symmetrise(self):
        """Force P symmetric.  Drift here is pure float error, and it
        compounds; half a microsecond spent now saves a diverged filter."""
        self.P = 0.5 * (self.P + self.P.T)

    def reset(self, x=None, P=None):
        self.x = (np.zeros((self.dim_x, 1)) if x is None
                  else np.asarray(x, dtype=float).reshape(self.dim_x, 1))
        self.P = np.eye(self.dim_x) if P is None else np.asarray(P, float)


# =============================================================================
#  Extended Kalman filter
# =============================================================================
class ExtendedKalmanFilter:
    """
    EKF for nonlinear motion models.

    You supply f(x, u, dt) and its Jacobian F(x, u, dt); measurements are folded
    in with update(), supplying h(x) and H(x) per measurement type.  That last
    part matters: a robot has several sensors with different rates and
    different observation models, and forcing them into one stacked H means the
    filter stalls whenever any one sensor is late.  Per-sensor updates let each
    arrive whenever it likes.
    """

    def __init__(self, dim_x, fx, Fx):
        self.dim_x = int(dim_x)
        self._fx = fx
        self._Fx = Fx

        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x) * 0.1
        self.Q = np.eye(self.dim_x) * 0.01
        self._I = np.eye(self.dim_x)

        self.mahalanobis = 0.0
        self.last_innovation = None

    def predict(self, u, dt, Q=None):
        Q = self.Q if Q is None else Q
        F = np.asarray(self._Fx(self.x, u, dt), dtype=float)
        self.x = np.asarray(self._fx(self.x, u, dt),
                            dtype=float).reshape(self.dim_x, 1)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    def update(self, z, hx, Hx, R, gate=None, angle_indices=()):
        """
        Fold in one measurement.

        Parameters
        ----------
        z             : measurement vector
        hx            : callable(x) -> predicted measurement
        Hx            : callable(x) -> Jacobian dh/dx
        R             : measurement covariance
        gate          : if set, reject innovations beyond this many sigma
        angle_indices : indices of z that are angles and must be wrapped to
                        (-pi, pi] after subtraction.  Forgetting this is the
                        classic EKF bug: the robot crosses from +179 deg to
                        -179 deg, the filter sees a 358 degree innovation, and
                        the estimate snaps violently sideways.
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        dim_z = z.shape[0]

        H = np.asarray(Hx(self.x), dtype=float).reshape(dim_z, self.dim_x)
        R = np.asarray(R, dtype=float).reshape(dim_z, dim_z)

        y = z - np.asarray(hx(self.x), dtype=float).reshape(dim_z, 1)
        for i in angle_indices:
            y[i, 0] = wrap_angle(y[i, 0])

        PHT = self.P @ H.T
        S = H @ PHT + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + np.eye(dim_z) * 1e-9)

        d = math.sqrt(max(float((y.T @ S_inv @ y).item()), 0.0))
        self.mahalanobis = d
        self.last_innovation = y
        if gate is not None and d > gate:
            return False, d

        K = PHT @ S_inv
        self.x = self.x + K @ y

        IKH = self._I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, d


# =============================================================================
#  Attitude filter (Madgwick / Mahony)
# =============================================================================
class AttitudeFilter:
    """
    Quaternion attitude estimation from a 9-DOF IMU.

    Why not just run the attitude through the EKF too?  Because attitude lives
    on a manifold, not in R^n.  A quaternion has four components and three
    degrees of freedom, so an unconstrained Kalman filter over quaternion
    components drifts off the unit sphere and you spend your life renormalising
    and fighting the covariance.  Madgwick's gradient-descent formulation
    handles the constraint natively, costs about 280 floating-point operations
    per update, and has been the default on small robots for fifteen years for
    good reason.

    Two algorithms, selected by `algorithm`:

      'madgwick' : gradient descent on the objective |f(q, a, m)|^2.
                   One tuning knob (beta), which trades gyro drift correction
                   against accelerometer noise rejection.  Good default.
      'mahony'   : PI feedback on the cross-product error.  Two knobs (kp, ki).
                   The integral term actively estimates gyro bias, which
                   matters if your gyro's bias wanders with temperature —
                   and the MPU9250's does.

    Both accept magnetometer data optionally.  With magnetometer, yaw is
    absolute (referenced to magnetic north).  Without, yaw is relative and will
    drift — which is fine, because the EKF downstream gets its yaw reference
    from wheel odometry and scan matching anyway.
    """

    def __init__(self, algorithm='madgwick', beta=0.06, kp=1.0, ki=0.05):
        self.algorithm = algorithm
        self.beta = float(beta)
        self.kp = float(kp)
        self.ki = float(ki)

        self.q = np.array([1.0, 0.0, 0.0, 0.0])      # w, x, y, z
        self._bias = np.zeros(3)                     # Mahony integral term

    # ------------------------------------------------------------------
    def update(self, gyro, accel, mag=None, dt=0.02):
        """
        gyro  : (3,) rad/s, body frame
        accel : (3,) m/s^2, body frame
        mag   : (3,) any consistent unit, body frame, or None
        dt    : seconds
        """
        if dt <= 0.0 or dt > 0.5:
            # A dt of half a second means we stalled; integrating it would
            # inject a huge rotation.  Skip rather than corrupt.
            return self.q

        g = np.asarray(gyro, dtype=float).ravel()[:3]
        a = np.asarray(accel, dtype=float).ravel()[:3]
        m = None if mag is None else np.asarray(mag, dtype=float).ravel()[:3]

        if self.algorithm == 'mahony':
            return self._mahony(g, a, m, dt)
        return self._madgwick(g, a, m, dt)

    # ------------------------------------------------------------------
    def _madgwick(self, g, a, m, dt):
        q0, q1, q2, q3 = self.q

        # Rate of change from the gyro alone.
        qd = 0.5 * np.array([
            -q1 * g[0] - q2 * g[1] - q3 * g[2],
            q0 * g[0] + q2 * g[2] - q3 * g[1],
            q0 * g[1] - q1 * g[2] + q3 * g[0],
            q0 * g[2] + q1 * g[1] - q2 * g[0],
        ])

        an = np.linalg.norm(a)
        if an > 1e-6:
            a = a / an

            if m is not None and np.linalg.norm(m) > 1e-9:
                m = m / np.linalg.norm(m)

                # Reference direction of Earth's magnetic field, rotated into
                # the earth frame and then flattened: we only trust its
                # horizontal component, because the dip angle varies with
                # latitude and is not worth estimating.
                h = _quat_rotate(self.q, m)
                bx = math.hypot(h[0], h[1])
                bz = h[2]

                f = np.array([
                    2 * (q1 * q3 - q0 * q2) - a[0],
                    2 * (q0 * q1 + q2 * q3) - a[1],
                    2 * (0.5 - q1 * q1 - q2 * q2) - a[2],
                    2 * bx * (0.5 - q2 * q2 - q3 * q3)
                    + 2 * bz * (q1 * q3 - q0 * q2) - m[0],
                    2 * bx * (q1 * q2 - q0 * q3)
                    + 2 * bz * (q0 * q1 + q2 * q3) - m[1],
                    2 * bx * (q0 * q2 + q1 * q3)
                    + 2 * bz * (0.5 - q1 * q1 - q2 * q2) - m[2],
                ])

                J = np.array([
                    [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                    [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                    [0.0, -4 * q1, -4 * q2, 0.0],
                    [-2 * bz * q2, 2 * bz * q3,
                     -4 * bx * q2 - 2 * bz * q0,
                     -4 * bx * q3 + 2 * bz * q1],
                    [-2 * bx * q3 + 2 * bz * q1,
                     2 * bx * q2 + 2 * bz * q0,
                     2 * bx * q1 + 2 * bz * q3,
                     -2 * bx * q0 + 2 * bz * q2],
                    [2 * bx * q2, 2 * bx * q3 - 4 * bz * q1,
                     2 * bx * q0 - 4 * bz * q2, 2 * bx * q1],
                ])
            else:
                f = np.array([
                    2 * (q1 * q3 - q0 * q2) - a[0],
                    2 * (q0 * q1 + q2 * q3) - a[1],
                    2 * (0.5 - q1 * q1 - q2 * q2) - a[2],
                ])
                J = np.array([
                    [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                    [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                    [0.0, -4 * q1, -4 * q2, 0.0],
                ])

            step = J.T @ f
            sn = np.linalg.norm(step)
            if sn > 1e-12:
                qd = qd - self.beta * (step / sn)

        self.q = self.q + qd * dt
        self.q = self.q / np.linalg.norm(self.q)
        return self.q

    # ------------------------------------------------------------------
    def _mahony(self, g, a, m, dt):
        e = np.zeros(3)

        an = np.linalg.norm(a)
        if an > 1e-6:
            a = a / an
            # Gravity direction predicted by the current attitude.
            v = _quat_rotate_inv(self.q, np.array([0.0, 0.0, 1.0]))
            e += np.cross(a, v)

        if m is not None and np.linalg.norm(m) > 1e-9:
            m = m / np.linalg.norm(m)
            h = _quat_rotate(self.q, m)
            b = np.array([math.hypot(h[0], h[1]), 0.0, h[2]])
            w = _quat_rotate_inv(self.q, b)
            e += np.cross(m, w)

        if self.ki > 0.0:
            self._bias += e * self.ki * dt
            # Clamp: an unbounded integrator will wind up during the seconds
            # the accelerometer spends saturated over a bump, and then spend
            # a minute unwinding.
            self._bias = np.clip(self._bias, -0.2, 0.2)
            g = g + self._bias

        g = g + self.kp * e

        q0, q1, q2, q3 = self.q
        qd = 0.5 * np.array([
            -q1 * g[0] - q2 * g[1] - q3 * g[2],
            q0 * g[0] + q2 * g[2] - q3 * g[1],
            q0 * g[1] - q1 * g[2] + q3 * g[0],
            q0 * g[2] + q1 * g[1] - q2 * g[0],
        ])

        self.q = self.q + qd * dt
        self.q = self.q / np.linalg.norm(self.q)
        return self.q

    # ------------------------------------------------------------------
    @property
    def euler(self):
        """(roll, pitch, yaw) in radians."""
        return quat_to_euler(self.q)

    @property
    def gyro_bias(self):
        """Mahony's estimated gyro bias, rad/s.  Zero for Madgwick."""
        return -self._bias.copy()

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._bias = np.zeros(3)

    def set_from_euler(self, roll, pitch, yaw):
        self.q = euler_to_quat(roll, pitch, yaw)


# =============================================================================
#  Quaternion / angle helpers
# =============================================================================
def _quat_rotate(q, v):
    """Rotate v from body frame into earth frame."""
    w, x, y, z = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def _quat_rotate_inv(q, v):
    """Rotate v from earth frame into body frame."""
    w, x, y, z = q
    return _quat_rotate([w, -x, -y, -z], v)


def quat_to_euler(q):
    """Quaternion (w,x,y,z) to (roll, pitch, yaw), ZYX convention."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    # Clamp before asin: float error can push this to 1.0000000002 near
    # vertical, and asin then raises.  Gimbal lock is a real orientation, not
    # an error, so we saturate instead of failing.
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat_msg(q):
    """Yaw from a geometry_msgs/Quaternion."""
    return quat_to_euler([q.w, q.x, q.y, q.z])[2]
