#!/usr/bin/env python3
"""
firmware_monitor_node.py
========================
A small watchdog that turns the firmware's telemetry into a single yes/no
answer: is the low-level board healthy right now?

The serial bridge already publishes everything on individual topics, but every
consumer that wants to know "can I drive?" would otherwise have to subscribe
to five topics and implement its own staleness logic.  That duplication is how
you end up with three slightly different definitions of "healthy" in one
robot.  So it lives here, once.

Publishes
---------
/arduino/healthy      std_msgs/Bool     latched, True while everything is fresh
/arduino/health_text  std_msgs/String   one-line human summary
/diagnostics          diagnostic_msgs/DiagnosticArray

Subscribes
----------
/imu/data_raw, /ultrasonic/front, /ultrasonic/rear, /battery,
/arduino/status, /arduino/failsafe
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Imu, Range, BatteryState
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class FirmwareMonitorNode(Node):
    """Aggregate firmware telemetry into one health signal."""

    def __init__(self):
        super().__init__('firmware_monitor_node')

        self.declare_parameter('stale_after_s', 3.0)
        self.declare_parameter('log_period_s', 10.0)
        self.declare_parameter('battery_warn_percent', 25.0)
        self.declare_parameter('battery_error_percent', 12.0)
        self.declare_parameter('imu_stale_after_s', 1.0)
        self.declare_parameter('range_stale_after_s', 1.5)
        # Which sonars are actually fitted. A sensor that was deliberately
        # removed must not pin /arduino/healthy to false forever - that would
        # make every consumer of it permanently distrust a robot that is fine,
        # and the pet face would sit on "sad" for the rest of its life.
        self.declare_parameter('expect_front_sonar', True)
        self.declare_parameter('expect_rear_sonar', True)
        # /ultrasonic/* keeps arriving at 10 Hz even when the sensor is dead:
        # the firmware sends -1 ("no valid reading") and the bridge turns it
        # into +inf. So message age alone cannot catch a miswired sonar. This
        # warns (without marking the board unhealthy) when a fitted sonar has
        # produced no finite echo for this long.
        self.declare_parameter('sonar_no_echo_warn_s', 10.0)

        g = self.get_parameter
        self._stale = float(g('stale_after_s').value)
        self._log_period = float(g('log_period_s').value)
        self._bat_warn = float(g('battery_warn_percent').value)
        self._bat_err = float(g('battery_error_percent').value)
        self._imu_stale = float(g('imu_stale_after_s').value)
        self._rng_stale = float(g('range_stale_after_s').value)
        self._want_front = bool(g('expect_front_sonar').value)
        self._want_rear = bool(g('expect_rear_sonar').value)

        # ---- last-seen timestamps ------------------------------------
        self._startup = time.monotonic()
        self._t_imu = time.monotonic()
        self._t_front = time.monotonic()
        self._t_rear = time.monotonic()
        self._t_batt = time.monotonic()
        self._t_status = 0.0

        self._batt_pct = float('nan')
        self._batt_v = float('nan')
        self._failsafe = False
        self._status_text = 'no status yet'
        self._last_log = 0.0
        self._last_healthy = None

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._healthy_pub = self.create_publisher(
            Bool, '/arduino/healthy', latched)
        self._text_pub = self.create_publisher(
            String, '/arduino/health_text', latched)
        self._diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 10)

        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/imu/data_raw', self._imu_cb, sensor_qos)
        self.create_subscription(Range, '/ultrasonic/front',
                                 self._front_cb, sensor_qos)
        self.create_subscription(Range, '/ultrasonic/rear',
                                 self._rear_cb, sensor_qos)
        self.create_subscription(BatteryState, '/battery', self._batt_cb, 10)
        self.create_subscription(String, '/arduino/status',
                                 self._status_cb, 10)
        self.create_subscription(Bool, '/arduino/failsafe',
                                 self._failsafe_cb, 10)

        self.create_timer(0.5, self._tick)
        fitted = []
        if self._want_front:
            fitted.append('front')
        if self._want_rear:
            fitted.append('rear')
        self.get_logger().info(
            'firmware_monitor_node up - sonars expected: %s'
            % (', '.join(fitted) if fitted else 'none'))

    # ------------------------------------------------------------------
    # Callbacks — each one only stamps a clock; no work in the hot path.
    # ------------------------------------------------------------------
    def _imu_cb(self, _msg):
        self._t_imu = time.monotonic()

    def _front_cb(self, msg: Range):
        self._t_front = time.monotonic()
        if math.isfinite(msg.range):
            self._t_front_echo = self._t_front

    def _rear_cb(self, msg: Range):
        self._t_rear = time.monotonic()
        if math.isfinite(msg.range):
            self._t_rear_echo = self._t_rear

    def _batt_cb(self, msg: BatteryState):
        self._t_batt = time.monotonic()
        self._batt_v = float(msg.voltage)
        # BatteryState.percentage is 0..1 by REP convention.
        pct = float(msg.percentage)
        self._batt_pct = pct * 100.0 if pct <= 1.0001 else pct

    def _status_cb(self, msg: String):
        self._t_status = time.monotonic()
        self._status_text = msg.data

    def _failsafe_cb(self, msg: Bool):
        if msg.data and not self._failsafe:
            self.get_logger().warn('firmware entered FAILSAFE')
        self._failsafe = bool(msg.data)

    # ------------------------------------------------------------------
    def _age(self, stamp):
        return float('inf') if stamp <= 0.0 else time.monotonic() - stamp

    def _tick(self):
        # Some copies of this node never initialise these in __init__.
        for _k, _v in (('_t_start', time.monotonic()), ('_t_front_echo', 0.0),
                       ('_t_rear_echo', 0.0), ('_no_echo_warn', 10.0)):
            if not hasattr(self, _k):
                setattr(self, _k, _v)
        problems = []

        grace = (time.monotonic() - self._startup) > 3.0
        if self._age(self._t_imu) > self._imu_stale:
            problems.append('IMU silent')
        if self._want_front and self._age(self._t_front) > self._rng_stale:
            problems.append('front sonar silent')
        if self._want_rear and self._age(self._t_rear) > self._rng_stale:
            problems.append('rear sonar silent')
        if self._age(self._t_batt) > self._stale:
            problems.append('battery silent')
        if self._failsafe:
            problems.append('firmware failsafe')

        warnings = []
        up = time.monotonic() - self._t_start
        if up > self._no_echo_warn:
            for want, t_msg, t_echo, name in (
                    (self._want_front, self._t_front, self._t_front_echo, 'front'),
                    (self._want_rear, self._t_rear, self._t_rear_echo, 'rear')):
                if (want and self._age(t_msg) <= self._rng_stale
                        and self._age(t_echo) > self._no_echo_warn):
                    warnings.append(
                        '%s sonar: no echo for >%.0fs (only inf) - check '
                        'wiring/RX pin/variant' % (name, self._no_echo_warn))

        level = DiagnosticStatus.OK
        if self._batt_pct == self._batt_pct:          # not NaN
            if self._batt_pct <= self._bat_err:
                problems.append('battery critical %.0f%%' % self._batt_pct)
                level = DiagnosticStatus.ERROR
            elif self._batt_pct <= self._bat_warn:
                problems.append('battery low %.0f%%' % self._batt_pct)
                level = max(level, DiagnosticStatus.WARN)

        healthy = not problems
        if problems and level == DiagnosticStatus.OK:
            level = DiagnosticStatus.ERROR

        if healthy and warnings:
            level = max(level, DiagnosticStatus.WARN)
        summary = '; '.join(problems + warnings) if (problems or warnings) \
            else 'nominal'

        b = Bool()
        b.data = healthy
        self._healthy_pub.publish(b)

        s = String()
        s.data = summary
        self._text_pub.publish(s)

        # Log on change, and periodically while unhealthy, but never spam.
        now = time.monotonic()
        changed = (healthy != self._last_healthy)
        due = (now - self._last_log) > self._log_period
        if changed or (not healthy and due):
            self._last_log = now
            if healthy:
                self.get_logger().info('firmware healthy — %s'
                                       % self._status_text)
            else:
                self.get_logger().warn('firmware unhealthy: %s' % summary)
        self._last_healthy = healthy

        # ---- diagnostics ---------------------------------------------
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = 'a2r_bot: firmware'
        st.hardware_id = 'arduino_nano_esp32'
        st.level = bytes([level]) if isinstance(level, int) else level
        st.message = summary
        st.values = [
            KeyValue(key='battery_percent', value='%.1f' % self._batt_pct),
            KeyValue(key='battery_volts', value='%.2f' % self._batt_v),
            KeyValue(key='imu_age_s', value='%.2f' % self._age(self._t_imu)),
            KeyValue(key='front_age_s',
                     value='%.2f' % self._age(self._t_front)),
            KeyValue(key='rear_age_s', value='%.2f' % self._age(self._t_rear)),
            KeyValue(key='front_echo_age_s',
                     value='%.2f' % self._age(self._t_front_echo)),
            KeyValue(key='failsafe', value=str(self._failsafe)),
            KeyValue(key='status', value=self._status_text),
        ]
        arr.status.append(st)
        self._diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FirmwareMonitorNode()
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
