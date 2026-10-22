#!/usr/bin/env python3
"""
state_bridge_node.py
====================
Maps the state strings the Jetson publishes onto face expressions.

-----------------------------------------------------------------------------
There is deliberately no state machine here
-----------------------------------------------------------------------------
The original PET-mode code ran a finite state machine on the Raspberry Pi:
IDLE -> FIND -> GO_TO_OWNER -> FOLLOW_OWNER -> COMPLETED, with wave detection
driving the transitions.  That has been removed, as requested.

It was the right thing to remove.  The Jetson is the machine that can see —
it runs the person detector and the tracker, so it is the only machine that
knows whether an owner has been found, is being approached, or has been lost.
A second state machine on the Pi could only ever be guessing at the first
one's state from indirect evidence, and two state machines that are supposed
to agree but derive their state independently will eventually disagree.  The
usual symptom is a robot whose face says it is following someone while its
wheels have given up.

So the Jetson owns the FSM and publishes its current state.  This node's
entire job is translation: state string in, expression name out.  It holds no
state of its own beyond the last thing it was told, and it makes no decisions.

The mapping is data, not code — it lives in config/pet.yaml, so adding a state
the Jetson starts sending is a config edit, not a source change.

Subscribes
----------
/pet/state          std_msgs/String    from the Jetson
/pet/expression     std_msgs/String    direct override, bypasses the mapping
/arduino/healthy    std_msgs/Bool      face reacts to the robot being unwell
/battery            sensor_msgs/BatteryState

Publishes
---------
/pet/face           std_msgs/String    the expression face_node should show
/pet/bridge_status  std_msgs/String
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from sensor_msgs.msg import BatteryState


class StateBridgeNode(Node):
    def __init__(self):
        super().__init__('state_bridge_node')

        self.declare_parameter('state_topic', '/pet/state')
        self.declare_parameter('face_topic', '/pet/face')
        self.declare_parameter('default_expression', 'idle')
        self.declare_parameter('unknown_expression', 'curious')
        self.declare_parameter('case_sensitive', False)
        self.declare_parameter('republish_period_s', 1.0)

        # The mapping arrives as two parallel lists because ROS 2 parameters
        # cannot hold a dictionary. Ugly, but it keeps the mapping in YAML
        # where it can be edited without touching Python.
        self.declare_parameter('state_names', [
            'IDLE', 'FIND', 'GO_TO_OWNER', 'FOLLOW_OWNER', 'COMPLETED',
            'ERROR',
            'NEUTRAL', 'SURPRISED', 'EXCITED', 'INNOCENT', 'HAPPY', 'SAD',
        ])
        self.declare_parameter('state_expressions', [
            'idle', 'curious', 'happy', 'happy', 'happy',
            'sad',
            'idle', 'curious', 'happy', 'happy', 'happy', 'sad',
        ])

        self.declare_parameter('react_to_health', True)
        self.declare_parameter('unhealthy_expression', 'sad')
        self.declare_parameter('low_battery_expression', 'bored')
        self.declare_parameter('low_battery_percent', 15.0)

        g = self.get_parameter
        self._default = str(g('default_expression').value)
        self._unknown = str(g('unknown_expression').value)
        self._case = bool(g('case_sensitive').value)
        self._react = bool(g('react_to_health').value)
        self._unwell = str(g('unhealthy_expression').value)
        self._low = str(g('low_battery_expression').value)
        self._low_pct = float(g('low_battery_percent').value)

        names = list(g('state_names').value)
        exprs = list(g('state_expressions').value)
        if len(names) != len(exprs):
            self.get_logger().error(
                'state_names has %d entries but state_expressions has %d — '
                'the mapping is ignored and everything will show "%s"'
                % (len(names), len(exprs), self._unknown))
            names, exprs = [], []
        self._map = {}
        for n, e in zip(names, exprs):
            self._map[n if self._case else n.upper()] = e

        self._current = self._default
        self._last_state = ''
        self._override = None
        self._override_t = 0.0
        self._healthy = True
        self._battery = 100.0
        self._warned = set()

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._face_pub = self.create_publisher(
            String, str(g('face_topic').value), latched)
        self._status_pub = self.create_publisher(
            String, '/pet/bridge_status', 10)

        self.create_subscription(String, str(g('state_topic').value),
                                 self._state_cb, 10)
        self.create_subscription(String, '/pet/expression',
                                 self._override_cb, 10)
        self.create_subscription(Bool, '/arduino/healthy',
                                 self._health_cb, 10)
        self.create_subscription(BatteryState, '/battery', self._batt_cb, 10)

        self.create_timer(float(g('republish_period_s').value), self._tick)
        self._publish()

        self.get_logger().info(
            'state_bridge_node up — %d states mapped, no FSM here (the '
            'Jetson owns that)' % len(self._map))

    # ------------------------------------------------------------------
    def _state_cb(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        key = raw if self._case else raw.upper()
        self._last_state = raw

        if key in self._map:
            expr = self._map[key]
        else:
            expr = self._unknown
            # Warn once per unseen state, not once per message: the Jetson
            # publishes at 10 Hz and an unmapped state would otherwise fill
            # the log in seconds.
            if key not in self._warned:
                self._warned.add(key)
                self.get_logger().warn(
                    'no mapping for state "%s" — showing "%s". Add it to '
                    'state_names/state_expressions in pet.yaml.'
                    % (raw, expr))

        if expr != self._current:
            self.get_logger().info('%s -> %s' % (raw, expr))
        self._current = expr
        self._publish()

    def _override_cb(self, msg: String):
        # A direct expression request wins over the state mapping for a
        # while. Useful for scripted behaviour and for testing the screen.
        self._override = msg.data.strip()
        self._override_t = time.monotonic()
        self._publish()

    def _health_cb(self, msg: Bool):
        self._healthy = bool(msg.data)

    def _batt_cb(self, msg: BatteryState):
        pct = float(msg.percentage)
        self._battery = pct * 100.0 if pct <= 1.0001 else pct

    # ------------------------------------------------------------------
    def _effective(self):
        if self._override and (time.monotonic() - self._override_t) < 10.0:
            return self._override
        if self._react:
            # The face is the only output this robot has that a person can
            # read from across a room, so it is worth using to say something
            # is wrong. Health beats battery: a failsafe matters more than a
            # low cell.
            if not self._healthy:
                return self._unwell
            if self._battery <= self._low_pct:
                return self._low
        return self._current

    def _tick(self):
        self._publish()

    def _publish(self):
        m = String()
        m.data = self._effective()
        self._face_pub.publish(m)

        s = String()
        s.data = ('state=%s expr=%s healthy=%s batt=%.0f%%'
                  % (self._last_state or '-', m.data, self._healthy,
                     self._battery))
        self._status_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = StateBridgeNode()
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
