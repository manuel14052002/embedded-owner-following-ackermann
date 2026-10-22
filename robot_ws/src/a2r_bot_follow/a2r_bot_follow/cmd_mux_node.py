#!/usr/bin/env python3
"""
cmd_mux_node.py — priority velocity multiplexer (replaces the apt twist_mux)
═══════════════════════════════════════════════════════════════════════════════
Same job and the same topics as twist_mux, built into a2r_bot_follow so there
is no external package that can fail to start (the apt twist_mux died at
launch on this robot, which silently cut the whole drive chain).

  inputs   "name:topic:priority:timeout_s"   (params.yaml -> cmd_mux.inputs)
           teleop     /cmd_vel_teleop   100   0.5
           escape     /cmd_vel_escape    50   0.25
           navigation /cmd_vel_nav       10   0.5
  lock     /estop_lock (std_msgs/Bool)  true = everything blocked, zero sent
  output   /cmd_vel_mux  -> collision_monitor -> serial bridge
  status   /cmd_vel_mux/active (String): which input is driving, or "none"/"LOCKED"

Rules (twist_mux semantics):
  * an input is ACTIVE while its last message is younger than its timeout
  * a message is forwarded only if its input has the highest priority among
    the active inputs
  * while the lock is true nothing is forwarded (a zero is published once)
One addition: when the driving input goes quiet and no other input is active,
ONE zero command is published, so the car stops at once instead of coasting
on the last command until the serial bridge's own timeout.

Commands use the a2r_bot drive convention (fractions -1..1); the mux does not
change them.
"""
import time


class MuxCore:
    """Pure logic, testable without ROS."""

    def __init__(self, inputs):
        # inputs: list of (name, topic, priority, timeout)
        self.inputs = {n: {'topic': t, 'prio': int(p), 'timeout': float(to), 'last': -1e9}
                       for n, t, p, to in inputs}
        self.locked = False
        self.active = 'none'

    def _is_active(self, name, now):
        i = self.inputs[name]
        return now - i['last'] <= i['timeout']

    def _best(self, now):
        act = [n for n in self.inputs if self._is_active(n, now)]
        return max(act, key=lambda n: self.inputs[n]['prio']) if act else None

    def on_msg(self, name, now):
        """A message arrived on input `name`. Returns True if it must be forwarded."""
        self.inputs[name]['last'] = now
        if self.locked:
            self.active = 'LOCKED'
            return False
        best = self._best(now)
        if best == name:
            self.active = name
            return True
        return False

    def on_lock(self, value: bool):
        """Returns True if a zero must be published now (lock just engaged)."""
        engaged = bool(value) and not self.locked
        self.locked = bool(value)
        if self.locked:
            self.active = 'LOCKED'
        elif self.active == 'LOCKED':
            self.active = 'none'
        return engaged

    def tick(self, now):
        """Periodic check. Returns True if a zero must be published (driver went quiet)."""
        if self.locked:
            return False
        if self.active not in ('none', 'LOCKED') and self._best(now) is None:
            self.active = 'none'
            return True
        best = self._best(now)
        if best is not None and best != self.active:
            self.active = best          # next message from `best` is forwarded anyway
        return False


def parse_inputs(spec: str):
    out = []
    for item in [s.strip() for s in spec.split(',') if s.strip()]:
        name, topic, prio, timeout = [x.strip() for x in item.split(':')]
        out.append((name, topic, int(prio), float(timeout)))
    return out


def main(args=None):  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import Bool, String

    class CmdMux(Node):
        def __init__(self):
            super().__init__('cmd_mux')
            self.declare_parameter('inputs', 'teleop:/cmd_vel_teleop:100:0.5,'
                                             'escape:/cmd_vel_escape:50:0.25,'
                                             'navigation:/cmd_vel_nav:10:0.5')
            self.declare_parameter('lock_topic', '/estop_lock')
            self.declare_parameter('output_topic', '/cmd_vel_mux')
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            inputs = parse_inputs(str(g('inputs')))
            self.core = MuxCore(inputs)
            self.pub = self.create_publisher(Twist, str(g('output_topic')), 10)
            self.pub_active = self.create_publisher(String, str(g('output_topic')) + '/active', 10)
            for name, topic, prio, to in inputs:
                self.create_subscription(Twist, topic, lambda m, n=name: self._on(n, m), 10)
            self.create_subscription(Bool, str(g('lock_topic')), self._on_lock, 10)
            self.create_timer(0.05, self._tick)
            self.create_timer(0.5, self._status)
            self._last_active = None
            desc = ', '.join(f'{n}({p}) {t}' for n, t, p, _ in sorted(inputs, key=lambda x: -x[2]))
            self.get_logger().info(f"cmd_mux: {desc} -> {g('output_topic')}; lock {g('lock_topic')}")

        def _on(self, name, msg):
            if self.core.on_msg(name, time.monotonic()):
                self.pub.publish(msg)
            self._log_change()

        def _on_lock(self, msg):
            if self.core.on_lock(msg.data):
                self.pub.publish(Twist())
                self.get_logger().warn('cmd_mux: E-STOP LOCK engaged - all commands blocked')
            elif not msg.data:
                self.get_logger().info('cmd_mux: e-stop lock released', throttle_duration_sec=5.0)
            self._log_change()

        def _tick(self):
            if self.core.tick(time.monotonic()):
                self.pub.publish(Twist())
            self._log_change()

        def _log_change(self):
            # v2.5: when the planner is starved its 50 Hz stream pauses for a few
            # 100 ms and the input flips navigation<->none dozens of times a
            # minute (log: ~90 lines in 2 min, each an SD-card write). Log "none"
            # only if it lasts 1 s; count the short gaps and report every 30 s.
            a, now = self.core.active, time.monotonic()
            if not hasattr(self, "_gap_t"):
                self._gap_t, self._gaps, self._gap_rep = None, 0, now
            if a == "none" and self._last_active not in (None, "none"):
                if self._gap_t is None:
                    self._gap_t = now
                if now - self._gap_t < 1.0:
                    return
            elif a != "none" and self._gap_t is not None:
                if a == self._last_active:
                    self._gaps += 1            # short gap, same input back: count, do not log
                self._gap_t = None
            if now - self._gap_rep >= 30.0:
                if self._gaps:
                    self.get_logger().info(f"cmd_mux: {self._gaps} short input gaps (<1 s) in the last 30 s "
                                           "- a starved publisher (check CPU: a2r_v24.sh --health)")
                self._gaps, self._gap_rep = 0, now
            if a != self._last_active:
                self.get_logger().info(f'cmd_mux: driving input -> {a}')
                self._last_active = a

        def _status(self):
            self.pub_active.publish(String(data=self.core.active))

    rclpy.init(args=args)
    node = CmdMux()
    try:
        import gc               # a2r_merge: init objects out of the GC (no full-pass stalls)
        gc.collect()
        gc.freeze()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
