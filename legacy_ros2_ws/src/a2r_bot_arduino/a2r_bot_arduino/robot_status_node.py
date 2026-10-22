#!/usr/bin/env python3
"""robot_status_node (v1) - one live, colour-coded status panel for the robot.

Shown in RViz as text floating above the car (MarkerArray on
/robot_status/markers), as plain text on /robot_status/text, and - with
tui:=true (make status) - as a live terminal panel.

  PLANNER   current state: IDLE / FOLLOW / STUCK / ESCAPING / BOXED IN
  MANEUVER  what it is doing about it: pre-steer, reverse distance, exit turn
  OBSTACLE  the closest lidar return and which side it is on
  ARDUINO   USB link: connected / silent / disconnected / unplugged / bridge
            down, firmware, uptime, reboots and their reset reason
  PCA9685   present on the Arduino's I2C bus, ESC armed, failsafe
  IMU       present on the Arduino's I2C bus, stream rate, filter, dropouts
  BATTERY   voltage, percent, volts per cell
  LIDAR     scan rate

Inputs: /local_planner/status, /arduino/link, /arduino/status (heartbeat),
/battery, /imu/data_raw, /imu/data, /scan. Every line turns red or grey when
its source goes quiet, so a frozen panel can never look healthy.
"""
import collections
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import BatteryState, Imu, LaserScan
from visualization_msgs.msg import Marker, MarkerArray

OK, WARN, BAD, INFO = 'ok', 'warn', 'bad', 'info'
RGB = {OK: (0.30, 0.95, 0.40), WARN: (1.00, 0.78, 0.15),
       BAD: (1.00, 0.30, 0.30), INFO: (0.80, 0.82, 0.88)}
ANSI = {OK: '\033[32m', WARN: '\033[33m', BAD: '\033[31m', INFO: '\033[37m'}
RESET_REASONS = {0: "unknown", 1: "power-on", 2: "reset pin", 3: "software",
                 4: "CRASH (panic)", 5: "interrupt watchdog", 6: "task watchdog",
                 7: "watchdog", 8: "deep sleep", 9: "BROWNOUT", 10: "sdio"}
MAX_LINES = 9


class Rate:
    """Messages per second over a sliding window, plus time since the last."""

    def __init__(self, window=2.0):
        self.w = window
        self.t = collections.deque()

    def tick(self):
        self.t.append(time.monotonic())

    def _trim(self, now):
        while self.t and now - self.t[0] > self.w:
            self.t.popleft()

    def hz(self):
        now = time.monotonic()
        self._trim(now)
        return len(self.t) / self.w

    def age(self):
        return time.monotonic() - self.t[-1] if self.t else float('inf')


def side_name(deg):
    a = abs(deg)
    if a <= 30:
        return "front"
    if a >= 150:
        return "rear"
    lr = "left" if deg > 0 else "right"
    if a < 70:
        return "front-" + lr
    if a > 110:
        return "rear-" + lr
    return lr


class RobotStatusNode(Node):

    def __init__(self):
        super().__init__('robot_status_node')
        g = lambda n, d: self.declare_parameter(n, d).value
        self._markers = bool(g('markers', True))
        self._tui = bool(g('tui', False))
        self._usb = str(g('usb_device', '/dev/ttyARDUINO'))
        self._frame = str(g('frame', 'base_link'))
        self._text_h = float(g('text_height', 0.07))
        self._z_base = float(g('z_base', 0.85))   # bottom line; above the planner's 0.6 m label
        self._bat_warn = float(g('battery_warn_percent', 25.0))
        self._bat_bad = float(g('battery_error_percent', 12.0))

        self._planner, self._planner_t = None, 0.0
        self._link, self._link_t = None, 0.0
        self._hb, self._hb_t = None, 0.0
        self._bat, self._bat_t = None, 0.0
        self._reboots, self._last_up = 0, None
        self._imu_raw, self._imu_f, self._scan = Rate(), Rate(), Rate()

        self.create_subscription(String, '/local_planner/status', self._planner_cb, 10)
        self.create_subscription(String, '/arduino/link', self._link_cb, 10)
        self.create_subscription(String, '/arduino/status', self._status_cb, 20)
        self.create_subscription(BatteryState, '/battery', self._bat_cb, 10)
        self.create_subscription(Imu, '/imu/data_raw', lambda _m: self._imu_raw.tick(),
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/data', lambda _m: self._imu_f.tick(),
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', lambda _m: self._scan.tick(),
                                 qos_profile_sensor_data)
        self._txt_pub = self.create_publisher(String, '/robot_status/text', 10)
        self._mk_pub = (self.create_publisher(MarkerArray, '/robot_status/markers', 1)
                        if self._markers else None)
        self.create_timer(0.25, self._tick)
        self.get_logger().info(
            "robot_status_node up - RViz: add MarkerArray /robot_status/markers"
            + ("  (terminal panel on)" if self._tui else ""))

    # ---- inputs -------------------------------------------------------
    def _planner_cb(self, m):
        try:
            self._planner, self._planner_t = json.loads(m.data), time.monotonic()
        except ValueError:
            pass

    def _link_cb(self, m):
        try:
            self._link, self._link_t = json.loads(m.data), time.monotonic()
        except ValueError:
            pass

    def _status_cb(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if d.get('status') != 'heartbeat':
            return
        up = d.get('up_ms')
        if isinstance(up, (int, float)):
            if self._last_up is not None and up + 1000 < self._last_up:
                self._reboots += 1
            self._last_up = up
        self._hb, self._hb_t = d, time.monotonic()

    def _bat_cb(self, m):
        self._bat, self._bat_t = m, time.monotonic()

    # ---- the panel ----------------------------------------------------
    def lines(self):
        now = time.monotonic()
        out = []
        hb = self._hb if now - self._hb_t < 12.0 else None

        # PLANNER + MANEUVER
        p = self._planner if now - self._planner_t < 1.5 else None
        if p is None:
            out.append(("PLANNER", "no status - planner not running?", BAD))
        else:
            st = str(p.get('state', '?'))
            att = int(p.get('attempt', 0) or 0)
            v, sd = p.get('v', 0.0), p.get('steer_deg', 0.0)
            if st.startswith('IDLE'):
                out.append(("PLANNER", "IDLE - waiting for a target", INFO))
            elif st.startswith('BOXED'):
                out.append(("PLANNER", "BOXED IN - TRAPPED, stopped", BAD))
            elif st in ('PRE-STEER', 'ESCAPE-REVERSE'):
                out.append(("PLANNER", "ESCAPING - %s (attempt %d)" % (st, max(att, 1)), WARN))
            elif 'stuck_s' in p:
                out.append(("PLANNER", "STUCK %.1f s - escape starts at %.1f s%s"
                            % (p['stuck_s'], p.get('stuck_need_s', 0.0),
                               (" (attempt %d next)" % (att + 1)) if att else ""), WARN))
            else:
                out.append(("PLANNER", "%s   v %+.2f   steer %+.0f deg" % (st, v, sd), OK))
            if p.get('maneuver'):
                lvl = BAD if st.startswith('BOXED') else WARN
                out.append(("MANEUVER", str(p['maneuver']), lvl))
            if 'closest_m' in p:
                d = float(p['closest_m'])
                lvl = BAD if d < 0.35 else WARN if d < 0.60 else OK
                out.append(("OBSTACLE", "closest %.2f m, %s"
                            % (d, side_name(float(p.get('closest_deg', 0.0)))), lvl))

        # ARDUINO USB
        present = True if self._usb in ('', 'auto') else os.path.exists(self._usb)
        L = self._link if now - self._link_t < 3.0 else None
        reb = max(self._reboots, int((L or {}).get('reboots', 0) or 0))
        rst = RESET_REASONS.get((hb or {}).get('rst'), None)
        extra = ""
        if reb:
            extra = " - REBOOTED %dx%s" % (reb, (" (last: %s)" % rst) if rst else "")
        if L is None:
            out.append(("ARDUINO", ("UNPLUGGED - %s is missing" % self._usb) if not present
                        else "bridge not running (no /arduino/link)", BAD))
        else:
            s = L.get('state')
            if s == 'connected':
                up = (hb or {}).get('up_ms')
                txt = "USB connected %s   fw %s   up %s" % (
                    L.get('port', ''), (hb or {}).get('fw', '?'),
                    ("%.0f s" % (up / 1000.0)) if isinstance(up, (int, float)) else "?")
                out.append(("ARDUINO", txt + extra, WARN if reb else OK))
            elif s == 'silent':
                out.append(("ARDUINO", "SILENT - port open, no data for %s s%s"
                            % (L.get('rx_age_s', '?'), extra), BAD))
            elif s == 'dry_run':
                out.append(("ARDUINO", "dry run - no serial port opened", WARN))
            else:
                out.append(("ARDUINO", ("DISCONNECTED - USB unplugged (%s missing)" % self._usb)
                            if not present else "DISCONNECTED - reconnecting..." + extra, BAD))

        # PCA9685
        if hb is None:
            out.append(("PCA9685", "unknown - no heartbeat from the Arduino", INFO))
        elif not hb.get('pca'):
            out.append(("PCA9685", "MISSING on the Arduino's I2C bus - no steering/throttle", BAD))
        elif hb.get('failsafe'):
            out.append(("PCA9685", "OK - FAILSAFE active (no drive commands), wheels held", WARN))
        else:
            out.append(("PCA9685", "OK - ESC %s   ESC %s us   servo %s us" % (
                "armed" if hb.get('armed') else "NOT armed", hb.get('esc_us', '?'),
                hb.get('srv_us', '?')), OK if hb.get('armed') else WARN))

        # IMU
        raw_hz, f_hz = self._imu_raw.hz(), self._imu_f.hz()
        rec = int((hb or {}).get('imu_rec', 0) or 0)
        tail = (" - recovered %d dropout(s)" % rec) if rec else ""
        if hb is not None and not hb.get('imu') and raw_hz < 1.0:
            out.append(("IMU", "MISSING / DROPPED on the Arduino's I2C bus" + tail, BAD))
        elif raw_hz < 1.0:
            out.append(("IMU", "silent - no /imu/data_raw" + tail, BAD if hb else INFO))
        else:
            mag = "" if hb is None else ("  mag ok" if hb.get('mag') else "  mag MISSING")
            filt = "filter %.0f Hz" % f_hz if f_hz >= 5.0 else "filter SILENT (calibrating?)"
            lvl = OK if (raw_hz >= 30.0 and f_hz >= 5.0 and not rec) else WARN
            out.append(("IMU", "OK %.0f Hz   %s%s%s" % (raw_hz, filt, mag, tail), lvl))

        # BATTERY
        b = self._bat if now - self._bat_t < 5.0 else None
        if b is None:
            out.append(("BATTERY", "no battery data", INFO if hb is None else WARN))
        else:
            pct = float(b.percentage)
            pct = pct * 100.0 if pct <= 1.0 else pct          # REP: 0..1
            cells = [c for c in b.cell_voltage if c > 0] if len(b.cell_voltage) else []
            cell = (" (%.2f V/cell)" % cells[0]) if cells else ""
            lvl = BAD if pct < self._bat_bad else WARN if pct < self._bat_warn else OK
            out.append(("BATTERY", "%.2f V   %.0f %%%s" % (b.voltage, pct, cell), lvl))

        # LIDAR
        s_hz = self._scan.hz()
        out.append(("LIDAR", ("%.1f Hz" % s_hz) if s_hz > 0 else "SILENT - no /scan",
                    OK if s_hz >= 5.0 else WARN if s_hz > 0 else BAD))
        return out[:MAX_LINES]

    # ---- output -------------------------------------------------------
    def _tick(self):
        rows = self.lines()
        self._txt_pub.publish(String(data="\n".join("%-9s %s" % (a, b) for a, b, _ in rows)))
        if self._mk_pub is not None:
            self._mk_pub.publish(self._markers_for(rows))
        if self._tui:
            self._print(rows)

    def _markers_for(self, rows):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i in range(MAX_LINES):
            m = Marker()
            m.header.frame_id = self._frame
            m.header.stamp = stamp
            m.ns, m.id = 'robot_status', i
            if i >= len(rows):
                m.action = Marker.DELETE
                arr.markers.append(m)
                continue
            label, text, lvl = rows[i]
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.z = self._z_base + (len(rows) - 1 - i) * self._text_h * 1.45
            m.pose.orientation.w = 1.0
            m.scale.z = self._text_h
            m.color.r, m.color.g, m.color.b = RGB[lvl]
            m.color.a = 1.0
            m.text = "%s  %s" % (label, text)
            m.lifetime.sec = 2              # vanishes if this node dies
            arr.markers.append(m)
        return arr

    def _print(self, rows):
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write("\033[1mA2R_bot live status\033[0m   %s   (Ctrl-C to quit)\n\n"
                         % time.strftime("%H:%M:%S"))
        for label, text, lvl in rows:
            sys.stdout.write("  \033[1m%-9s\033[0m %s%s\033[0m\n" % (label, ANSI[lvl], text))
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
