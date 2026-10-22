#!/usr/bin/env python3
"""
serial_bridge_node.py                            (Raspberry Pi — a2r_bot_arduino)
═══════════════════════════════════════════════════════════════════════════════
THE ONE PROCESS THAT OWNS /dev/ttyARDUINO.

In the previous workspace the planner opened the serial port itself. That works
right up until you want a second thing to talk to the firmware — a battery
monitor, a display test, a SLAM stack that also wants IMU data — and then two
processes fight over one file descriptor and you get silent, intermittent,
impossible-to-reproduce frame corruption.

So: exactly one node holds the port. Everything else is ROS topics.

  IN   geometry_msgs/Twist on drive_topic
         linear.x  = normalised throttle  (-1..1) when input_is_normalised,
                     otherwise m/s and divided by max_speed_mps
         angular.z = normalised steer (-1..1) when input_is_normalised,
                     otherwise steering angle in rad, divided by max_steer
       std_msgs/String on /arduino/command     one-shot firmware commands
       std_msgs/String on /pet/display_text    write 4 chars to the 7-segment

  OUT  sensor_msgs/Imu          /imu/data_raw      50 Hz, no orientation
       sensor_msgs/MagneticField /imu/mag          50 Hz, when a mag is fitted
       sensor_msgs/Range        /ultrasonic/front  10 Hz
       sensor_msgs/Range        /ultrasonic/rear   10 Hz
       sensor_msgs/BatteryState /battery            2 Hz
       sensor_msgs/FluidPressure /baro              5 Hz
       std_msgs/String          /arduino/status    heartbeat, verbatim
       std_msgs/Bool            /arduino/failsafe  firmware watchdog state
       diagnostic_msgs/DiagnosticArray /diagnostics

WHAT IT GUARANTEES
──────────────────
  • The serial port is opened once and reconnected automatically if it drops.
    A brownout that resets the Arduino does not require restarting ROS.
  • Reads happen on a dedicated thread, so a stalled ROS executor can never
    let the OS receive buffer overflow and corrupt the frame boundaries.
  • Writes are rate-limited and deduplicated: repeated identical STOP frames
    are sent once, which keeps the link quiet enough for telemetry to flow.
  • A malformed telemetry line is counted and dropped, never crashes the node.
═══════════════════════════════════════════════════════════════════════════════
"""

import glob
import json
import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import (Imu, MagneticField, Range, BatteryState,
                             FluidPressure)
from std_msgs.msg import String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

try:
    import serial
    import serial.tools.list_ports as list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# USB VID/PID pairs seen on the boards this firmware targets. Matching on
# these rather than on /dev/ttyACM0 means plugging in a second USB device
# cannot silently steal the port.
_BOARD_HINTS = [
    ("303a", "1001"),   # Espressif native USB — Arduino Nano ESP32
    ("2341", "0070"),   # Arduino VID, Nano ESP32
    ("2341", "0058"),   # Arduino Nano Every
    ("10c4", "ea60"),   # CP2102
    ("1a86", "7523"),   # CH340
    ("0403", "6001"),   # FT232
]


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__("serial_bridge_node")

        # ── Port ────────────────────────────────────────────────────────
        self.declare_parameter("port", "/dev/ttyARDUINO")
        self.declare_parameter("baud", 921600)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("reconnect_period_s", 0.5)
        # fw >= 3.4.0 sends a link beat at 20 Hz no matter which sensors are
        # fitted, so 1 s of silence really is a dead link. Until the first
        # beat is seen (older firmware) the legacy 8 s limit still applies,
        # because the only guaranteed line there is the 5 s heartbeat.
        self.declare_parameter("serial_silence_s", 1.0)
        # Native USB (Nano ESP32) does NOT reset when the port opens, so there
        # is nothing to wait for. Raise to ~2.0 only for a classic AVR Nano.
        self.declare_parameter("open_settle_s", 0.2)
        # Command round-trip (Pi -> Arduino -> Pi) above this is a warning.
        self.declare_parameter("rtt_warn_ms", 60.0)
        # While idle, re-send STOP at this rate. Keeps the round-trip
        # measurement alive and makes a single lost STOP harmless.
        self.declare_parameter("idle_keepalive_hz", 5.0)

        # ── Command conversion ──────────────────────────────────────────
        self.declare_parameter("drive_topic", "/local_planner/cmd_out")
        # True  → linear.x is already -1..1 (what smart_planner_node sends)
        # False → linear.x is m/s and gets divided by max_speed_mps
        self.declare_parameter("input_is_normalised", True)
        self.declare_parameter("max_speed_mps", 1.2)
        self.declare_parameter("max_steer_deg", 30.0)
        # HARD speed cap, applied to every command right before the Arduino:
        # a non-zero throttle is held within [min_throttle, max_throttle]
        # (fractions of full throttle, both directions). 0 stays 0.
        self.declare_parameter("min_throttle", 0.06)
        # Flip steering if the car turns the wrong way (planner: + = LEFT).
        self.declare_parameter("invert_steering", False)
        # Keep the servo off its end stops (stall current): |steer| <= this.
        self.declare_parameter("max_steer_frac", 0.85)
        self.declare_parameter("max_throttle", 0.09)
        self.declare_parameter("send_rate_hz", 50.0)
        self.declare_parameter("cmd_timeout_s", 0.5)

        # ── Frames ──────────────────────────────────────────────────────
        self.declare_parameter("imu_frame", "imu_link")
        self.declare_parameter("us_front_frame", "ultrasonic_front_link")
        self.declare_parameter("us_rear_frame", "ultrasonic_rear_link")
        self.declare_parameter("us_fov_rad", 0.52)
        self.declare_parameter("us_min_range", 0.03)
        self.declare_parameter("us_max_range", 5.0)

        # Covariances the EKF downstream will actually use. These are not
        # decoration: set them from a bench measurement of your own unit at
        # rest, not from the datasheet's best case.
        self.declare_parameter("imu_gyro_variance", 4.0e-4)
        self.declare_parameter("imu_accel_variance", 8.0e-3)
        self.declare_parameter("mag_variance", 1.0e-5)

        # Pack description. Previously hard-coded, which meant swapping the
        # battery required editing source — and the numbers published on
        # /battery would silently describe the old pack.
        # Tattu TAA67504S25X9: 4S1P, 14.8 V nominal, 6750 mAh.
        self.declare_parameter("battery_cells", 4)
        self.declare_parameter("battery_design_capacity_ah", 6.75)

        g = lambda n: self.get_parameter(n).value
        self._port_param = str(g("port"))
        self._baud = int(g("baud"))
        self._dry = bool(g("dry_run"))
        self._reconnect_s = float(g("reconnect_period_s"))
        self._silence_s = float(g("serial_silence_s"))
        self._settle_s = float(g("open_settle_s"))
        self._rtt_warn = float(g("rtt_warn_ms"))
        self._idle_period = 1.0 / max(float(g("idle_keepalive_hz")), 0.1)
        self._normalised = bool(g("input_is_normalised"))
        self._max_mps = float(g("max_speed_mps"))
        self._max_steer = math.radians(float(g("max_steer_deg")))
        self._min_thr = abs(float(g("min_throttle")))
        self._max_thr = max(self._min_thr, abs(float(g("max_throttle"))))
        self._steer_lim = min(1.0, abs(float(g("max_steer_frac"))))
        self._inv_steer = bool(g("invert_steering"))
        self._rate = float(g("send_rate_hz"))
        self._timeout = float(g("cmd_timeout_s"))
        self._imu_frame = str(g("imu_frame"))
        self._usf_frame = str(g("us_front_frame"))
        self._usr_frame = str(g("us_rear_frame"))
        self._us_fov = float(g("us_fov_rad"))
        self._us_min = float(g("us_min_range"))
        self._us_max = float(g("us_max_range"))
        self._gyro_var = float(g("imu_gyro_variance"))
        self._accel_var = float(g("imu_accel_variance"))
        self._mag_var = float(g("mag_variance"))
        self._bat_cells = int(g("battery_cells"))
        self._bat_capacity = float(g("battery_design_capacity_ah"))

        # ── State ───────────────────────────────────────────────────────
        self._ser: Optional["serial.Serial"] = None
        self._lock = threading.Lock()
        self._ser_lock = threading.Lock()
        self._rx_buf = ""
        self._out_v = 0.0
        self._out_s = 0.0
        self._last_cmd_t = 0.0
        self._stopped = True
        self._last_status = ""
        self._failsafe = True
        self._running = True

        self._n_rx = 0
        self._n_bad = 0
        self._n_tx = 0
        self._last_rx_t = 0.0

        # ── Link timing (fw >= 3.4.0) ────────────────────────────────────
        # Every DRIVE/STOP carries "q"; the firmware's 20 Hz link beat echoes
        # the last q it received and how long ago ("qa"), so
        #     rtt = (now - time we sent q) - qa
        # is the true Pi->Arduino->Pi transport time, excluding the wait for
        # the next beat.
        self._seq = 0
        self._sent_t = {}              # q -> monotonic send time
        self._seen_lk = False
        self._rtt_ms = None            # smoothed
        self._rtt_last_ms = None
        self._rtt_max_ms = 0.0         # since last diag tick
        self._fw_drops = 0
        self._fw_torn = 0

        # ── Publishers ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST)
        self._imu_pub = self.create_publisher(Imu, "/imu/data_raw", sensor_qos)
        self._mag_pub = self.create_publisher(MagneticField, "/imu/mag", sensor_qos)
        self._usf_pub = self.create_publisher(Range, "/ultrasonic/front", sensor_qos)
        self._usr_pub = self.create_publisher(Range, "/ultrasonic/rear", sensor_qos)
        self._bat_pub = self.create_publisher(BatteryState, "/battery", 10)
        self._baro_pub = self.create_publisher(FluidPressure, "/baro", sensor_qos)
        self._status_pub = self.create_publisher(String, "/arduino/status", 10)
        self._fs_pub = self.create_publisher(Bool, "/arduino/failsafe", 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._link_pub = self.create_publisher(String, "/arduino/link", 10)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Twist, str(g("drive_topic")), self._drive_cb, 10)
        self.create_subscription(String, "/arduino/command", self._cmd_cb, 10)
        self.create_subscription(String, "/pet/display_text", self._disp_cb, 10)

        # ── Open the port and start the threads ─────────────────────────
        self._open_serial()

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx_thread.start()

        self.create_timer(1.0, self._diag_tick)

        self.get_logger().info(
            f"serial_bridge_node ready — drive from '{g('drive_topic')}' "
            f"({'normalised' if self._normalised else 'm/s'})")

    # ══════════════════════════════════════════════════════════════════
    #  SERIAL
    # ══════════════════════════════════════════════════════════════════

    def _autodetect(self) -> Optional[str]:
        """Prefer a board whose USB VID/PID we recognise; fall back to the
        first ACM/USB device. Returning the wrong port is worse than
        returning none, so the VID/PID match is tried first and completely."""
        try:
            for p in list_ports.comports():
                vid = f"{p.vid:04x}" if p.vid else ""
                pid = f"{p.pid:04x}" if p.pid else ""
                if (vid, pid) in _BOARD_HINTS:
                    self.get_logger().info(
                        f"Auto-detected board at {p.device} (VID:PID {vid}:{pid})")
                    return p.device
        except Exception:
            pass
        cands = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        return cands[0] if cands else None

    def _open_serial(self) -> bool:
        if self._dry:
            self.get_logger().warn("dry_run — no serial port will be opened")
            return False
        if not HAS_SERIAL:
            self.get_logger().error(
                "pyserial not installed (pip3 install pyserial) — DRY RUN")
            self._dry = True
            return False

        port = self._port_param
        if port == "auto":
            port = self._autodetect()
        if not port:
            return False

        try:
            ser = serial.Serial(port, self._baud, timeout=0.0,
                                write_timeout=0.5)
            # A classic Nano resets when the port opens; the Nano ESP32 (native
            # USB) does not. Settle briefly, drop any half line, move on - a
            # blind 2 s sleep here was 2 s in which nobody drained the board.
            time.sleep(self._settle_s)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            with self._ser_lock:
                self._ser = ser
            self._open_t = time.monotonic()
            self._last_rx_t = 0.0
            # _seen_lk is deliberately NOT reset here: once this board has
            # shown it sends link beats, keep the fast 1 s limit across
            # reopens, so a long freeze is retried every second, not every 8.
            self._rx_buf = ""
            self.get_logger().info(f"Serial open: {port} @ {self._baud}")
            self._write_raw({"cmd": "PING"})
            return True
        except Exception as e:
            self.get_logger().error(f"Could not open {port}: {e}",
                                    throttle_duration_sec=2.0)
            with self._ser_lock:
                self._ser = None
            return False

    def _drop_port(self):
        """Close and forget the handle, once, under the lock."""
        with self._ser_lock:
            ser, self._ser = self._ser, None
            if ser is not None:
                try:
                    ser.close()
                except Exception:      # noqa: BLE001 - already failing
                    pass
        # Forget the last command too. Without this the TX thread still held a
        # fresh throttle value, so the moment the Arduino came back the car
        # resumed the speed it had been doing before it vanished, before the
        # planner had said anything about the meantime.
        with self._lock:
            self._out_v = self._out_s = 0.0
            self._last_cmd_t = 0.0
        self._stopped = False          # force one explicit STOP on reconnect

    def _write_raw(self, obj: dict):
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        if self._dry:
            self.get_logger().info(f"[DRY] → {line.strip()}",
                                   throttle_duration_sec=1.0)
            return
        if self._ser is None:
            # Not dry-run: the link is down (reconnecting). Say so plainly -
            # "[DRY]" here used to look like dry_run had been switched on.
            self.get_logger().warn(f"[LINK DOWN - not sent] {line.strip()}",
                                   throttle_duration_sec=1.0)
            return
        payload = line.encode("ascii")
        with self._ser_lock:
            ser = self._ser
        if ser is None:
            return
        try:
            n = ser.write(payload)
            self._n_tx += 1
            if n is not None and n < len(payload):
                # A short write means the OS buffer is full, which on this
                # link almost always means the Arduino has stopped draining
                # its RX — i.e. it has wedged or browned out.
                self.get_logger().warn(
                    f"Short write {n}/{len(payload)} — is the Arduino alive?")
        except Exception as e:
            self.get_logger().error(f"Write failed: {e} — will reconnect",
                                    throttle_duration_sec=2.0)
            with self._ser_lock:
                if self._ser is ser:
                    try:
                        ser.close()
                    except Exception:
                        pass
                    self._ser = None

    # ══════════════════════════════════════════════════════════════════
    #  RX THREAD
    # ══════════════════════════════════════════════════════════════════

    def _rx_loop(self):
        """Dedicated reader. Runs outside the ROS executor so a slow callback
        elsewhere in the process can never let the kernel buffer overflow and
        split a JSON line down the middle."""
        while self._running and rclpy.ok():
            if self._ser is None:
                if not self._dry:
                    time.sleep(self._reconnect_s)
                    self._open_serial()
                else:
                    time.sleep(0.5)
                continue
            with self._ser_lock:
                ser = self._ser
            if ser is None:
                continue
            try:
                n = ser.in_waiting
                if n:
                    self._rx_buf += ser.read(n).decode("ascii", errors="ignore")
                    self._last_rx_t = time.monotonic()
                else:
                    time.sleep(0.002)
            except Exception as e:
                self.get_logger().warn(f"Read failed: {e}",
                                       throttle_duration_sec=2.0)
                with self._ser_lock:
                    if self._ser is ser:
                        try:
                            ser.close()
                        except Exception:
                            pass
                        self._ser = None
                continue

            # Silence watchdog: the IMU alone streams 50 lines/s, so 3 s of
            # nothing with the port still "open" means the link is dead in a
            # way the OS never reported. Reopen instead of waiting forever.
            _ref = self._last_rx_t or getattr(self, "_open_t", 0.0)
            _limit = self._silence_s if self._seen_lk else max(self._silence_s, 8.0)
            if _ref and time.monotonic() - _ref > _limit:
                self.get_logger().error(
                    "Arduino silent for %.1f s with the port still open - "
                    "reopening the serial link%s" % (
                        _limit,
                        "" if self._seen_lk else
                        " (no link beat seen: flash fw 3.4.0 for 1 s detection "
                        "and self-recovery)"))
                with self._ser_lock:
                    if self._ser is ser:
                        try:
                            ser.close()
                        except Exception:
                            pass
                        self._ser = None
                self._last_rx_t = 0.0
                self._open_t = 0.0
                continue

            while "\n" in self._rx_buf:
                line, self._rx_buf = self._rx_buf.split("\n", 1)
                line = line.strip()
                if line.startswith("{"):
                    self._handle_line(line)

            # A runaway buffer means we are receiving bytes but never a
            # newline — usually a baud mismatch. Drop it rather than grow.
            if len(self._rx_buf) > 4096:
                self._rx_buf = ""
                self._n_bad += 1

    _RESET_REASONS = {0: "unknown", 1: "power-on", 2: "reset pin", 3: "software",
                      4: "CRASH (panic)", 5: "interrupt watchdog",
                      6: "task watchdog", 7: "watchdog", 8: "deep sleep",
                      9: "BROWNOUT - its supply dipped", 10: "sdio"}

    def _check_reboot(self, d):
        """Spot an Arduino reboot from its heartbeat uptime going backwards,
        and say why it rebooted (fw >= 3.3.1 reports esp_reset_reason)."""
        up = d.get("up_ms")
        if not isinstance(up, (int, float)):
            return
        last = getattr(self, "_last_up_ms", None)
        self._last_up_ms = up
        rst = d.get("rst")
        why = self._RESET_REASONS.get(rst, "not reported (flash fw 3.3.1)")
        if last is None:
            if rst in (4, 5, 6, 7, 9):
                self.get_logger().warn(f"Arduino's last reset was: {why}")
            return
        if up + 1000 < last:
            self._reboots = getattr(self, "_reboots", 0) + 1
            advice = ("power the Arduino from its own regulator, not the servo/ESC "
                      "rail, and add a large capacitor on the servo supply"
                      if rst == 9 else
                      "the loop watchdog rebooted a frozen board (fw 3.4.0 "
                      "self-recovery). If this repeats, reseat the I2C wires and "
                      "check txdrop in the heartbeat" if rst in (6, 7) else
                      "send this log - the firmware crashed" if rst in (4, 5)
                      else "check the USB cable and the Arduino's power")
            self.get_logger().error(
                f"ARDUINO REBOOTED (#{self._reboots}, was up {last/1000:.0f} s) - "
                f"reset reason: {why}. The ESC disarms on every reboot, so the "
                f"car stops. Fix: {advice}.")

    def _handle_line(self, line: str):
        if not rclpy.ok():
            return
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            self._n_bad += 1
            return
        self._n_rx += 1
        stamp = self.get_clock().now().to_msg()

        if d.get("wdog"):
            self.get_logger().error(
                "FIRMWARE WATCHDOG TRIPPED — the board stopped servicing the "
                "PCA9685 and its outputs were cut. Check power/brown-out.",
                throttle_duration_sec=5.0)

        if "status" in d:
            self._last_status = str(d["status"])
            self._status_pub.publish(String(data=line))
            self.get_logger().info(f"Arduino: {self._last_status}")
            if d.get("status") == "heartbeat":
                self._check_reboot(d)
                drops, torn = d.get("txdrop"), d.get("torn")
                if isinstance(drops, int) and isinstance(torn, int):
                    if drops > self._fw_drops or torn > self._fw_torn:
                        self.get_logger().warn(
                            f"Arduino dropped {drops - self._fw_drops} telemetry "
                            f"line(s) (+{torn - self._fw_torn} cut) because the Pi "
                            f"was not reading fast enough - check CPU load",
                            throttle_duration_sec=10.0)
                    self._fw_drops, self._fw_torn = drops, torn
            return

        if "e_stop" in d:
            self._failsafe = bool(d["e_stop"])
            self._fs_pub.publish(Bool(data=self._failsafe))
            return

        if "lk" in d:
            self._seen_lk = True
            self._failsafe = bool(d.get("fs", self._failsafe))
            q, qa = d.get("lk"), d.get("qa")
            sent = self._sent_t.get(q) if isinstance(q, int) else None
            if sent is not None and isinstance(qa, (int, float)) and qa >= 0:
                rtt = (time.monotonic() - sent) * 1000.0 - float(qa)
                rtt = max(rtt, 0.0)
                self._rtt_last_ms = rtt
                self._rtt_ms = rtt if self._rtt_ms is None else 0.9 * self._rtt_ms + 0.1 * rtt
                self._rtt_max_ms = max(self._rtt_max_ms, rtt)
            return

        if "ax" in d:
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = self._imu_frame
            imu.linear_acceleration.x = float(d.get("ax", 0.0))
            imu.linear_acceleration.y = float(d.get("ay", 0.0))
            imu.linear_acceleration.z = float(d.get("az", 9.80665))
            imu.angular_velocity.x = float(d.get("gx", 0.0))
            imu.angular_velocity.y = float(d.get("gy", 0.0))
            imu.angular_velocity.z = float(d.get("gz", 0.0))
            # -1 in element 0 is the REP-145 way of saying "no orientation
            # estimate here". The firmware does not compute one on purpose;
            # a2r_bot_sensor_fusion does.
            imu.orientation_covariance[0] = -1.0
            for i in (0, 4, 8):
                imu.angular_velocity_covariance[i] = self._gyro_var
                imu.linear_acceleration_covariance[i] = self._accel_var
            self._imu_pub.publish(imu)

            if "mx" in d:
                m = MagneticField()
                m.header.stamp = stamp
                m.header.frame_id = self._imu_frame
                # Telemetry is in µT; the message wants tesla.
                m.magnetic_field.x = float(d["mx"]) * 1e-6
                m.magnetic_field.y = float(d.get("my", 0.0)) * 1e-6
                m.magnetic_field.z = float(d.get("mz", 0.0)) * 1e-6
                # An all-zero covariance means "unknown" in ROS, and the
                # attitude filter downstream would rather be told what to
                # expect than guess.
                m.magnetic_field_covariance = [
                    self._mag_var, 0.0, 0.0,
                    0.0, self._mag_var, 0.0,
                    0.0, 0.0, self._mag_var,
                ]
                self._mag_pub.publish(m)
            return

        if "us_f" in d:
            self._publish_range(self._usf_pub, self._usf_frame,
                                float(d.get("us_f", -1.0)), stamp)
            self._publish_range(self._usr_pub, self._usr_frame,
                                float(d.get("us_r", -1.0)), stamp)
            return

        if "bat_v" in d:
            b = BatteryState()
            b.header.stamp = stamp
            b.voltage = float(d["bat_v"])
            b.percentage = float(d.get("bat_pct", 0.0)) / 100.0
            b.charge = float(d.get("bat_mah", 0.0)) / 1000.0   # Ah
            b.capacity = self._bat_capacity
            b.design_capacity = self._bat_capacity
            b.present = True
            b.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
            b.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            b.power_supply_health = (
                BatteryState.POWER_SUPPLY_HEALTH_GOOD
                if not d.get("bat_low", 0)
                else BatteryState.POWER_SUPPLY_HEALTH_DEAD)
            cell = float(d.get("bat_cell", 0.0))
            b.cell_voltage = [cell] * self._bat_cells
            self._bat_pub.publish(b)
            if d.get("bat_low", 0):
                self.get_logger().warn(
                    f"BATTERY LOW: {b.voltage:.2f} V ({cell:.3f} V/cell, "
                    f"{d.get('bat_pct', 0):.0f}%) — land it",
                    throttle_duration_sec=30.0)
            return

        if "press" in d:
            f = FluidPressure()
            f.header.stamp = stamp
            f.header.frame_id = self._imu_frame
            f.fluid_pressure = float(d["press"])
            f.variance = 4.0
            self._baro_pub.publish(f)
            return

    def _publish_range(self, pub, frame: str, value: float, stamp):
        r = Range()
        r.header.stamp = stamp
        r.header.frame_id = frame
        r.radiation_type = Range.ULTRASOUND
        r.field_of_view = self._us_fov
        r.min_range = self._us_min
        r.max_range = self._us_max
        # The firmware sends -1 for "no echo". REP-117 says an out-of-range
        # reading should be +inf, which is meaningfully different from a
        # measured max_range — the fusion node relies on that distinction.
        r.range = float("inf") if value < 0.0 else value
        pub.publish(r)

    # ══════════════════════════════════════════════════════════════════
    #  TX
    # ══════════════════════════════════════════════════════════════════

    def _drive_cb(self, msg: Twist):
        v = float(msg.linear.x)
        if not self._normalised:
            v = v / max(self._max_mps, 1e-3)
        # The planner already sends normalised steer (-1..1). Dividing it by
        # max_steer (0.524 rad) again made the car hit full lock at ~52 % of
        # what the planner asked for, and disagree with the EKF's model.
        s = float(msg.angular.z)
        if not self._normalised:
            s = s / max(self._max_steer, 1e-3)
        if self._inv_steer:
            s = -s                     # servo steers the other way round
        s = max(-self._steer_lim, min(self._steer_lim, s))   # off the end stops
        # HARD speed cap (min_throttle / max_throttle): nothing upstream -
        # planner, Jetson, teleop, escape - can drive faster than max_throttle,
        # and a moving command never drops below min_throttle (where the ESC
        # would just hum). A zero command stays zero.
        if abs(v) > 1e-3:
            v = math.copysign(min(max(abs(v), self._min_thr), self._max_thr), v)
        else:
            v = 0.0
        with self._lock:
            self._out_v = max(-1.0, min(1.0, v))
            self._out_s = max(-1.0, min(1.0, s))
            self._last_cmd_t = time.monotonic()

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd in ("CAL_IMU", "PING", "STOP", "RESET_ENC"):
            self._write_raw({"cmd": cmd})
        else:
            self.get_logger().warn(f"Unknown firmware command: {cmd}")

    def _disp_cb(self, msg: String):
        self._write_raw({"cmd": "DISP", "text": msg.data[:4]})

    def _send_cmd(self, obj: dict):
        """Stamp a command with the next sequence number and remember when it
        left, so the link beat that echoes it back gives a real round-trip."""
        self._seq = (self._seq + 1) & 0x7FFFFFFF
        obj["q"] = self._seq
        self._sent_t[self._seq] = time.monotonic()
        # keep ~2 s of history at 50 Hz; older echoes are meaningless anyway
        self._sent_t.pop(self._seq - 128, None)
        self._write_raw(obj)

    def _tx_loop(self):
        """Fixed-rate sender on its own thread. Sending from a ROS timer would
        make the drive rate hostage to whatever else the executor is doing."""
        interval = 1.0 / max(self._rate, 1.0)
        while self._running and rclpy.ok():
            now = time.monotonic()
            with self._lock:
                v, s = self._out_v, self._out_s
                fresh = (now - self._last_cmd_t) <= self._timeout

            if not fresh or (abs(v) < 1e-3 and abs(s) < 1e-3):
                # STOP immediately on the transition, then a slow keepalive
                # STOP (idle_keepalive_hz, default 5 Hz) - not fifty a second.
                # ~150 B/s is nothing on the link, it makes one lost STOP
                # harmless, and it keeps the round-trip measurement alive.
                if not self._stopped or now - getattr(self, "_last_stop_t", 0.0) >= self._idle_period:
                    self._send_cmd({"cmd": "STOP"})
                    self._stopped = True
                    self._last_stop_t = now
            else:
                self._stopped = False
                self._send_cmd({"cmd": "DRIVE",
                                "v": round(v, 3), "s": round(s, 3)})
            time.sleep(interval)

    # ══════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def _diag_tick(self):
        # USB link state for the live status panel (robot_status_node).
        _age = (time.monotonic() - self._last_rx_t) if self._last_rx_t else None
        if self._dry:
            _st = "dry_run"
        elif self._ser is None:
            _st = "disconnected"
        elif _age is None or _age > 1.5:
            _st = "silent"
        else:
            _st = "connected"
        _rtt = None if self._rtt_ms is None else round(self._rtt_ms, 1)
        _rtt_max = round(self._rtt_max_ms, 1)
        self._link_pub.publish(String(data=json.dumps({
            "state": _st, "port": self._port_param,
            "rx_age_s": None if _age is None else round(_age, 1),
            "reboots": getattr(self, "_reboots", 0),
            "rtt_ms": _rtt, "rtt_max_ms": _rtt_max,
            "link_beat": self._seen_lk,
            "fw_txdrop": self._fw_drops, "fw_torn": self._fw_torn})))
        if _rtt_max > self._rtt_warn:
            self.get_logger().warn(
                f"Arduino command round-trip {_rtt_max:.0f} ms (limit "
                f"{self._rtt_warn:.0f} ms) - drive commands are arriving late",
                throttle_duration_sec=5.0)
        self._rtt_max_ms = 0.0
        da = DiagnosticArray()
        da.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = "a2r_bot: arduino link"
        st.hardware_id = self._port_param

        age = time.monotonic() - self._last_rx_t if self._last_rx_t else 999.0
        if self._dry:
            st.level = DiagnosticStatus.WARN
            st.message = "dry_run — no hardware attached"
        elif self._ser is None:
            st.level = DiagnosticStatus.ERROR
            st.message = "serial port closed, retrying"
        elif age > 3.0:
            st.level = DiagnosticStatus.ERROR
            st.message = f"no telemetry for {age:.1f} s"
        elif _rtt_max > self._rtt_warn:
            st.level = DiagnosticStatus.WARN
            st.message = f"slow link: command round-trip {_rtt_max:.0f} ms"
        elif self._failsafe:
            st.level = DiagnosticStatus.WARN
            st.message = "firmware in failsafe (no recent DRIVE frames)"
        else:
            st.level = DiagnosticStatus.OK
            st.message = "ok"

        st.values = [
            KeyValue(key="rx_lines", value=str(self._n_rx)),
            KeyValue(key="rx_bad", value=str(self._n_bad)),
            KeyValue(key="tx_frames", value=str(self._n_tx)),
            KeyValue(key="last_status", value=self._last_status),
            KeyValue(key="rtt_ms", value=str(_rtt)),
            KeyValue(key="rtt_max_ms", value=str(_rtt_max)),
            KeyValue(key="fw_txdrop", value=str(self._fw_drops)),
        ]
        da.status.append(st)
        self._diag_pub.publish(da)

    def destroy_node(self):
        self._running = False
        # Two STOPs, because the whole point of shutdown is that the car is
        # not left driving if the first one is lost in a buffer.
        try:
            self._write_raw({"cmd": "STOP"})
            time.sleep(0.05)
            self._write_raw({"cmd": "STOP"})
            time.sleep(0.05)
        except Exception:      # noqa: BLE001 - shutting down anyway
            pass
        self._drop_port()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
