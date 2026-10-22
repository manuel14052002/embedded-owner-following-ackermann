#!/usr/bin/env python3
"""
serial_bridge_node.py
─────────────────────────────────────────────────────────────────────────────
Converts ROS2 /cmd_vel (geometry_msgs/Twist) into newline-delimited JSON DRIVE
commands for the a2r_bot ESP32 firmware, and republishes ESP32 telemetry back
onto the ROS graph.

Command contract (matches a2r_bot_drive.ino):
    TX:  {"cmd":"DRIVE","v":<-1..1>,"s":<-1..1>}   20 Hz
         {"cmd":"STOP"}                            on zero / loss / shutdown
         {"cmd":"RESET_ENC"}  {"cmd":"CAL_IMU"}    via /bridge/command
    RX:  {"enc":..,"v_cmd":..,"cur_l":..,"cur_r":..,"bat_v":..,"e_stop":..,
          "ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..}  50 Hz

Safety:
  * Speed & steering clamped to conservative limits (parameters).
  * Zero Twist  → STOP.
  * Bridge watchdog: no /cmd_vel within cmd_timeout → STOP.
  * Explicit STOP on shutdown (SIGINT/SIGTERM and destroy_node).
  * dry_run parameter: logs frames instead of opening the serial port.
  * Auto-detects the ESP32 USB port by VID/PID (override with 'port').

Ackermann steering:
    δ = atan(wheelbase · ω / v)         (v floored to avoid singularity)
    s = clamp(δ / max_steer_angle, -1, 1)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import glob
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu, BatteryState

try:
    import serial  # pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# Known USB IDs for Arduino Nano ESP32 / common ESP32 USB-serial bridges
_ESP32_HINTS = [
    ("303a", "1001"),   # Espressif native USB (Nano ESP32)
    ("10c4", "ea60"),   # CP2102
    ("1a86", "7523"),   # CH340
    ("0403", "6001"),   # FT232
]


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__("serial_bridge_node")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("port", "auto")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("dry_run", False)

        # Kinematics / limits — START LOW, tune up after first test
        self.declare_parameter("max_speed_mps", 1.2)     # physical speed @ v=1.0
        self.declare_parameter("max_speed_cmd", 0.7)     # throttle cap (70%)
        self.declare_parameter("wheelbase_m", 0.30)      # front↔rear axle
        self.declare_parameter("max_steer_deg", 30.0)    # mechanical steer limit
        self.declare_parameter("min_speed_for_steer", 0.15)  # m/s singularity floor

        self.declare_parameter("cmd_rate_hz", 20.0)      # DRIVE send rate
        self.declare_parameter("cmd_timeout_s", 0.3)     # bridge watchdog

        g = lambda n: self.get_parameter(n).value
        self._baud       = int(g("baud"))
        self._dry        = bool(g("dry_run"))
        self._max_mps    = float(g("max_speed_mps"))
        self._max_cmd    = float(g("max_speed_cmd"))
        self._wheelbase  = float(g("wheelbase_m"))
        self._max_steer  = math.radians(float(g("max_steer_deg")))
        self._v_floor    = float(g("min_speed_for_steer"))
        self._rate       = float(g("cmd_rate_hz"))
        self._timeout    = float(g("cmd_timeout_s"))

        # ── State ──────────────────────────────────────────────────────
        self._v = 0.0
        self._s = 0.0
        self._last_cmd_t = 0.0
        self._stopped = True
        self._ser: Optional["serial.Serial"] = None
        self._rx_buf = ""
        self._lock = threading.Lock()

        # ── Serial ─────────────────────────────────────────────────────
        if not self._dry:
            self._open_serial(g("port"))
        else:
            self.get_logger().warn("DRY RUN — no serial port opened, frames logged only")

        # ── QoS: latest-only, reliable ─────────────────────────────────
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # ── Subscriptions ──────────────────────────────────────────────
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, qos)
        self.create_subscription(String, "/bridge/command", self._special_cb, 10)

        # ── Telemetry publishers ───────────────────────────────────────
        self._imu_pub  = self.create_publisher(Imu,          "/imu/data_raw",  10)
        self._batt_pub = self.create_publisher(BatteryState, "/battery_state", 10)
        self._enc_pub  = self.create_publisher(Float32,      "/wheel_encoder", 10)
        self._estop_pub= self.create_publisher(Bool,         "/e_stop",        10)

        # ── Timers ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate, self._send_loop)   # DRIVE @ 20Hz
        self.create_timer(0.01, self._read_loop)               # serial RX poll

        self.get_logger().info(
            f"Serial bridge ready | max_speed={self._max_mps}m/s cap={self._max_cmd} "
            f"wheelbase={self._wheelbase}m steer=±{math.degrees(self._max_steer):.0f}° "
            f"{'[DRY RUN]' if self._dry else ''}")

    # ──────────────────────────────────────────────────────────────────
    def _open_serial(self, port_param: str):
        if not HAS_SERIAL:
            self.get_logger().error("pyserial not installed: pip install pyserial")
            self._dry = True
            return
        port = port_param if port_param != "auto" else self._autodetect()
        if port is None:
            self.get_logger().error("No ESP32 port found — falling back to DRY RUN")
            self._dry = True
            return
        try:
            self._ser = serial.Serial(port, self._baud, timeout=0.0)
            time.sleep(2.0)   # ESP32 auto-resets on port open
            self._ser.reset_input_buffer()
            self.get_logger().info(f"Opened serial: {port} @ {self._baud}")
        except Exception as e:
            self.get_logger().error(f"Serial open failed ({port}): {e} — DRY RUN")
            self._dry = True

    def _autodetect(self) -> Optional[str]:
        """Match ESP32 by USB VID/PID via /sys, else first ttyACM/ttyUSB."""
        candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        if not candidates:
            return None
        # Try VID/PID match first
        try:
            import serial.tools.list_ports as lp
            for p in lp.comports():
                vid = f"{p.vid:04x}" if p.vid else ""
                pid = f"{p.pid:04x}" if p.pid else ""
                if (vid, pid) in _ESP32_HINTS:
                    self.get_logger().info(f"ESP32 matched by USB ID {vid}:{pid} → {p.device}")
                    return p.device
        except Exception:
            pass
        self.get_logger().warn(f"No VID/PID match; using first port {candidates[0]}")
        return candidates[0]

    # ──────────────────────────────────────────────────────────────────
    def _cmd_cb(self, msg: Twist):
        lin = msg.linear.x
        ang = msg.angular.z

        # Zero command → STOP
        if abs(lin) < 1e-3 and abs(ang) < 1e-3:
            with self._lock:
                self._v = 0.0
                self._s = 0.0
                self._last_cmd_t = time.monotonic()
            return

        # ── Throttle: m/s → normalized, capped ─────────────────────────
        v = lin / self._max_mps
        v = max(-self._max_cmd, min(self._max_cmd, v))

        # ── Ackermann steering ─────────────────────────────────────────
        v_eff = max(abs(lin), self._v_floor)              # avoid /0 at low speed
        steer_angle = math.atan(self._wheelbase * ang / v_eff)
        s = steer_angle / self._max_steer
        s = max(-1.0, min(1.0, s))

        with self._lock:
            self._v = v
            self._s = s
            self._last_cmd_t = time.monotonic()

    def _special_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd in ("RESET_ENC", "CAL_IMU"):
            self._write({"cmd": cmd})
            self.get_logger().info(f"Sent {cmd}")

    # ──────────────────────────────────────────────────────────────────
    def _send_loop(self):
        """20 Hz: send DRIVE, or STOP on watchdog timeout / zero."""
        now = time.monotonic()
        with self._lock:
            v, s, last_t = self._v, self._s, self._last_cmd_t

        # Bridge watchdog: no fresh command → STOP
        if (now - last_t) > self._timeout:
            v = s = 0.0

        if abs(v) < 1e-3 and abs(s) < 1e-3:
            if not self._stopped:
                self._write({"cmd": "STOP"})
                self._stopped = True
            return

        self._stopped = False
        self._write({"cmd": "DRIVE",
                     "v": round(v, 3),
                     "s": round(s, 3)})

    def _write(self, obj: dict):
        line = json.dumps(obj) + "\n"
        if self._dry or self._ser is None:
            self.get_logger().info(f"[DRY] → {line.strip()}")
            return
        try:
            self._ser.write(line.encode())
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    def _read_loop(self):
        if self._dry or self._ser is None:
            return
        try:
            n = self._ser.in_waiting
            if n:
                self._rx_buf += self._ser.read(n).decode(errors="ignore")
        except Exception:
            return
        while "\n" in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split("\n", 1)
            line = line.strip()
            if line.startswith("{"):
                self._parse_telemetry(line)

    def _parse_telemetry(self, line: str):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return
        if "status" in d:
            self.get_logger().info(f"ESP32: {d['status']}")
            return
        # IMU
        if "ax" in d:
            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = float(d.get("ax", 0.0))
            imu.linear_acceleration.y = float(d.get("ay", 0.0))
            imu.linear_acceleration.z = float(d.get("az", 0.0))
            imu.angular_velocity.x = float(d.get("gx", 0.0))
            imu.angular_velocity.y = float(d.get("gy", 0.0))
            imu.angular_velocity.z = float(d.get("gz", 0.0))
            self._imu_pub.publish(imu)
        # Battery
        if "bat_v" in d:
            b = BatteryState()
            b.header.stamp = self.get_clock().now().to_msg()
            b.voltage = float(d["bat_v"])
            b.present = True
            self._batt_pub.publish(b)
        # Encoder
        if "enc" in d:
            self._enc_pub.publish(Float32(data=float(d["enc"])))
        # E-stop
        if "e_stop" in d:
            self._estop_pub.publish(Bool(data=bool(d["e_stop"])))

    # ──────────────────────────────────────────────────────────────────
    def send_stop(self):
        """Force an immediate STOP — used on shutdown."""
        self._write({"cmd": "STOP"})

    def destroy_node(self):
        try:
            self.send_stop()
            time.sleep(0.05)
            self.send_stop()   # twice for reliability
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown — sending STOP")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
