#!/usr/bin/env python3
"""
local_planner_bridge_node.py                       (Raspberry Pi — a2r_bot_follow)
═══════════════════════════════════════════════════════════════════════════════
OWNER-FOLLOWING SERIAL BRIDGE WITH SWITCHABLE LOCAL PATH PLANNERS
for a 1:10 Ackermann RC car (Tamiya Wild Dagger WR-01 class chassis).

Pipeline
────────
  Jetson Nano (camera_node.py → tracker_node.py, same ROS_DOMAIN_ID + WiFi)
        │ /cmd_vel  (geometry_msgs/Twist)
        │   • points the car toward the owner
        │   • zero / silence when owner is inside the follow threshold
        ▼
  THIS NODE (RPi)
        /scan (LiDAR) ─► rolling local costmap + footprint-based INFLATION LAYER
        LOCAL PLANNER  ─► collision-free (v, steering) — 7 algorithms, switch by
                          comment/uncomment at "PLANNER DISPATCH" below:
                            1. DWA            (ACTIVE)
                            2. APF            Artificial Potential Field
                            3. TEB-style      Timed-Elastic-Band path deformation
                            4. VFH            Vector Field Histogram
                            5. Pure Pursuit   geometric lookahead tracking
                            6. FGM            Follow-the-Gap Method
                            7. LATTICE        two-segment arc lattice (predictive)
        SPEED / STALL MANAGER
            • speed never collapses below `min_drive_speed` while a path exists
              (linear clearance scaling with a floor — NOT exponential decay)
            • stall detector + REVERSE-ESCAPE: if boxed in or stalled, the car
              backs out along the least-cost reverse arc, swinging its nose
              toward the open side, then replans forward
        ▼
  USB serial ─► Arduino Nano ESP32   {"cmd":"DRIVE","v":..,"s":..}\n   @ 20 Hz
               (PCA9685 steering servo · BTS7960 motor · MPU6050 IMU telemetry)

RViz topics (see follow.launch.py / follow.rviz)
────────────────────────────────────────────────
  /local_costmap                 OccupancyGrid  obstacles + inflation layer
  /local_planner/path            nav_msgs/Path  chosen trajectory
  /local_planner/trajectories    MarkerArray    candidate arcs (grey/red)
  /local_planner/best_traj       MarkerArray    winning arc (green)
  /local_planner/desired_cmd     Marker         Jetson's raw wish (blue arrow)
  /local_planner/state           Marker         FOLLOW / ESCAPE-REVERSE text

Safety
──────
  • zero desired Twist → STOP (owner at threshold — the Jetson decides this)
  • /cmd_vel silence > cmd_timeout_s → STOP (tracker lost owner / link down)
  • every trajectory lethal AND reverse blocked → STOP (truly boxed in)
  • explicit double-STOP on shutdown; dry_run:=true for bench testing
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
import glob
import time
import threading
from typing import Optional, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, Imu, BatteryState
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray

try:
    import serial  # pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

_ESP32_HINTS = [
    ("303a", "1001"),   # Espressif native USB (Nano ESP32)
    ("2341", "0070"),   # Arduino VID for Nano ESP32
    ("10c4", "ea60"),   # CP2102
    ("1a86", "7523"),   # CH340
    ("0403", "6001"),   # FT232
]

COST_FREE      = 0
COST_INSCRIBED = 99
COST_LETHAL    = 100

# Planner state machine
ST_FOLLOW  = 0
ST_REVERSE = 1


class LocalPlannerBridgeNode(Node):

    # ══════════════════════════════════════════════════════════════════
    #  INIT
    # ══════════════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__("local_planner_bridge")

        # ── Serial / firmware ─────────────────────────────────────────
        self.declare_parameter("port", "/dev/ttyARDUINO")   # "auto" → VID/PID scan
        self.declare_parameter("baud", 921600)
        self.declare_parameter("dry_run", False)

        # ── Ackermann kinematics (WR-01: 31 cm wide, 25.7 cm wheelbase) ─
        self.declare_parameter("max_speed_mps",       1.2)
        self.declare_parameter("max_speed_cmd",       0.7)    # firmware throttle cap
        self.declare_parameter("wheelbase_m",         0.280)
        self.declare_parameter("max_steer_deg",       30.0)
        self.declare_parameter("min_speed_for_steer", 0.15)

        # ── Speed management (anti-stall) ─────────────────────────────
        self.declare_parameter("min_drive_speed",  0.28)  # m/s floor while a path exists
        self.declare_parameter("clearance_gain",   0.6)   # v = des·(0.4 + 0.6·clear)
        self.declare_parameter("stall_time_s",     1.5)   # stalled this long → escape
        self.declare_parameter("reverse_speed",    0.35)  # m/s during escape
        self.declare_parameter("reverse_time_s",   1.4)   # escape manoeuvre duration
        self.declare_parameter("reverse_cooldown_s", 2.0) # min gap between escapes

        # ── Rates / watchdog ──────────────────────────────────────────
        self.declare_parameter("cmd_rate_hz",   20.0)
        self.declare_parameter("cmd_timeout_s", 0.6)
        self.declare_parameter("plan_rate_hz",  10.0)

        # ── Topics ────────────────────────────────────────────────────
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic",    "/scan")

        # ── Costmap + inflation (footprint-based) ─────────────────────
        # Footprint: 0.31 m wide → inscribed (lethal) radius = half-width
        # + 4 cm safety = 0.195 m. Inflation extends a modest 0.25 m past
        # that so corridors ≥ ~0.9 m stay traversable — "reasonable", not
        # paralysing.
        self.declare_parameter("costmap_size_m",       4.0)
        self.declare_parameter("costmap_res_m",        0.05)
        self.declare_parameter("robot_radius_m",       0.195)
        self.declare_parameter("inflation_radius_m",   0.45)
        self.declare_parameter("inflation_cost_scale", 5.0)
        self.declare_parameter("costmap_frame",        "base_link")
        self.declare_parameter("lidar_offset_x",       0.0)
        self.declare_parameter("lidar_offset_y",       0.0)
        self.declare_parameter("lidar_yaw_offset",     0.0)   # A2M12 on this mount: 0.0 (set in params.yaml)
        self.declare_parameter("scan_max_range",       3.5)
        self.declare_parameter("scan_min_range",       0.30)

        # ── Shared planner settings ───────────────────────────────────
        self.declare_parameter("horizon_m",          1.3)  # path length examined
        self.declare_parameter("rollout_dt",         0.1)
        self.declare_parameter("lethal_cost_thresh", 90)

        # DWA
        self.declare_parameter("dwa_v_samples",     4)
        self.declare_parameter("dwa_steer_samples", 15)
        self.declare_parameter("dwa_w_cmd",         1.0)
        self.declare_parameter("dwa_w_clearance",   0.7)
        self.declare_parameter("dwa_w_speed",       0.35)

        g = lambda n: self.get_parameter(n).value
        self._baud       = int(g("baud"));            self._dry = bool(g("dry_run"))
        self._max_mps    = float(g("max_speed_mps")); self._max_cmd = float(g("max_speed_cmd"))
        self._wheelbase  = float(g("wheelbase_m"))
        self._max_steer  = math.radians(float(g("max_steer_deg")))
        self._v_floor    = float(g("min_speed_for_steer"))
        self._min_drive  = float(g("min_drive_speed"))
        self._clear_gain = float(g("clearance_gain"))
        self._stall_time = float(g("stall_time_s"))
        self._rev_speed  = float(g("reverse_speed"))
        self._rev_time   = float(g("reverse_time_s"))
        self._rev_cool   = float(g("reverse_cooldown_s"))
        self._rate       = float(g("cmd_rate_hz"));   self._timeout = float(g("cmd_timeout_s"))
        self._plan_rate  = float(g("plan_rate_hz"))
        self._map_size   = float(g("costmap_size_m")); self._res = float(g("costmap_res_m"))
        self._robot_r    = float(g("robot_radius_m"))
        self._infl_r     = float(g("inflation_radius_m"))
        self._infl_k     = float(g("inflation_cost_scale"))
        self._frame      = str(g("costmap_frame"))
        self._lx, self._ly = float(g("lidar_offset_x")), float(g("lidar_offset_y"))
        self._lyaw       = float(g("lidar_yaw_offset"))
        self._scan_max   = float(g("scan_max_range"))
        self._scan_min   = float(g("scan_min_range"))
        self._horizon_m  = float(g("horizon_m"));     self._dt = float(g("rollout_dt"))
        self._lethal     = int(g("lethal_cost_thresh"))
        self._n_v        = int(g("dwa_v_samples"));   self._n_s = int(g("dwa_steer_samples"))
        self._w_cmd      = float(g("dwa_w_cmd"))
        self._w_clear    = float(g("dwa_w_clearance"))
        self._w_speed    = float(g("dwa_w_speed"))

        # ── Costmap grid ──────────────────────────────────────────────
        self._cells = int(round(self._map_size / self._res))
        self._half  = self._map_size / 2.0
        self._grid  = np.zeros((self._cells, self._cells), dtype=np.int8)
        self._grid_lock = threading.Lock()
        self._build_inflation_offsets()

        # ── Latest scan in base frame (for FGM) ───────────────────────
        self._scan_pts: Optional[np.ndarray] = None   # (N,2) x,y
        self._scan_ang: Optional[np.ndarray] = None   # (N,)  bearings

        # ── Command / output state ────────────────────────────────────
        self._des_v = 0.0
        self._des_w = 0.0
        self._last_cmd_t = 0.0
        self._out_v = 0.0          # normalised [-1..1] for firmware
        self._out_s = 0.0
        self._stopped = True
        self._lock = threading.Lock()

        # ── Stall / escape state ──────────────────────────────────────
        self._state        = ST_FOLLOW
        self._stall_since  = None      # monotonic time when stall began
        self._rev_until    = 0.0       # escape ends at this time
        self._rev_steer    = 0.0
        self._last_rev_end = 0.0

        # ── Serial ────────────────────────────────────────────────────
        self._ser: Optional["serial.Serial"] = None
        self._rx_buf = ""
        if not self._dry:
            self._open_serial(str(g("port")))
        else:
            self.get_logger().warn("DRY RUN — frames logged only")

        # ── QoS ───────────────────────────────────────────────────────
        cmd_qos  = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        scan_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST)
        map_qos  = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST)

        # ── Subs / pubs ───────────────────────────────────────────────
        self.create_subscription(Twist, str(g("cmd_vel_topic")), self._cmd_cb, cmd_qos)
        self.create_subscription(LaserScan, str(g("scan_topic")), self._scan_cb, scan_qos)
        self.create_subscription(String, "/bridge/command", self._special_cb, 10)

        self._map_pub   = self.create_publisher(OccupancyGrid, "/local_costmap", map_qos)
        self._path_pub  = self.create_publisher(Path, "/local_planner/path", 1)
        self._trajs_pub = self.create_publisher(MarkerArray, "/local_planner/trajectories", 1)
        self._best_pub  = self.create_publisher(MarkerArray, "/local_planner/best_traj", 1)
        self._wish_pub  = self.create_publisher(Marker, "/local_planner/desired_cmd", 1)
        self._state_pub = self.create_publisher(Marker, "/local_planner/state", 1)
        self._imu_pub   = self.create_publisher(Imu,          "/imu/data_raw",  10)
        self._batt_pub  = self.create_publisher(BatteryState, "/battery_state", 10)
        self._enc_pub   = self.create_publisher(Float32,      "/wheel_encoder", 10)
        self._estop_pub = self.create_publisher(Bool,         "/e_stop",        10)

        # ── Timers ────────────────────────────────────────────────────
        self.create_timer(1.0 / self._plan_rate, self._plan_loop)
        self.create_timer(1.0 / self._rate,      self._send_loop)
        self.create_timer(0.01,                  self._read_loop)

        self.get_logger().info(
            f"LocalPlannerBridge ready | planner=DWA | costmap {self._map_size} m @ "
            f"{self._res} m | lethal r={self._robot_r} m infl={self._infl_r} m | "
            f"v∈[{self._min_drive},{self._max_mps}] m/s steer=±"
            f"{math.degrees(self._max_steer):.0f}° {'[DRY RUN]' if self._dry else ''}")

    # ══════════════════════════════════════════════════════════════════
    #  SERIAL  (JSON protocol of a2r_bot_drive.ino)
    # ══════════════════════════════════════════════════════════════════
    def _open_serial(self, port_param: str):
        if not HAS_SERIAL:
            self.get_logger().error("pyserial missing (pip install pyserial) — DRY RUN")
            self._dry = True
            return
        port = port_param if port_param != "auto" else self._autodetect()
        if port is None:
            self.get_logger().error("No Arduino port found — DRY RUN")
            self._dry = True
            return
        try:
            self._ser = serial.Serial(port, self._baud, timeout=0.0)
            time.sleep(2.0)                       # board auto-resets on open
            self._ser.reset_input_buffer()
            self.get_logger().info(f"Opened serial: {port} @ {self._baud}")
        except Exception as e:
            self.get_logger().error(f"Serial open failed ({port}): {e} — DRY RUN")
            self._dry = True

    def _autodetect(self) -> Optional[str]:
        candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        if not candidates:
            return None
        try:
            import serial.tools.list_ports as lp
            for p in lp.comports():
                vid = f"{p.vid:04x}" if p.vid else ""
                pid = f"{p.pid:04x}" if p.pid else ""
                if (vid, pid) in _ESP32_HINTS:
                    return p.device
        except Exception:
            pass
        return candidates[0]

    def _write(self, obj: dict):
        line = json.dumps(obj) + "\n"
        if self._dry or self._ser is None:
            self.get_logger().info(f"[DRY] → {line.strip()}", throttle_duration_sec=0.5)
            return
        try:
            self._ser.write(line.encode("ascii"))
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")

    def _special_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd in ("RESET_ENC", "CAL_IMU"):
            self._write({"cmd": cmd})

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
            self.get_logger().info(f"Arduino: {d['status']}")
            return
        ts = self.get_clock().now().to_msg()
        if "ax" in d:
            imu = Imu()
            imu.header.stamp = ts
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = float(d.get("ax", 0.0))
            imu.linear_acceleration.y = float(d.get("ay", 0.0))
            imu.linear_acceleration.z = float(d.get("az", 9.80665))
            imu.angular_velocity.x = float(d.get("gx", 0.0))
            imu.angular_velocity.y = float(d.get("gy", 0.0))
            imu.angular_velocity.z = float(d.get("gz", 0.0))
            imu.orientation_covariance[0] = -1.0
            self._imu_pub.publish(imu)
        if "bat_v" in d:
            b = BatteryState()
            b.header.stamp = ts
            b.voltage = float(d["bat_v"])
            b.present = True
            self._batt_pub.publish(b)
        if "enc" in d:
            self._enc_pub.publish(Float32(data=float(d["enc"])))
        if "e_stop" in d:
            self._estop_pub.publish(Bool(data=bool(d["e_stop"])))

    # ══════════════════════════════════════════════════════════════════
    #  DESIRED COMMAND from Jetson tracker
    # ══════════════════════════════════════════════════════════════════
    def _cmd_cb(self, msg: Twist):
        with self._lock:
            self._des_v = msg.linear.x
            self._des_w = msg.angular.z
            self._last_cmd_t = time.monotonic()

    # ══════════════════════════════════════════════════════════════════
    #  COSTMAP + INFLATION LAYER
    # ══════════════════════════════════════════════════════════════════
    def _build_inflation_offsets(self):
        """Vectorised inflation: for each kernel offset (di,dj) with cost c,
        the whole hit-mask is shifted and max-combined. cost(d):
            d ≤ robot_radius        → 100 (lethal — footprint would collide)
            robot_r < d ≤ infl_r    → 99·exp(-k·(d − robot_r))   (decay)
        """
        rc = int(math.ceil(self._infl_r / self._res))
        offs = []
        for di in range(-rc, rc + 1):
            for dj in range(-rc, rc + 1):
                d = math.hypot(di, dj) * self._res
                if d > self._infl_r:
                    continue
                if d <= self._robot_r:
                    c = COST_LETHAL
                else:
                    c = int(round(COST_INSCRIBED
                                  * math.exp(-self._infl_k * (d - self._robot_r))))
                    c = max(1, min(COST_INSCRIBED - 1, c))
                offs.append((di, dj, c))
        self._infl_offs = offs

    def _scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        a = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        # Full scan for FGM: out-of-range / inf rays count as FREE space,
        # clipped to scan_max (a long ray is an open gap, not an obstacle)
        min_r = max(msg.range_min, self._scan_min)
        r_full = np.where(np.isfinite(r) & (r > min_r), r, self._scan_max)
        r_full = np.minimum(r_full, self._scan_max)
        ab_full = a + self._lyaw
        xf = self._lx + r_full * np.cos(ab_full)
        yf = self._ly + r_full * np.sin(ab_full)
        self._scan_ang = np.arctan2(yf, xf)
        self._scan_rng = np.hypot(xf, yf)
        # Costmap only marks REAL hits (inside scan_max)
        valid = np.isfinite(r) & (r > min_r) & \
                (r < min(msg.range_max, self._scan_max))
        r, a = r[valid], a[valid]
        # lidar frame → base_link (fixed mounting offset, no TF lookup needed)
        ab = a + self._lyaw
        x = self._lx + r * np.cos(ab)
        y = self._ly + r * np.sin(ab)
        self._scan_pts = np.stack([x, y], axis=1)

        # Hit mask
        i = ((x + self._half) / self._res).astype(np.int32)
        j = ((y + self._half) / self._res).astype(np.int32)
        ok = (i >= 0) & (i < self._cells) & (j >= 0) & (j < self._cells)
        hits = np.zeros((self._cells, self._cells), dtype=np.int16)
        hits[i[ok], j[ok]] = 1

        # Vectorised inflation via shifted-max
        grid = np.zeros_like(hits)
        C = self._cells
        for di, dj, c in self._infl_offs:
            si0, si1 = max(0, di),  C + min(0, di)
            ti0, ti1 = max(0, -di), C + min(0, -di)
            sj0, sj1 = max(0, dj),  C + min(0, dj)
            tj0, tj1 = max(0, -dj), C + min(0, -dj)
            src = hits[ti0:ti1, tj0:tj1]
            tgt = grid[si0:si1, sj0:sj1]
            np.maximum(tgt, src * c, out=tgt)

        with self._grid_lock:
            self._grid = np.clip(grid, 0, 100).astype(np.int8)
        self._publish_costmap(msg.header.stamp)

    def _publish_costmap(self, stamp):
        m = OccupancyGrid()
        m.header.stamp = stamp
        m.header.frame_id = self._frame
        m.info.resolution = self._res
        m.info.width = m.info.height = self._cells
        m.info.origin.position.x = -self._half
        m.info.origin.position.y = -self._half
        m.info.origin.orientation.w = 1.0
        with self._grid_lock:
            m.data = self._grid.T.flatten().astype(np.int8).tolist()
        self._map_pub.publish(m)

    def _cost_at(self, x: float, y: float) -> int:
        i = int((x + self._half) / self._res)
        j = int((y + self._half) / self._res)
        if 0 <= i < self._cells and 0 <= j < self._cells:
            return int(self._grid[i, j])
        return COST_FREE

    # ══════════════════════════════════════════════════════════════════
    #  SHARED PLANNER HELPERS
    # ══════════════════════════════════════════════════════════════════
    def _desired_steer(self, des_v: float, des_w: float) -> float:
        v_eff = max(abs(des_v), self._v_floor)
        return max(-self._max_steer, min(self._max_steer,
                   math.atan(self._wheelbase * des_w / v_eff)))

    def _rollout(self, v: float, steer: float,
                 dist: Optional[float] = None) -> Tuple[List[Tuple[float, float]], int]:
        """Constant-(v,steer) Ackermann arc. FIXED DISTANCE horizon so slow
        candidates cannot hide a collision beyond a short time window.
        Works for reverse (v<0) too."""
        if dist is None:
            dist = self._horizon_m
        x = y = yaw = 0.0
        pts, max_c = [], 0
        speed = max(abs(v), 0.05)
        steps = int(dist / (speed * self._dt))
        w = (v / self._wheelbase) * math.tan(steer)
        for _ in range(steps):
            yaw += w * self._dt
            x += v * math.cos(yaw) * self._dt
            y += v * math.sin(yaw) * self._dt
            pts.append((x, y))
            c = self._cost_at(x, y)
            if c > max_c:
                max_c = c
                if max_c >= self._lethal:
                    break
        return pts, max_c

    def _speed_law(self, des_v: float, max_cost: int) -> float:
        """Anti-stall speed shaping. Clearance scales speed LINEARLY with a
        hard floor: v = des·(1−gain + gain·clear), then clamped ≥ min_drive.
        With gain=0.6 even max_cost=89 keeps ≥40 % of desired speed."""
        clear = 1.0 - max_cost / 100.0
        v = abs(des_v) * ((1.0 - self._clear_gain) + self._clear_gain * clear)
        v = min(v, self._max_mps)
        return max(v, self._min_drive)

    def _side_openness(self) -> float:
        """+1 = left much freer, −1 = right much freer (front semicircle)."""
        with self._grid_lock:
            grid = self._grid.astype(np.float32)
        C, h = self._cells, self._cells // 2
        front = grid[h:, :]                      # x > 0 rows
        left  = float(front[:, h:].sum())        # y > 0
        right = float(front[:, :h].sum())        # y < 0
        tot = left + right + 1e-6
        return (right - left) / tot              # right blocked → positive → go left

    # ══════════════════════════════════════════════════════════════════
    #  PLANNER LOOP + STALL / REVERSE-ESCAPE MANAGER
    # ══════════════════════════════════════════════════════════════════
    def _plan_loop(self):
        now = time.monotonic()
        with self._lock:
            des_v, des_w = self._des_v, self._des_w
            fresh = (now - self._last_cmd_t) <= self._timeout

        # Owner reached (Jetson sends zero) or tracker silent → STOP
        if (not fresh) or (abs(des_v) < 1e-3 and abs(des_w) < 1e-3):
            self._set_out(0.0, 0.0)
            self._state = ST_FOLLOW
            self._stall_since = None
            self._publish_wish(0.0, 0.0)
            self._publish_state("IDLE")
            return

        self._publish_wish(des_v, des_w)

        # ── ESCAPE-REVERSE state: execute the manoeuvre until timer ends ─
        if self._state == ST_REVERSE:
            if now < self._rev_until:
                self._set_out(-self._rev_speed, self._rev_steer)
                self._publish_state("ESCAPE-REVERSE")
                return
            self._state = ST_FOLLOW
            self._stall_since = None
            self._last_rev_end = now

        # ─────────────────────────────────────────────────────────────
        #  PLANNER DISPATCH — exactly ONE line active. All seven are
        #  implemented below; swap by commenting/uncommenting.
        # ─────────────────────────────────────────────────────────────
        v, steer = self._plan_dwa(des_v, des_w)            # 1. DWA  ← ACTIVE
        # v, steer = self._plan_apf(des_v, des_w)          # 2. Artificial Potential Field
        # v, steer = self._plan_teb(des_v, des_w)          # 3. TEB-style elastic band
        # v, steer = self._plan_vfh(des_v, des_w)          # 4. Vector Field Histogram
        # v, steer = self._plan_pure_pursuit(des_v, des_w) # 5. Pure Pursuit
        # v, steer = self._plan_fgm(des_v, des_w)          # 6. Follow-the-Gap
        # v, steer = self._plan_lattice(des_v, des_w)      # 7. Arc-lattice (predictive)

        # ── Stall detection → trigger reverse escape ──────────────────
        blocked = abs(v) < 1e-3
        if blocked:
            if self._stall_since is None:
                self._stall_since = now
            stalled = (now - self._stall_since) >= self._stall_time
            can_escape = (now - self._last_rev_end) >= self._rev_cool
            if (stalled or self._front_blocked()) and can_escape:
                self._begin_escape(now)
                return
        else:
            self._stall_since = None

        self._set_out(v, steer)
        self._publish_state("FOLLOW")

    def _front_blocked(self) -> bool:
        """Immediate front arc lethal at min speed → no forward candidate exists."""
        _, c = self._rollout(self._min_drive, 0.0, dist=0.45)
        return c >= self._lethal

    def _begin_escape(self, now: float):
        """Choose the least-cost reverse arc. Steering toward the MORE
        blocked side swings the nose toward the open side while backing."""
        open_side = self._side_openness()          # +1 → left open
        # candidate reverse steers, biased so nose rotates toward open side
        cands = sorted(np.linspace(-self._max_steer, self._max_steer, 7),
                       key=lambda s: abs(s - (-open_side * self._max_steer)))
        best_s, best_c = 0.0, 101
        for s in cands:
            _, c = self._rollout(-self._rev_speed, s, dist=0.5)
            if c < best_c:
                best_c, best_s = c, s
            if c < self._lethal:
                break
        if best_c >= self._lethal:
            # Reverse also blocked → truly boxed in, stop and wait
            self._set_out(0.0, 0.0)
            self._publish_state("BOXED-IN (STOP)")
            self.get_logger().warn("Boxed in: forward and reverse blocked — stopping",
                                   throttle_duration_sec=2.0)
            return
        self._state = ST_REVERSE
        self._rev_steer = best_s
        self._rev_until = now + self._rev_time
        self.get_logger().info(
            f"Stall → ESCAPE-REVERSE (steer {math.degrees(best_s):+.0f}°, "
            f"{self._rev_time}s)")

    def _set_out(self, v: float, steer: float):
        v_n = max(-self._max_cmd, min(self._max_cmd, v / self._max_mps))
        s_n = max(-1.0, min(1.0, steer / self._max_steer))
        with self._lock:
            self._out_v, self._out_s = v_n, s_n

    # ══════════════════════════════════════════════════════════════════
    #  1) DWA — Dynamic Window Approach                        [ACTIVE]
    # ══════════════════════════════════════════════════════════════════
    def _plan_dwa(self, des_v: float, des_w: float) -> Tuple[float, float]:
        des_steer = self._desired_steer(des_v, des_w)
        des_v_c = max(-self._max_mps, min(self._max_mps, des_v))

        sign = 1.0 if des_v_c >= 0 else -1.0
        v_lo = max(self._min_drive, 0.5 * abs(des_v_c))
        vs = sign * np.linspace(v_lo, max(v_lo, abs(des_v_c)), self._n_v)
        ss = np.linspace(-self._max_steer, self._max_steer, self._n_s)

        best, best_score, best_pts, best_cost = None, -1e9, [], 100
        markers = MarkerArray()
        mid = 0
        for v in vs:
            for s in ss:
                pts, max_c = self._rollout(float(v), float(s))
                lethal = max_c >= self._lethal
                if not lethal:
                    sim = 1.0 - (abs(s - des_steer) / (2 * self._max_steer)) \
                              - 0.5 * abs(v - des_v_c) / self._max_mps
                    clear = 1.0 - max_c / 100.0
                    spd = abs(v) / self._max_mps
                    score = self._w_cmd * sim + self._w_clear * clear + self._w_speed * spd
                    if score > best_score:
                        best_score = score
                        best, best_pts, best_cost = (float(v), float(s)), pts, max_c
                markers.markers.append(self._traj_marker(pts, mid, lethal=lethal))
                mid += 1
        self._trajs_pub.publish(markers)

        if best is None:
            self._publish_best([])
            return 0.0, 0.0            # stall manager decides on reverse escape

        v_out = self._speed_law(best[0], best_cost) * (1 if best[0] >= 0 else -1)
        self._publish_best(best_pts)
        return v_out, best[1]

    # ══════════════════════════════════════════════════════════════════
    #  2) APF — Artificial Potential Field
    # ══════════════════════════════════════════════════════════════════
    def _plan_apf(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Attractive vector along the desired heading; repulsive vectors from
        occupied cells out to the planning horizon (not just the inflation
        radius — the car must react BEFORE an obstacle is on top of it).
        A dead-ahead obstacle creates a symmetric field with zero lateral
        push, so if the resultant arc is lethal the steer slides toward the
        open side until a free arc is found."""
        K_ATT, K_REP = 1.0, 0.55
        REP_RANGE = self._horizon_m               # react within the horizon
        des_steer = self._desired_steer(des_v, des_w)
        fx, fy = K_ATT * math.cos(des_steer), K_ATT * math.sin(des_steer)
        with self._grid_lock:
            occ = np.argwhere(self._grid >= COST_INSCRIBED)
        for i, j in occ:
            ox = i * self._res - self._half
            oy = j * self._res - self._half
            d = math.hypot(ox, oy)
            if 0.05 < d < REP_RANGE:
                mag = K_REP * (1.0 / d - 1.0 / REP_RANGE) / (d * d)
                fx -= mag * ox / d
                fy -= mag * oy / d
        heading = math.atan2(fy, fx)
        base_steer = max(-self._max_steer, min(self._max_steer, heading))
        # Symmetry breaker: slide toward the open side if the arc is lethal
        open_sign = 1.0 if self._side_openness() >= 0 else -1.0
        for dσ in (0.0, 0.15, 0.30, 0.45, 0.60):
            steer = max(-self._max_steer,
                        min(self._max_steer, base_steer + open_sign * dσ))
            pts, max_c = self._rollout(max(abs(des_v), self._min_drive), steer)
            if max_c < self._lethal:
                self._publish_best(pts)
                return self._speed_law(des_v, max_c), steer
        self._publish_best([])
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════════
    #  3) TEB-style — Timed-Elastic-Band path deformation
    # ══════════════════════════════════════════════════════════════════
    def _plan_teb(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Seed a path along the desired arc, then iteratively deform it:
        each waypoint is pushed down the local cost gradient (obstacle force)
        and pulled toward its neighbours' midpoint (elasticity). The deformed
        band is then tracked with the Pure-Pursuit steering law. This mirrors
        TEB's obstacle/smoothness optimisation in a light-weight form suitable
        for the Pi (real TEB is a C++ Nav2 plugin)."""
        N, ITER, STEP, ELASTIC = 10, 10, 0.06, 0.20
        GRAD_SCALE = 25.0          # gradient force divisor (was too weak at 100)
        des_steer = self._desired_steer(des_v, des_w)
        v_seed = max(abs(des_v), self._min_drive)
        seg = self._horizon_m / N

        def seed_band(bow: float):
            """Seed along the desired arc with a lateral bow (metres). The
            profile sin(π(k+1)/(N+1)) keeps the ENDPOINT bowed too — pinning
            it at y=0 would park it inside a dead-ahead obstacle. Left/right
            bows emulate TEB's homotopy-class exploration and break the
            symmetric saddle of a centred obstacle."""
            band, x, y, yaw = [], 0.0, 0.0, 0.0
            w = (v_seed / self._wheelbase) * math.tan(des_steer)
            for k in range(N):
                yaw += w * (seg / v_seed)
                x += seg * math.cos(yaw)
                y += seg * math.sin(yaw)
                lat = bow * math.sin(math.pi * (k + 1) / (N + 1))
                band.append([x, y + lat])
            return band

        def deform(band):
            eps = self._res
            for _ in range(ITER):
                for k in range(1, N):
                    bx, by = band[k]
                    gx = (self._cost_at(bx + eps, by) - self._cost_at(bx - eps, by))
                    gy = (self._cost_at(bx, by + eps) - self._cost_at(bx, by - eps))
                    band[k][0] -= STEP * gx / GRAD_SCALE
                    band[k][1] -= STEP * gy / GRAD_SCALE
                    if 0 < k < N - 1:
                        mx = 0.5 * (band[k - 1][0] + band[k + 1][0])
                        my = 0.5 * (band[k - 1][1] + band[k + 1][1])
                        band[k][0] += ELASTIC * (mx - band[k][0])
                        band[k][1] += ELASTIC * (my - band[k][1])
            return band, max((self._cost_at(px, py) for px, py in band), default=0)

        # Three homotopy candidates: straight, bow-left, bow-right.
        # Bow 0.45 m clears the widest inflated obstacle the horizon can meet.
        cands = [deform(seed_band(b)) for b in (0.0, +0.45, -0.45)]
        cands = [(b, c) for b, c in cands if c < self._lethal]
        if not cands:
            self._publish_best([])
            return 0.0, 0.0
        band, max_c = min(cands, key=lambda bc: bc[1])
        # Track first reachable waypoint ≥ lookahead with pure pursuit law
        LOOK = 0.55
        tgt = band[-1]
        for p in band:
            if math.hypot(p[0], p[1]) >= LOOK:
                tgt = p
                break
        alpha = math.atan2(tgt[1], tgt[0])
        Ld = max(math.hypot(tgt[0], tgt[1]), 0.2)
        steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), Ld)
        steer = max(-self._max_steer, min(self._max_steer, steer))
        max_c = max((self._cost_at(px, py) for px, py in band), default=0)
        if max_c >= self._lethal:
            # Band could not escape the obstacle → let stall manager act
            self._publish_best([])
            return 0.0, 0.0
        self._publish_best([(p[0], p[1]) for p in band])
        return self._speed_law(des_v, max_c), steer

    # ══════════════════════════════════════════════════════════════════
    #  4) VFH — Vector Field Histogram
    # ══════════════════════════════════════════════════════════════════
    def _plan_vfh(self, des_v: float, des_w: float) -> Tuple[float, float]:
        N_SECT, THRESH = 36, 25.0
        des_heading = self._desired_steer(des_v, des_w)
        hist = np.zeros(N_SECT)
        with self._grid_lock:
            occ = np.argwhere(self._grid >= COST_INSCRIBED)
            grid = self._grid
        for i, j in occ:
            ox = i * self._res - self._half
            oy = j * self._res - self._half
            d = math.hypot(ox, oy)
            if d < 0.05 or d > self._half:
                continue
            sect = int(((math.atan2(oy, ox) + math.pi) / (2 * math.pi)) * N_SECT) % N_SECT
            hist[sect] += float(grid[i, j]) * (1.0 - d / self._half)
        best_sect, best_diff = None, 1e9
        for k in range(N_SECT):
            ang = -math.pi + (k + 0.5) * 2 * math.pi / N_SECT
            if abs(ang) > self._max_steer * 1.6 or hist[k] > THRESH:
                continue
            diff = abs(ang - des_heading)
            if diff < best_diff:
                best_diff, best_sect = diff, k
        if best_sect is None:
            self._publish_best([])
            return 0.0, 0.0
        ang = -math.pi + (best_sect + 0.5) * 2 * math.pi / N_SECT
        steer = max(-self._max_steer, min(self._max_steer, ang))
        pts, max_c = self._rollout(max(abs(des_v), self._min_drive), steer)
        if max_c >= self._lethal:
            self._publish_best([])
            return 0.0, 0.0
        self._publish_best(pts)
        return self._speed_law(des_v, max_c), steer

    # ══════════════════════════════════════════════════════════════════
    #  5) PURE PURSUIT — geometric lookahead with lateral sliding
    # ══════════════════════════════════════════════════════════════════
    def _plan_pure_pursuit(self, des_v: float, des_w: float) -> Tuple[float, float]:
        LOOK = 0.8
        alpha0 = self._desired_steer(des_v, des_w)
        for dalpha in (0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6):
            alpha = alpha0 + dalpha
            gx, gy = LOOK * math.cos(alpha), LOOK * math.sin(alpha)
            steer = math.atan2(2.0 * self._wheelbase * math.sin(alpha), LOOK)
            steer = max(-self._max_steer, min(self._max_steer, steer))
            pts, max_c = self._rollout(max(abs(des_v), self._min_drive), steer)
            if max_c < self._lethal and self._cost_at(gx, gy) < self._lethal:
                self._publish_best(pts)
                return self._speed_law(des_v, max_c), steer
        self._publish_best([])
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════════
    #  6) FGM — Follow-the-Gap Method
    # ══════════════════════════════════════════════════════════════════
    def _plan_fgm(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Classic reactive planner on the raw scan: find the widest angular
        gap of ranges beyond a safety distance in the front ±100° cone, aim at
        the gap point weighted toward the desired heading."""
        if self._scan_ang is None or len(self._scan_ang) == 0:
            return 0.0, 0.0
        SAFE_D = self._robot_r + 0.35
        des_heading = self._desired_steer(des_v, des_w)
        ang, rng = self._scan_ang, self._scan_rng
        cone = np.abs(ang) <= math.radians(100)
        a, r = ang[cone], rng[cone]
        if len(a) == 0:
            return 0.0, 0.0
        order = np.argsort(a)
        a, r = a[order], r[order]
        # Dynamic free/blocked threshold (F1TENTH-style): anything not clearly
        # beyond the closest obstacle is "blocked", so gaps form BESIDE an
        # obstacle within the horizon rather than straight through it.
        closest = float(np.min(r))
        thresh = max(SAFE_D, min(self._horizon_m, closest + 0.45))
        free = r > thresh
        # widest contiguous free run
        best_len, best_i0, cur_len, cur_i0 = 0, None, 0, 0
        for idx, f in enumerate(free):
            if f:
                if cur_len == 0:
                    cur_i0 = idx
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_i0 = cur_len, cur_i0
            else:
                cur_len = 0
        if best_i0 is None or best_len < 3:
            self._publish_best([])
            return 0.0, 0.0
        gap = a[best_i0: best_i0 + best_len]
        # Aim: 70 % gap centre, 30 % pull toward desired heading (if inside gap)
        centre = float(np.mean(gap))
        target = 0.7 * centre + 0.3 * float(np.clip(des_heading, gap[0], gap[-1]))
        steer = max(-self._max_steer, min(self._max_steer, target))
        pts, max_c = self._rollout(max(abs(des_v), self._min_drive), steer)
        if max_c >= self._lethal:
            self._publish_best([])
            return 0.0, 0.0
        self._publish_best(pts)
        return self._speed_law(des_v, max_c), steer

    # ══════════════════════════════════════════════════════════════════
    #  7) LATTICE — two-segment arc lattice (predictive)
    # ══════════════════════════════════════════════════════════════════
    def _plan_lattice(self, des_v: float, des_w: float) -> Tuple[float, float]:
        """Receding-horizon search over TWO consecutive steering segments
        (0.65 m each). Looks past the first obstacle to check whether an
        avoidance arc actually leads somewhere useful — anticipates walls
        instead of reacting to them. Executes only segment one."""
        SEG = 0.65
        des_steer = self._desired_steer(des_v, des_w)
        v = max(abs(des_v), self._min_drive)
        s_set = np.linspace(-self._max_steer, self._max_steer, 9)
        best, best_score, best_pts, best_cost = None, -1e9, [], 100
        for s1 in s_set:
            pts1, c1 = self._rollout(v, float(s1), dist=SEG)
            if c1 >= self._lethal or not pts1:
                continue
            x1, y1 = pts1[-1]
            yaw1 = (v / self._wheelbase) * math.tan(s1) * (SEG / v)
            for s2 in s_set[::2]:
                # roll segment 2 from the end pose of segment 1
                x, y, yaw = x1, y1, yaw1
                w2 = (v / self._wheelbase) * math.tan(s2)
                c2 = c1
                pts2 = []
                steps = int(SEG / (v * self._dt))
                for _ in range(steps):
                    yaw += w2 * self._dt
                    x += v * math.cos(yaw) * self._dt
                    y += v * math.sin(yaw) * self._dt
                    pts2.append((x, y))
                    c2 = max(c2, self._cost_at(x, y))
                    if c2 >= self._lethal:
                        break
                if c2 >= self._lethal:
                    continue
                sim = 1.0 - abs(s1 - des_steer) / (2 * self._max_steer) \
                          - 0.4 * abs(s2 - des_steer) / (2 * self._max_steer)
                clear = 1.0 - c2 / 100.0
                score = sim + 0.7 * clear
                if score > best_score:
                    best_score = score
                    best, best_pts, best_cost = float(s1), pts1 + pts2, c2
        if best is None:
            self._publish_best([])
            return 0.0, 0.0
        self._publish_best(best_pts)
        return self._speed_law(des_v, best_cost), best

    # ══════════════════════════════════════════════════════════════════
    #  20 Hz DRIVE SENDER
    # ══════════════════════════════════════════════════════════════════
    def _send_loop(self):
        with self._lock:
            v, s = self._out_v, self._out_s
        if abs(v) < 1e-3 and abs(s) < 1e-3:
            if not self._stopped:
                self._write({"cmd": "STOP"})
                self._stopped = True
            return
        self._stopped = False
        self._write({"cmd": "DRIVE", "v": round(v, 3), "s": round(s, 3)})

    # ══════════════════════════════════════════════════════════════════
    #  RViz OUTPUT
    # ══════════════════════════════════════════════════════════════════
    def _traj_marker(self, pts, mid: int, lethal=False, best=False) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "best" if best else "candidates"
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD if pts else Marker.DELETE
        m.scale.x = 0.035 if best else 0.008
        if best:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        elif lethal:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.30
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.6, 0.45
        m.lifetime.nanosec = int(0.4e9)
        m.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in pts]
        return m

    def _publish_best(self, pts):
        self._best_pub.publish(MarkerArray(
            markers=[self._traj_marker(pts, 0, best=True)]))
        path = Path()
        path.header.frame_id = self._frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in pts:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self._path_pub.publish(path)

    def _publish_wish(self, v: float, w: float):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "desired", 0, Marker.ARROW, Marker.ADD
        v_eff = max(abs(v), self._v_floor)
        ang = math.atan(self._wheelbase * w / v_eff) if abs(v) + abs(w) > 1e-3 else 0.0
        L = 0.3 + 0.6 * min(abs(v) / self._max_mps, 1.0)
        m.points = [Point(x=0.0, y=0.0, z=0.06),
                    Point(x=L * math.cos(ang), y=L * math.sin(ang), z=0.06)]
        m.scale.x, m.scale.y, m.scale.z = 0.02, 0.05, 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.9
        self._wish_pub.publish(m)

    def _publish_state(self, text: str):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "state", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.z = 0.6
        m.scale.z = 0.12
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.text = text
        self._state_pub.publish(m)

    # ══════════════════════════════════════════════════════════════════
    def destroy_node(self):
        try:
            self._write({"cmd": "STOP"})
            time.sleep(0.05)
            self._write({"cmd": "STOP"})
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown — sending STOP")
        node.destroy_node()
        if rclpy.ok():          # SIGINT handler may have already shut down
            rclpy.shutdown()


if __name__ == "__main__":
    main()
