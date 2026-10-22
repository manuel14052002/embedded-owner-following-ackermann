#!/usr/bin/env python3
"""FollowMe ROS 2 node (Humble): perception -> state machine -> Nav2.

Subscribes
    color_topic (sensor_msgs/Image), depth_topic (aligned to colour, 16UC1 mm or
    32FC1 m), camera_info_topic (sensor_msgs/CameraInfo of the colour stream)
Uses
    tf2: global_frame <- camera optical frame (Eq. 2), global_frame <- base_frame (search angle)
    Nav2 NavigateToPose action (goals), cmd_vel (search rotation / stop)
Publishes
    ~/state (std_msgs/String), ~/target_pose (PoseStamped, global frame),
    ~/safety_circle (visualization_msgs/Marker), ~/debug_image (Image, optional)
Services (std_srvs/Trigger)
    ~/calibrate  start calibration (the target alone in front of the camera)
    ~/save_target  save the calibrated target model
    ~/follow, ~/wait  inject gesture commands
    ~/reset  drop the calibration and restart the state machine

The heavy perception runs in a worker thread on the newest synchronised
frame pair; older frames are dropped rather than queued.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

import message_filters
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformException, TransformListener

from followme_ros.singleton import check_singleton, release_singleton
from followme_ros.web_stream import MjpegServer
from followme_ros.minimap import draw_phone  # a2r_phone_view
from followme_ros.minimap import draw_gps  # a2r_step3d
from followme_ros.minimap import (composite, draw_ground_path, project_ground_path, render_minimap,
                                  scan_to_base)
from followme_ros.conversions import (depth_to_meters, image_to_bgr, transform_to_matrix, yaw_from_matrix,
                                      yaw_to_quaternion)

# The FollowMe library lives in the owner_tracker repository: `pip install -e <repo>`
# or point FOLLOWME_ROOT at it.
if os.environ.get("FOLLOWME_ROOT"):
    sys.path.insert(0, os.environ["FOLLOWME_ROOT"])

from followme.config import FollowMeConfig  # noqa: E402
from followme.factory import build_perception  # noqa: E402
from followme.gestures import FOLLOW, WAIT  # noqa: E402
from followme.localization import CameraIntrinsics, Goal2D  # noqa: E402
from followme.direct_control import DirectFollowController  # noqa: E402
from followme.localization import heading_angle, safety_radius, target_position  # noqa: E402
from followme.navigation import NavigationBackend, RecordingNavigation  # noqa: E402
from followme.ownership import OwnershipManager  # noqa: E402
from followme.pipeline import OPTICAL_TO_BASE, FollowMeController  # noqa: E402
from followme.visualization import draw  # noqa: E402
from owner_tracking.runtime import resolve_device  # noqa: E402


class Nav2Backend(NavigationBackend):
    """NavigationBackend on Nav2's NavigateToPose action and cmd_vel."""

    def __init__(self, node: Node, frame: str, action_name: str, cmd_vel_topic: str) -> None:
        self.node = node
        self.frame = frame
        self.client = ActionClient(node, NavigateToPose, action_name)
        self.cmd_pub = node.create_publisher(Twist, cmd_vel_topic, 10)
        self._handle = None
        self._lock = threading.Lock()
        self._warned = False

    def send_goal(self, goal: Goal2D) -> None:
        if not self.client.server_is_ready():
            if not self._warned:
                self.node.get_logger().warn("Nav2 action server not ready; goal dropped")
                self._warned = True
            return
        self._warned = False
        msg = NavigateToPose.Goal()
        msg.pose = PoseStamped()
        msg.pose.header.frame_id = self.frame
        msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = goal.x, goal.y
        q = yaw_to_quaternion(goal.yaw)
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        future = self.client.send_goal_async(msg)          # a new goal pre-empts the old one in Nav2
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        with self._lock:
            self._handle = handle if handle is not None and handle.accepted else None

    def cancel_goal(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            handle.cancel_goal_async()

    def set_velocity(self, linear: float, angular: float) -> None:
        t = Twist()
        t.linear.x, t.angular.z = float(linear), float(angular)
        self.cmd_pub.publish(t)


class CmdVelBackend(NavigationBackend):
    """Direct following: every goal (owner in the base frame) becomes a Twist on cmd_vel."""

    continuous = True

    def __init__(self, node: Node, topic: str, controller: DirectFollowController,
                 angular_sign: float = 1.0) -> None:
        self.pub = node.create_publisher(Twist, topic, 10)
        self.angular_sign = angular_sign        # -1 if the base steers the other way
        self.ctrl = controller
        self.last_cmd_t = 0.0
        self.moving = False
        self._lock = threading.Lock()

    muted = False               # True: another node (external search planner) owns /cmd_vel

    def _publish(self, v: float, w: float) -> None:
        if self.muted:
            return
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w) * self.angular_sign
        with self._lock:
            self.pub.publish(t)
            self.last_twist = t
            self.moving = abs(v) > 1e-6 or abs(w) > 1e-6

    allow_forward = True        # set every frame by the node (fresh owner + nothing too close)

    def send_goal(self, goal: Goal2D) -> None:
        self.last_cmd_t = time.monotonic()
        v, w = self.ctrl.command(goal, time.monotonic())
        if not self.allow_forward:
            v = 0.0                             # may still steer toward the owner, but not drive
            self.ctrl.v = 0.0
        self._publish(v, w)

    def cancel_goal(self) -> None:
        self.ctrl.reset()

    obstacle_clear = True       # set every frame: False while the camera/lidar stop is active

    def set_velocity(self, linear: float, angular: float) -> None:
        self.last_cmd_t = time.monotonic()
        if linear > 0.0 and not self.obstacle_clear:
            linear = 0.0                        # search / arc never drive into an obstacle
        if linear == 0.0 and angular == 0.0:
            self.ctrl.reset()
        self._publish(linear, angular)

    def watchdog(self, timeout_s: float) -> None:
        """Called at 20 Hz: repeat the last command while it is fresh (keeps the base's own
        watchdog happy between perception frames); zero velocity once it is stale."""
        if not self.moving or self.muted:
            return
        if time.monotonic() - self.last_cmd_t > timeout_s:
            self.ctrl.reset()
            self._publish(0.0, 0.0)
        else:
            with self._lock:
                self.pub.publish(self.last_twist)


VOICE_ACTIONS = {
    "follow": "follow", "follow_me": "follow", "start": "follow", "come": "follow", "go": "follow",
    "stop": "wait", "wait": "wait", "stay": "wait", "halt": "wait", "pause": "wait",
    "release": "release", "release_owner": "release", "free": "release",
    "enroll": "enroll", "enroll_owner": "enroll", "calibrate": "enroll", "learn_me": "enroll",
    "new_owner": "enroll", "claim": "enroll", "register": "enroll", "remember_me": "enroll",
    # voice_bridge.py (set_robot_command enum)
    "wake_trigger": "enroll", "go_to": "follow", "idle": "wait", "circular": "circular",
}


def parse_voice(data: str) -> Optional[str]:
    """JSON {"action": ...} / {"command": ...} or plain text -> follow | wait | release | enroll."""
    raw = data.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            raw = str(obj.get("action") or obj.get("command") or obj.get("cmd") or "")
        elif isinstance(obj, str):
            raw = obj
    except (ValueError, TypeError):
        pass
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return VOICE_ACTIONS.get(key)


def parse_voice_args(data: str) -> dict:
    """Extra fields of a JSON voice command (e.g. circular: direction/duration/angular)."""
    try:
        obj = json.loads(data)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def parse_point(data: str) -> Optional[tuple[float, float]]:
    """/select_target payload -> (x, y). Accepts JSON {"x":..,"y":..} / {"u":..,"v":..} /
    [x, y] or text "x,y" / "x y"; pixels or 0..1 normalised."""
    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            for kx, ky in (("x", "y"), ("u", "v"), ("px", "py")):
                if kx in obj and ky in obj:
                    return float(obj[kx]), float(obj[ky])
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return float(obj[0]), float(obj[1])
    except (ValueError, TypeError):
        pass
    parts = data.replace(",", " ").split()
    try:
        return (float(parts[0]), float(parts[1])) if len(parts) >= 2 else None
    except ValueError:
        return None


def lidar_front_min(ranges, angle_min: float, angle_inc: float, range_min: float, range_max: float,
                    laser_yaw: float = 0.0, laser_xy=(0.0, 0.0), sector: float = math.radians(45)) -> Optional[float]:
    """Closest point in front of base_link (|bearing| <= sector, ahead of the base origin),
    measured from the base origin. Laser pose in base_link: yaw + (x, y) offset."""
    r = np.asarray(ranges, dtype=np.float64)
    if r.size == 0:
        return None
    a = angle_min + angle_inc * np.arange(r.size) + laser_yaw
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= range_max)
    if not ok.any():
        return None
    x = r[ok] * np.cos(a[ok]) + laser_xy[0]
    y = r[ok] * np.sin(a[ok]) + laser_xy[1]
    front = (x > 0.0) & (np.abs(np.arctan2(y, x)) <= sector)
    if not front.any():
        return None
    return float(np.min(np.hypot(x[front], y[front])))


def scan_points_base(ranges, angle_min, angle_inc, range_min, range_max, laser_yaw=0.0,
                     laser_xy=(0.0, 0.0), max_points: int = 360) -> np.ndarray:
    """Valid scan points as (N, 2) in base_link (for the potential field), downsampled."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(r.size) + laser_yaw
    ok = np.isfinite(r) & (r >= max(range_min, 0.05)) & (r <= range_max)
    pts = np.stack([r[ok] * np.cos(a[ok]) + laser_xy[0], r[ok] * np.sin(a[ok]) + laser_xy[1]], axis=1)
    if len(pts) > max_points:
        pts = pts[:: int(np.ceil(len(pts) / max_points))]
    return pts


def overlay_data(result, state: str, owner: str, fps: float, message: str = "") -> dict:
    """Small, drawing-ready summary of the latest result (boxes, labels, status line)."""
    boxes = []
    for i, p in enumerate(result.persons):
        d = result.distances[i] if i < len(result.distances) else float("inf")
        boxes.append((p.bbox.astype(int).tolist(), i == result.target_index,
                      ("OWNER" if i == result.target_index else "person") + (f" {d:.2f}" if np.isfinite(d) else "")))
    return {"boxes": boxes, "status": f"{state} | {owner} | {fps:.0f} Hz", "side": owner_side_text(result.theta),
            "message": f"[{owner}] {message}" if message else ""}


def draw_light(img: np.ndarray, ov: Optional[dict]) -> None:
    """Cheap overlay (boxes + text) for the camera-rate web video."""
    if not ov:
        return
    for (x1, y1, x2, y2), is_owner, label in ov["boxes"]:
        col = (0, 220, 0) if is_owner else (60, 60, 230)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if is_owner else 1)
        cv2.putText(img, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for k, txt in enumerate((ov["status"], ov["side"])):
        cv2.putText(img, txt, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    msg = ov.get("message", "")
    if msg:
        y = img.shape[0] - 12
        cv2.putText(img, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, msg, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)


def owner_side_text(theta: Optional[float]) -> str:
    """Where the owner is, FROM THE ROBOT'S POINT OF VIEW - removes the classic confusion
    ('I stepped left' = the robot's RIGHT when you face the robot)."""
    if theta is None:
        return "owner: not seen"
    deg = math.degrees(theta)
    if abs(deg) < 5:
        return f"owner STRAIGHT AHEAD ({deg:+.0f} deg)"
    side = "LEFT" if deg > 0 else "RIGHT"
    other = "RIGHT" if deg > 0 else "LEFT"
    return f"owner on ROBOT'S {side} ({deg:+.0f} deg) = your {other} if you face the robot"


def front_distance(depth_m: Optional[np.ndarray]) -> Optional[float]:
    """Near distance of whatever is straight ahead: 5th percentile of valid depth in the
    central window (middle 40 % of the width, 20-70 % of the height - the floor stays out)."""
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    roi = depth_m[int(0.2 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    valid = roi[(roi > 0.2) & (roi < 6.0)]
    if valid.size < 200:
        return None
    return float(np.percentile(valid, 5))


def bridge_state(state: str, owner: Optional[str], armed: bool, circular: bool) -> str:
    """Robot state in voice_bridge.py's vocabulary (its _guidance() table)."""
    if owner == "NO_OWNER":
        return "SEARCHING" if armed else "IDLE"
    if owner == "CLAIMING":
        return "ENROLLING"
    if circular:
        return "FOLLOWING"
    return {"FOLLOW": "FOLLOWING", "SEARCH": "FOLLOWING", "WAIT": "OWNED", "STEADY": "STOPPED"}.get(state, "OWNED")


def eye_expression(state: str, owner: Optional[str], armed: bool = False) -> str:
    """Robot state -> eye_display.py state (idle|listening|thinking|following|stopped|circular|unknown)."""
    if armed:
        return "listening"             # voice asked for a new owner: waiting for a raised hand
    if owner == "NO_OWNER":
        return "idle"
    if owner == "CLAIMING":
        return "thinking"              # learning the new owner
    return {"FOLLOW": "following", "WAIT": "stopped", "SEARCH": "circular",
            "STEADY": "unknown"}.get(state, "idle")


class NS_Result:
    """Stand-in for a Trigger response when a service callback is called from a key."""

    success = False
    message = ""


class FollowMeNode(Node):
    def __init__(self) -> None:
        # Claimed BEFORE creating any ROS/GPU resources: two followme_node processes
        # publishing under the same node name have raced /cmd_vel silently in the past
        # (whichever message lands last wins), making every param get/set and every
        # physical steering test unreliable without any error being shown anywhere.
        lock = check_singleton()
        if not lock.acquired:
            msg = (f"another followme_node is already running (pid {lock.other_pid}) - refusing to "
                  f"start a second instance (they would race on /cmd_vel). Stop it first, e.g.:\n"
                  f"  kill {lock.other_pid}   # or: ~/stop_a2rbot.sh")
            print(f"[followme] FATAL: {msg}", flush=True)
            raise SystemExit(msg)
        self._singleton_lock_held = True
        super().__init__("followme")
        p = self.declare_parameter
        p("config_path", "")
        p("target_model_path", "models/target.npz")
        p("color_topic", "/camera/camera/color/image_raw")
        p("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        p("camera_info_topic", "/camera/camera/color/camera_info")
        p("global_frame", "map")
        p("base_frame", "base_link")
        p("nav2_action", "navigate_to_pose")
        p("cmd_vel_topic", "/cmd_vel")
        p("enable_navigation", False)              # False = dry run: goals are only published/logged
        p("control_mode", "cmd_vel")               # cmd_vel (direct Twist) | nav2 (NavigateToPose goals)
        p("angular_sign", 1.0)                     # -1.0 flips steering (base turns the other way)
        p("mirror_x", False)                       # flip left/right at the vision source (see below)
        p("enable_voice", True)
        p("select_target_topic", "/select_target")     # std_msgs/String click on the video -> new owner
        p("face_event_passthrough", False)             # robot_face_ros.py reads /face_event itself
        p("circular_linear_speed", 0.1)                # m/s forward while arcing (Ackermann)
        p("hard_stop_m", 0.9)                          # anything closer in front -> no forward motion
        p("owner_position_topic", "/followme/owner_position")           # PointStamped, camera frame
        p("owner_position_filtered_topic", "/followme/owner_position_filtered")
        p("persons_topic", "/followme/persons")                          # PoseArray, all people
        p("fused_position_topic", "/owner_fused_position")               # from the lidar fusion
        p("fused_timeout_s", 0.5)
        p("fresh_detection_s", 0.5)                    # drive forward only on a recent real detection
        p("enable_lidar_stop", True)                   # 2nd independent stop: lidar front sector
        p("scan_topic", "/scan")
        p("lidar_stop_m", 0.5)                         # anything in the front sector closer -> no forward
        p("lidar_ignore_within_m", 0.45)               # points this close to the LASER are the robot itself
        p("obstacle_scan_topic", "/followme/obstacle_scan")   # camera obstacles as a LaserScan (for the Pi)
        p("obstacle_z_min", 0.04)                      # m above the floor: lower = floor, ignored
        p("obstacle_z_max", 0.80)                      # m: higher than the robot = cannot hit it
        p("obstacle_range_max", 3.0)                   # m
        p("obstacle_stride", 4)                        # pixel step when sampling chair masks
        p("cam_height_fallback_m", 0.29)               # camera height if TF base_link->camera is missing
        p("lidar_sector_deg", 45.0)                    # +/- this many degrees around straight ahead
        p("laser_yaw_deg", 0.0)                        # laser mounting yaw, used only if TF is missing
        p("auto_save_owner", True)                     # save the claimed owner, reload after a restart
        p("web_camera_rate", False)                    # True: web video at camera rate (boxes lag behind)
        p("minimap_enabled", True)                     # web view: top-down map + planned path on the floor
        p("minimap_scan_topic", "/scan/merged")        # lidar (Pi), chairs closed
        p("minimap_path_topic", "/local_planner/path") # planner's chosen path (Pi)
        p("minimap_owner_topic", "/follow/owner")      # owner as tracked by the Pi (base_link)
        p("minimap_goal_topic", "/follow/goal")        # stand-off goal (Pi)
        p("minimap_status_topic", "/local_planner/status")
        p("minimap_size", 220)                         # px
        p("minimap_ground_z", -0.09)                   # floor height in base_link (URDF: base 9 cm up)
        p("search_mode", "internal")                   # internal: own search | external: hand over
        p("voice_command_topic", "/voice_command")     # std_msgs/String, JSON {"action": ...} or text
        p("voice_state_cmd_topic", "/robot_state_cmd") # std_msgs/String "FOLLOW" / "STOP"
        p("owner_distance_topic", "/owner_distance")   # std_msgs/Float32 (-1 = owner not seen)
        p("robot_state_topic", "/robot_state")         # std_msgs/String
        p("enable_eyes", True)
        p("eye_state_topic", "/eye_state")             # std_msgs/String read by eye_display.py
        p("owner_bearing_topic", "/owner_bearing")     # std_msgs/Float32 rad, + = owner on the left
        p("publish_debug_image", True)
        p("show_window", False)                    # local OpenCV window like followme_app.py
        p("web_port", 8081)                        # browser view http://<ip>:8081 (0 = off)
        p("sync_slop_s", 0.05)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        cfg_path = g("config_path")
        self.cfg = FollowMeConfig.from_yaml(cfg_path) if cfg_path else FollowMeConfig()
        self.target_model_path = g("target_model_path")
        self.global_frame, self.base_frame = g("global_frame"), g("base_frame")
        self.debug_image = bool(g("publish_debug_image"))
        self.show_window = bool(g("show_window"))
        self.web = None
        if int(g("web_port")) > 0:
            try:
                self.web = MjpegServer(int(g("web_port")))
                self.get_logger().info(f"web view on http://<this-ip>:{self.web.port}/")
            except OSError as exc:
                self.get_logger().error(f"web view disabled: {exc}")
        self.device = resolve_device(self.cfg.device)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.control_mode = g("control_mode")
        if self.control_mode not in ("cmd_vel", "nav2"):
            raise ValueError("control_mode must be cmd_vel or nav2")
        self.goal_frame = self.base_frame if self.control_mode == "cmd_vel" else self.global_frame
        if g("enable_navigation") and self.control_mode == "nav2":
            self.backend: NavigationBackend = Nav2Backend(self, self.global_frame, g("nav2_action"),
                                                          g("cmd_vel_topic"))
        elif g("enable_navigation"):
            nav = self.cfg.navigation
            radius = nav.safety_radius_m if nav.safety_radius_m is not None else safety_radius(
                nav.max_velocity_mps, self.cfg.localization.expiration_time_s, nav.safety_factor)
            self.backend = CmdVelBackend(self, g("cmd_vel_topic"), DirectFollowController(nav, radius),
                                         float(g("angular_sign")))
            self.get_logger().info(f"control_mode cmd_vel on {g('cmd_vel_topic')}")
        else:
            self.backend = RecordingNavigation()
            self.get_logger().warn("enable_navigation:=false -> dry run, the robot will not move")
        self.controller = FollowMeController(self.cfg, self.backend)
        self.perception = None                       # built once CameraInfo arrives
        self.ownership = None

        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.target_pub = self.create_publisher(PoseStamped, "~/target_pose", 10)
        self.owner_pos_pub = self.create_publisher(PointStamped, g("owner_position_topic"), 10)
        self.owner_cam_pub = self.create_publisher(PointStamped, "/followme/owner_camera", 10)  # a2r_lidar_keeper: camera detections only
        self.owner_filt_pub = self.create_publisher(PointStamped, g("owner_position_filtered_topic"), 10)
        self.persons_pub = self.create_publisher(PoseArray, g("persons_topic"), 10)
        self._fused = None                       # (PointStamped, receive time)
        self._fused_timeout = float(g("fused_timeout_s"))
        self.lidar_assist = False
        self.create_subscription(PointStamped, g("fused_position_topic"), self._on_fused, 10)
        # a2r_idguard: phone / lidar keeper position as a second opinion on the owner's identity,
        # and the robot's own motion (odom TF) removed from the camera-frame owner filter
        self.declare_parameter("a2r_odom_frame", "odom")
        self.declare_parameter("a2r_phone_hint", True)
        self.declare_parameter("a2r_ego_motion", True)
        self._a2r_phone = self._a2r_phone_st = self._a2r_T_prev = None
        self._a2r_log_t = 0.0
        self.create_subscription(PointStamped, "/phone_owner/view",
                                 lambda m: setattr(self, "_a2r_phone", (m, time.monotonic())), 5)
        self.create_subscription(String, "/phone_owner/status", self._a2r_on_phone_st, 5)
        self.marker_pub = self.create_publisher(Marker, "~/safety_circle", 1)
        self.debug_pub = self.create_publisher(Image, "~/debug_image", 1)
        self.owner_dist_pub = self.create_publisher(Float32, g("owner_distance_topic"), 10)
        self.robot_state_pub = self.create_publisher(String, g("robot_state_topic"), 10)
        self.owner_bearing_pub = self.create_publisher(Float32, g("owner_bearing_topic"), 10)
        self.eye_pub = None
        self._last_eye: Optional[str] = None
        self._eye_reassert = False
        if g("enable_eyes"):
            self.eye_pub = self.create_publisher(String, g("eye_state_topic"), 10)
            self.create_subscription(String, g("eye_state_topic"), self._on_eye_state, 10)
        if g("enable_voice"):
            self.create_subscription(String, g("voice_command_topic"), self._on_voice, 10)
            self.create_subscription(String, g("voice_state_cmd_topic"), self._on_voice, 10)
            self.create_subscription(String, g("select_target_topic"), self._on_select, 10)
        self._circular_until = 0.0
        self._circular_cmd = (0.0, 0.0)
        self._hard_stop_m = float(g("hard_stop_m"))
        self._lidar_front = (None, 0.0)          # (distance m, receive time)
        self._lidar_stop_m = float(g("lidar_stop_m"))
        self._lidar_ignore = float(g("lidar_ignore_within_m"))
        self._obst_z = (float(g("obstacle_z_min")), float(g("obstacle_z_max")))
        self._obst_rmax = float(g("obstacle_range_max"))
        self._obst_stride = max(1, int(g("obstacle_stride")))
        self._cam_h_fallback = float(g("cam_height_fallback_m"))
        self.obstacle_scan_pub = self.create_publisher(LaserScan, g("obstacle_scan_topic"), 5)
        self._lidar_pts = None
        self._lidar_pts_t = -1e9
        self._cam_pts = None
        self._cam_pts_t = -1e9
        self._lidar_sector = math.radians(float(g("lidar_sector_deg")))
        self._laser_yaw = math.radians(float(g("laser_yaw_deg")))
        self._laser_T = None                     # cached base_link <- laser transform
        self._auto_save = bool(g("auto_save_owner"))
        self.web_camera_rate = bool(g("web_camera_rate"))
        self.minimap_on = bool(g("minimap_enabled"))
        self._minimap_size = int(g("minimap_size"))
        self._ground_z = float(g("minimap_ground_z"))
        self._nav = {}
        if self.minimap_on:
            from nav_msgs.msg import Path as NavPath
            self.create_subscription(LaserScan, g("minimap_scan_topic"),
                                     lambda m: self._nav_set("scan", m), qos_profile_sensor_data)
            self.create_subscription(NavPath, g("minimap_path_topic"), lambda m: self._nav_set("path", m), 1)
            self.create_subscription(PointStamped, g("minimap_owner_topic"), lambda m: self._nav_set("owner", m), 5)
            self.create_subscription(PoseStamped, g("minimap_goal_topic"), lambda m: self._nav_set("goal", m), 5)
            self.create_subscription(String, g("minimap_status_topic"), lambda m: self._nav_set("status", m), 5)
            self.create_subscription(PointStamped, "/phone_owner/view", lambda m: self._nav_set("phone", m), 5)  # a2r_phone_view
            self.create_subscription(String, "/phone_owner/status", lambda m: self._nav_set("phone_st", m), 5)  # a2r_phone_view
            self.create_subscription(PointStamped, "/gps/owner_estimate", lambda m: self._nav_set("gps", m), 5)  # a2r_step3d
            self.create_subscription(String, "/gps/status", lambda m: self._nav_set("gps_st", m), 5)  # a2r_step3d
        self.search_external = str(g("search_mode")).lower() == "external"
        if self.search_external:
            # FollowMe itself must not move while searching: the external planner drives
            self.cfg.navigation.search_angular_speed = 0.0
            self.cfg.navigation.search_linear_speed = 0.0
        self.owner_lost_pub = self.create_publisher(Bool, "/followme/owner_lost", 10)
        self.last_seen_pub = self.create_publisher(PointStamped, "/followme/owner_last_seen", 10)
        self._owner_lost = False
        self._lost_hb = 0.0
        self._last_seen_msg = None
        self._last_T = None
        if g("enable_lidar_stop"):
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data)
        self._fresh_s = float(g("fresh_detection_s"))
        self._last_detection = 0.0
        self._near_logged = 0.0
        self._circular_lin = float(g("circular_linear_speed"))

        self.create_subscription(CameraInfo, g("camera_info_topic"), self._on_info, 1)
        color_sub = message_filters.Subscriber(self, Image, g("color_topic"), qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, g("depth_topic"), qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5,
                                                                slop=float(g("sync_slop_s")))
        self.sync.registerCallback(self._on_frames)

        for name, cb in (("calibrate", self._srv_calibrate), ("save_target", self._srv_save),
                         ("follow", self._srv_follow), ("wait", self._srv_wait), ("reset", self._srv_reset)):
            self.create_service(Trigger, f"~/{name}", cb)

        self._latest: Optional[tuple[Image, Image]] = None
        self._frame_event = threading.Event()
        self._lock = threading.Lock()
        self._running = True
        self._fps = 0.0
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self.create_timer(1.0, self._publish_marker)
        self.create_timer(0.05, self._watchdog)
        self.cfg.localization.mirror_x = bool(g("mirror_x"))
        self.add_on_set_parameters_callback(self._on_param_change)
        self.get_logger().info(f"FollowMe ready (device {self.device}, safety circle "
                               f"{self.controller.safety_radius:.2f} m). Call ~/calibrate with the target alone.")

    def _on_param_change(self, params):
        """Make angular_sign and mirror_x ACTUALLY live-settable (previously angular_sign was only
        read once at startup: 'ros2 param set' would report success but change nothing running)."""
        for p in params:
            if p.name == "mirror_x":
                self.cfg.localization.mirror_x = bool(p.value)
                self.get_logger().info(f"mirror_x -> {p.value} (live)")
            elif p.name == "angular_sign":
                if hasattr(self.backend, "angular_sign"):
                    self.backend.angular_sign = float(p.value)
                    self.get_logger().info(f"angular_sign -> {p.value} (live)")
                else:
                    self.get_logger().warn("angular_sign has no effect: control_mode is not cmd_vel")
        return SetParametersResult(successful=True)

    # -- inputs -----------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self.perception is not None:
            return
        intr = CameraIntrinsics.from_camera_info(msg.k, msg.width, msg.height)
        try:
            self.perception = build_perception(self.cfg, self.device, intr, self.target_model_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"model loading failed: {exc}")
            raise
        if self.cfg.ownership.enabled:
            if self.perception.gestures is None:
                self.get_logger().error("ownership.enabled needs the gesture classifier; ownership OFF")
            else:
                self.ownership = OwnershipManager(self.cfg, self.perception)
        self.get_logger().info(f"perception built (fx={intr.fx:.1f}, {intr.width}x{intr.height}), "
                               f"mode {self.perception.mode.value}")

    def _on_frames(self, color: Image, depth: Image) -> None:
        self._latest = (color, depth)               # atomic swap; older pairs are dropped
        ev = getattr(self, "_frame_event", None)   # a2r_cuelog: may run before __init__ made it
        if ev is not None:
            ev.set()
        web = getattr(self, "web", None)
        if web is not None and web.watched and getattr(self, "web_camera_rate", False):
            now = time.monotonic()
            if now - getattr(self, "_web_t", 0.0) >= 0.035:        # ~28 fps max
                self._web_t = now
                try:
                    img = image_to_bgr(color)
                except ValueError:
                    return
                draw_light(img, getattr(self, "_overlay", None))
                web.push(img)

    def _lookup(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        fails = self.__dict__.setdefault("_tf_fail_until", {})
        if time.monotonic() < fails.get((target, source), 0.0):
            return None                             # known missing: retry every 2 s only
        T = self._lookup_now(target, source, stamp)
        if T is None:
            fails[(target, source)] = time.monotonic() + 2.0
        return T

    def _lookup_now(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        for when in (stamp, rclpy.time.Time()):     # exact stamp, then latest available
            try:
                tf = self.tf_buffer.lookup_transform(target, source, when,
                                                     timeout=rclpy.duration.Duration(seconds=0.02))
                return transform_to_matrix(tf.transform)
            except TransformException:
                continue
        return None

    # -- worker -----------------------------------------------------------------
    def _loop(self) -> None:
        t_prev = time.perf_counter()
        warned_tf = False
        while self._running and rclpy.ok():
            if not self._frame_event.wait(timeout=0.5):
                continue
            self._frame_event.clear()
            pair, self._latest = self._latest, None
            if pair is None or self.perception is None:
                continue
            color_msg, depth_msg = pair
            t_start = time.perf_counter()
            try:
                frame, depth = image_to_bgr(color_msg), depth_to_meters(depth_msg)
            except ValueError as exc:
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)
                continue
            stamp = rclpy.time.Time.from_msg(color_msg.header.stamp)
            if self.control_mode == "cmd_vel":
                # owner position in the robot base frame; no map needed
                T_map_cam = self._lookup(self.base_frame, color_msg.header.frame_id, stamp)
                if T_map_cam is None:
                    if not warned_tf:
                        self.get_logger().warn(f"no TF {self.base_frame} <- {color_msg.header.frame_id}: "
                                               "assuming a forward-facing camera at the base origin")
                        warned_tf = True
                    T_map_cam = OPTICAL_TO_BASE
                self._last_T = T_map_cam
                T_map_base = None
            else:
                T_map_cam = self._lookup(self.global_frame, color_msg.header.frame_id, stamp)
                T_map_base = self._lookup(self.global_frame, self.base_frame, stamp)
            if T_map_cam is None and not warned_tf:
                self.get_logger().warn(f"no TF {self.global_frame} <- {color_msg.header.frame_id}; "
                                       "not following until it is available")
                warned_tf = True
            with self._lock:
                now = time.monotonic()
                t_conv = time.perf_counter()
                self._a2r_prepare(color_msg.header.frame_id, now)        # a2r_idguard
                result = self.perception.process(frame, depth, now)
                t_perc = time.perf_counter()
                if T_map_cam is None:
                    result.p_filt = None             # without Eq. 2 there is no safe goal: treat as !alpha
                self.lidar_assist = False
                if result.p_filt is None:            # camera lost the owner: use the lidar track
                    p_l = self._fused_in_camera(color_msg.header.frame_id, now)
                    if p_l is not None:
                        result.p_filt, result.theta = p_l, heading_angle(p_l)
                        self.lidar_assist = True
                        self._last_detection = now
                        if self.ownership is not None:
                            self.ownership._owner_seen = now
                if result.p_cam is not None:
                    self._last_detection = now
                cam_near = self._camera_obstacles(result, depth, color_msg.header, now)
                near = front_distance(depth)
                too_close = near is not None and near < self._hard_stop_m
                if cam_near is not None and cam_near < self._hard_stop_m:
                    too_close = True
                    near = cam_near if near is None else min(near, cam_near)
                l_near, l_t = self._lidar_front
                if l_near is not None and now - l_t < 0.5 and l_near < self._lidar_stop_m:
                    too_close, near = True, min(near, l_near) if near is not None else l_near
                if too_close and now - self._near_logged > 2.0:
                    src = "lidar" if (l_near is not None and now - l_t < 0.5 and l_near < self._lidar_stop_m) \
                        else "camera depth"
                    self.get_logger().info(f"hard stop ({src}): something {near:.2f} m in front")
                    self._near_logged = now
                if hasattr(self.backend, "allow_forward"):
                    self.backend.obstacle_clear = not too_close          # applies to EVERY motion
                    self.backend.allow_forward = (not too_close) and (now - self._last_detection <= self._fresh_s)
                action = self.controller.step(result, T_map_cam,
                                              yaw_from_matrix(T_map_base) if T_map_base is not None else None)
                if self.ownership is not None:
                    for ev in self.ownership.update(frame, depth, result, now):
                        self.get_logger().info(f"ownership: {ev}")
                        self._persist_owner(ev)
                        if ev == "follow":
                            action = self.controller.command(FOLLOW, now)
                        elif ev in ("released", "claim_started", "claim_aborted"):
                            self.controller.machine.reset()
                            self.controller.stop(now)
                state = self.controller.state
                # a2r_step3: log every FOLLOW/SEARCH/WAIT/STEADY change and WHY (a WAIT from a hand
                # gesture stops the car until a FOLLOW gesture - the Pi only shows 'HOLD ... WAIT')
                _tr = getattr(self.controller.machine, "transitions", None)
                if _tr is not None:
                    _n0 = getattr(self, "_a2r_ntr", 0)
                    _n0 = 0 if _n0 > len(_tr) else _n0
                    for _a, _b, _why in list(_tr)[_n0:]:
                        self.get_logger().info(f"controller: {getattr(_a, 'value', _a)} -> "
                                               f"{getattr(_b, 'value', _b)} ({_why})")
                    self._a2r_ntr = len(_tr)
                t_own = time.perf_counter()
            t = time.perf_counter()
            self._fps = 0.9 * self._fps + 0.1 / max(1e-6, t - t_prev)
            t_prev = t
            owner = self.ownership.state.value if self.ownership is not None else "-"
            self.state_pub.publish(String(data=f"{state.value}|{result.mode.value}|{owner}"))
            now_m = time.monotonic()
            own = self.ownership
            circ = self._circular_step(now_m)
            self.robot_state_pub.publish(String(data=bridge_state(
                state.value, own.state.value if own is not None else None,
                own is not None and own.armed(now_m), circ)))
            if result.p_filt is not None and result.theta is not None:
                self.owner_bearing_pub.publish(Float32(data=float(result.theta)))
            self._publish_positions(result, color_msg.header, depth)
            self._external_search(result, state, color_msg.header, now_m)
            self._update_eyes(state.value, self.ownership.state.value if self.ownership is not None else None,
                              self.ownership is not None and self.ownership.armed(time.monotonic()))
            self.owner_dist_pub.publish(Float32(data=float(self.controller.last_distance)
                                                if result.p_filt is not None else -1.0))
            goal = self.controller.last_goal
            if result.p_filt is not None and goal is not None:
                ps = PoseStamped()
                ps.header.frame_id, ps.header.stamp = self.goal_frame, color_msg.header.stamp
                ps.pose.position.x, ps.pose.position.y = goal.x, goal.y
                o = ps.pose.orientation
                o.x, o.y, o.z, o.w = yaw_to_quaternion(goal.yaw)
                self.target_pub.publish(ps)
            self._overlay = overlay_data(result, state.value,
                                         self.ownership.state.value if self.ownership is not None else "-",
                                         self._fps,
                                         self.ownership.message if self.ownership is not None else "")
            publish = self.debug_image and self.debug_pub.get_subscription_count() > 0
            web = self.web is not None and self.web.watched and not getattr(self, "web_camera_rate", False)
            if self.web is not None:
                owner_txt = self.ownership.state.value if self.ownership is not None else "-"
                self.web.state_text = (f"{state.value} | {owner_txt} | {self._fps:.0f} Hz | "
                                       f"{owner_side_text(result.theta)}")
            if publish or self.show_window or web:
                disp = frame.copy()
                text = ""
                own = self.ownership
                if own is not None:
                    text = f"[{own.state.value}] {own.message}"
                    if own.claimant_box is not None and own.state.value == "CLAIMING":
                        x1, y1, x2, y2 = own.claimant_box.astype(int)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 3)
                draw(disp, result, state, action, self.controller.safety_radius, self._fps, text)
                self._draw_minimap(disp)
                draw_obstacles(disp, result)
                if publish:
                    img = Image()
                    img.header = color_msg.header
                    img.height, img.width = disp.shape[:2]
                    img.encoding, img.step = "bgr8", disp.shape[1] * 3
                    img.data = disp.tobytes()
                    self.debug_pub.publish(img)
                if self.show_window:
                    self._show(disp)
                if web and not getattr(self, "web_camera_rate", False):
                    self.web.push(disp)
            self._timing_log(result.timings_ms, 1e3 * (t_conv - t_start), 1e3 * (t_own - t_perc),
                             1e3 * (time.perf_counter() - t_start))

    def _timing_log(self, stages: dict, conv_ms: float, own_ms: float, total_ms: float) -> None:
        """Average per-stage latency, logged every 5 s."""
        acc = getattr(self, "_tacc", None)
        if acc is None:
            acc = self._tacc = {"n": 0, "t0": time.monotonic()}
        for k, v in list(stages.items()) + [("ros_convert", conv_ms), ("ownership", own_ms),
                                            ("loop_total", total_ms)]:
            acc[k] = acc.get(k, 0.0) + v
        acc["n"] += 1
        if time.monotonic() - acc["t0"] >= 5.0:
            n = acc.pop("n")
            acc.pop("t0")
            parts = "  ".join(f"{k}={v / n:.0f}" for k, v in acc.items())
            self.get_logger().info(f"timing ms (avg of {n}): {parts}")
            self._tacc = None

    def _show(self, disp: np.ndarray) -> None:
        """OpenCV window with the same keys as followme_app.py:
        C calibrate, S save, F follow, W wait, R release/reset, Q close the window."""
        cv2.imshow("FollowMe", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        res = NS_Result()
        actions = {ord("c"): self._srv_calibrate, ord("s"): self._srv_save, ord("f"): self._srv_follow,
                   ord("w"): self._srv_wait, ord("r"): self._srv_reset}
        if key in actions:
            actions[key](None, res)
            self.get_logger().info(f"key {chr(key).upper()}: {res.message}")
        elif key in (ord("q"), 27):
            self.show_window = False
            cv2.destroyAllWindows()
            self.get_logger().info("window closed (node keeps running)")

    def _update_eyes(self, state: str, owner: Optional[str], armed: bool = False) -> None:
        """Publish the eye expression when it changes (the voice system keeps priority
        while it shows listening/thinking; its final 'idle' is overridden again)."""
        if self.eye_pub is None:
            return
        eye = eye_expression(state, owner, armed)
        if eye != self._last_eye or self._eye_reassert:
            self.eye_pub.publish(String(data=eye))
            self._last_eye, self._eye_reassert = eye, False

    def _on_eye_state(self, msg: String) -> None:
        # someone else (voice) set the eyes back to idle while we are active -> restore ours
        if msg.data == "idle" and self._last_eye not in (None, "idle"):
            self._eye_reassert = True

    # -- camera <-> lidar fusion I/O ---------------------------------------------
    @staticmethod
    def _point_msg(header, p) -> PointStamped:
        m = PointStamped()
        m.header = header
        m.point = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        return m

    def _publish_positions(self, result, header, depth) -> None:
        """Owner (raw detection + filtered) and all people as 3-D points in the camera
        optical frame, stamped with the image time - for fusion with the lidar."""
        if result.p_cam is not None:
            self.owner_pos_pub.publish(self._point_msg(header, result.p_cam))
            self.owner_cam_pub.publish(self._point_msg(header, result.p_cam))  # a2r_lidar_keeper
            self.owner_cam_pub.publish(self._point_msg(header, result.p_cam))  # a2r_lidar_keeper
        elif False and getattr(self, "lidar_assist", False) and result.p_filt is not None:  # a2r_lidar_keeper:
            self.owner_pos_pub.publish(self._point_msg(header, result.p_filt))  # lidar holds the owner -> Pi
        if result.p_filt is not None and not self.lidar_assist:
            self.owner_filt_pub.publish(self._point_msg(header, result.p_filt))
        if self.persons_pub.get_subscription_count() > 0 and self.perception is not None and depth is not None:
            arr = PoseArray()
            arr.header = header
            for p in result.persons:
                c = target_position(depth, p.mask, self.perception.intrinsics, self.cfg.localization)
                if c is not None:
                    pose = Pose()
                    pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                    pose.orientation.w = 1.0
                    arr.poses.append(pose)
            self.persons_pub.publish(arr)

    # -- a2r_idguard -------------------------------------------------------------------------
    def _a2r_on_phone_st(self, msg) -> None:
        import json
        try:
            self._a2r_phone_st = (json.loads(msg.data), time.monotonic())
        except ValueError:
            pass

    def _a2r_prepare(self, cam_frame: str, now: float) -> None:
        """Before each frame: move the owner filter with the robot's own motion since the last
        frame, and give the identity check a second opinion - the lidar keeper's owner track if it
        has one, else the phone (only while it is aligned and reasonably sure)."""
        p = self.perception
        if p is None or not hasattr(p, "a2r"):
            return
        odom = str(self.get_parameter("a2r_odom_frame").value)
        T = self._lookup(odom, cam_frame, rclpy.time.Time())                  # odom <- camera (latest)
        if bool(self.get_parameter("a2r_ego_motion").value) and T is not None and self._a2r_T_prev is not None:
            rel = np.linalg.solve(T, self._a2r_T_prev)                         # camera now <- camera before
            if float(np.linalg.norm(rel[:3, 3])) < 0.5:                        # an odom jump is not motion
                p.a2r_ego(rel)
        self._a2r_T_prev = T
        hint = None
        p_l = self._fused_in_camera(cam_frame, now)                            # keeper (lidar) owner track
        if p_l is not None:
            hint = (p_l, 0.8, now, "lidar")
        elif bool(self.get_parameter("a2r_phone_hint").value) and T is not None:
            ph, st = self._a2r_phone, self._a2r_phone_st
            if ph is not None and st is not None and now - ph[1] <= 0.6 and now - st[1] <= 2.0 \
                    and st[0].get("aligned") and 0.0 <= float(ph[0].point.z) <= 1.5:
                m = ph[0]
                src = m.header.frame_id or odom
                Tc = np.linalg.inv(T) if src == odom else self._lookup(cam_frame, src, rclpy.time.Time())
                if Tc is not None:
                    q = Tc @ np.array([m.point.x, m.point.y, 0.0, 1.0])
                    hint = (q[:3], min(3.0, max(1.2, 2.5 * float(m.point.z))), now, "phone")
        p.a2r.hint = hint
        if now - self._a2r_log_t > 20.0:
            self._a2r_log_t = now
            n = {k: v for k, v in p.a2r.n.items() if v}
            self.get_logger().info(f"identity guard: {n or 'nothing yet'} | second opinion: "
                                   f"{hint[3] if hint else 'none'} | ego-motion: {'odom' if T is not None else 'no TF'}")

    def _on_fused(self, msg: PointStamped) -> None:
        self._fused = (msg, time.monotonic())    # receive time: no clock sync needed between PCs

    def _fused_in_camera(self, camera_frame: str, now: float) -> Optional[np.ndarray]:
        """Latest lidar-fused owner position, transformed into the camera optical frame."""
        if self._fused is None or now - self._fused[1] > self._fused_timeout:
            return None
        msg = self._fused[0]
        src = msg.header.frame_id or camera_frame
        if src == camera_frame:
            T = np.eye(4)
        else:
            T = self._lookup(camera_frame, src, rclpy.time.Time())
            if T is None:
                return None
        return (T @ np.array([msg.point.x, msg.point.y, msg.point.z, 1.0]))[:3]

    def _watchdog(self) -> None:
        wd = getattr(self.backend, "watchdog", None)
        if wd is not None:
            wd(self.cfg.navigation.cmd_timeout_s)

    def _external_search(self, result, state, header, now: float) -> None:
        """Hand the wheels to an external search planner while the owner is lost."""
        if result.p_cam is not None and self._last_T is not None:
            pb = (self._last_T @ np.append(np.asarray(result.p_cam, float), 1.0))[:3]
            m = PointStamped()
            m.header.stamp = header.stamp
            m.header.frame_id = self.base_frame
            m.point = Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2]))
            self._last_seen_msg = m
        owned = self.ownership is None or self.ownership.state.value == "OWNED"
        lost = owned and state.value in ("SEARCH", "STEADY") and self.perception is not None \
            and self.perception.model is not None
        if not self.search_external:
            lost_flag = False
        else:
            lost_flag = lost
        if lost_flag != self._owner_lost:
            self._owner_lost = lost_flag
            if hasattr(self.backend, "muted"):
                if lost_flag:
                    self.backend.set_velocity(0.0, 0.0)     # one clean stop, then silence
                    self.backend.muted = True
                    self.get_logger().info("owner lost: /cmd_vel handed to the external search planner")
                else:
                    self.backend.muted = False
                    self.get_logger().info("owner found again: FollowMe drives" if (self.ownership is None or self.ownership.state.value == "OWNED") else "ownership released by FollowMe - the Pi search ends")
            self._lost_hb = 0.0
        if now - self._lost_hb >= 0.5:                      # 2 Hz heartbeat (+ immediate on change)
            self._lost_hb = now
            self.owner_lost_pub.publish(Bool(data=self._owner_lost))
            if self._owner_lost and self._last_seen_msg is not None:
                self.last_seen_pub.publish(self._last_seen_msg)

    def _camera_obstacles(self, result, depth, header, now: float) -> Optional[float]:
        """Chairs/tables seen by the camera -> floor points in base_link: hard stop (front
        sector), potential-field avoidance, and /followme/obstacle_scan for the Pi."""
        obst = getattr(result, "obstacles", None) or []
        if depth is None or self.perception is None:
            return None
        T = self._last_T if self._last_T is not None else OPTICAL_TO_BASE
        # without TF base_link->camera the camera sits at height 0: add the measured height
        z_off = self._cam_h_fallback if np.allclose(T, OPTICAL_TO_BASE) else 0.0
        pts = obstacle_points(depth, [o.mask for o in obst], self.perception.intrinsics, T,
                              self._obst_stride, self._obst_z[0], self._obst_z[1], self._obst_rmax, z_off)
        self._cam_pts, self._cam_pts_t = pts, now
        self._merge_obstacles(now)
        if self.obstacle_scan_pub.get_subscription_count() > 0:
            self.obstacle_scan_pub.publish(points_to_scan(pts, header.stamp, self.base_frame, self._obst_rmax))
        return front_min_points(pts, self._lidar_sector)

    def _merge_obstacles(self, now: float) -> None:
        """Lidar + camera obstacle points for the potential field (each used while fresh)."""
        ctrl = getattr(self.backend, "ctrl", None)
        if ctrl is None:
            return
        parts = [p for p, t in ((self._lidar_pts, self._lidar_pts_t), (self._cam_pts, self._cam_pts_t))
                 if p is not None and len(p) and now - t < 0.5]
        ctrl.obstacles = np.vstack(parts) if parts else np.zeros((0, 2))
        ctrl.obstacles_t = now

    # -- navigation view (minimap + planned path on the floor) --------------------------------
    def _nav_set(self, key: str, msg) -> None:
        self._nav[key] = (msg, time.monotonic())

    def _nav_get(self, key: str, max_age: float = 1.0):
        v = self._nav.get(key)
        return v[0] if v is not None and time.monotonic() - v[1] <= max_age else None

    def _to_base(self, frame: str) -> Optional[np.ndarray]:
        if not frame or frame == self.base_frame:
            return np.eye(4)
        return self._lookup(self.base_frame, frame, rclpy.time.Time())

    def _nav_point(self, key: str, getter):
        m = self._nav_get(key)
        if m is None:
            return None
        T = self._to_base(m.header.frame_id)
        if T is None:
            return None
        x, y = getter(m)
        return tuple((T @ np.array([x, y, 0.0, 1.0]))[:2])

    def _draw_minimap(self, img: np.ndarray) -> None:
        """Top-down map (lidar, path, owner, goal, planner state) + the planned path on the floor."""
        if not getattr(self, "minimap_on", False):
            return
        scan_pts = path = None
        s = self._nav_get("scan")
        if s is not None:
            T = self._to_base(s.header.frame_id)
            if T is not None:
                scan_pts = scan_to_base(s.ranges, s.angle_min, s.angle_increment, T)
        p = self._nav_get("path", 1.5)
        if p is not None and len(p.poses) >= 2:
            T = self._to_base(p.header.frame_id)
            if T is not None:
                P = np.array([[q.pose.position.x, q.pose.position.y, 0.0, 1.0] for q in p.poses]).T
                path = (T @ P)[:2].T
        owner = self._nav_point("owner", lambda m: (m.point.x, m.point.y))
        goal = self._nav_point("goal", lambda m: (m.pose.position.x, m.pose.position.y))
        st = self._nav_get("status", 2.0)
        status = st.data if st is not None else "planner: no data from the Pi"
        if path is not None and self._last_T is not None and self.perception is not None:
            i = self.perception.intrinsics
            draw_ground_path(img, project_ground_path(path, self._last_T, i.fx, i.fy, i.cx, i.cy,
                                                      self._ground_z, img.shape[1], img.shape[0]))
        panel = render_minimap(self._minimap_size, scan_pts=scan_pts, path_pts=path,
                               owner=owner, goal=goal, status=status)
        ph = self._nav_get("phone", 1.0)                                   # a2r_phone_view
        php = self._nav_point("phone", lambda m: (m.point.x, m.point.y)) if ph is not None else None
        pst = self._nav_get("phone_st", 3.0)
        draw_phone(panel, php, float(ph.point.z) if ph is not None else 0.0, pst.data if pst is not None else "")
        gp = self._nav_get("gps", 5.0)                                    # a2r_step3d
        gpp = self._nav_point("gps", lambda m: (m.point.x, m.point.y)) if gp is not None else None
        gst = self._nav_get("gps_st", 3.0)
        draw_gps(panel, gpp, float(gp.point.z) if gp is not None else 0.0, gst.data if gst is not None else "")
        composite(img, panel)

    def _persist_owner(self, ev: str) -> None:
        """'claimed' -> save the owner model (survives a restart); 'released' -> delete it."""
        if not self._auto_save:
            return
        try:
            if ev == "claimed" and self.perception is not None and self.perception.model is not None:
                self.perception.model.save(self.target_model_path)
                self.get_logger().info(f"owner saved to {self.target_model_path}")
            elif ev == "released" and os.path.isfile(self.target_model_path):
                os.remove(self.target_model_path)
                self.get_logger().info(f"owner file {self.target_model_path} removed")
        except OSError as exc:
            self.get_logger().warn(f"owner persistence failed: {exc}")

    def _on_scan(self, scan: LaserScan) -> None:
        """Nearest obstacle in the robot's front sector, from the lidar (independent of the camera)."""
        if self._laser_T is None:
            T = self._lookup(self.base_frame, scan.header.frame_id, rclpy.time.Time())
            self._laser_T = T if T is not None else False
        yaw = self._laser_yaw if self._laser_T is False else math.atan2(self._laser_T[1, 0], self._laser_T[0, 0])
        off = (0.0, 0.0) if self._laser_T is False else (self._laser_T[0, 3], self._laser_T[1, 3])
        d = lidar_front_min(scan.ranges, scan.angle_min, scan.angle_increment,
                            max(scan.range_min, self._lidar_ignore), scan.range_max, yaw, off, self._lidar_sector)
        self._lidar_front = (d, time.monotonic())
        ctrl = getattr(self.backend, "ctrl", None)
        if ctrl is not None and self.cfg.navigation.apf_enabled:
            self._lidar_pts = scan_points_base(scan.ranges, scan.angle_min, scan.angle_increment,
                                              max(scan.range_min, self._lidar_ignore),
                                              min(scan.range_max, self.cfg.navigation.apf_influence_m + 0.5),
                                              yaw, off)
            self._lidar_pts_t = time.monotonic()
            self._merge_obstacles(self._lidar_pts_t)

    def _circular_step(self, now: float) -> bool:
        """Timed arc from a voice 'circular' command; True while active."""
        if not self._circular_until:
            return False
        if now < self._circular_until:
            v, w = self._circular_cmd
            if not getattr(self.backend, "obstacle_clear", True):
                v = 0.0                          # hard stop / lidar stop also applies to the arc
            self.backend.set_velocity(v, w)
            return True
        self._circular_until = 0.0
        self.backend.set_velocity(0.0, 0.0)
        return False

    def _start_circular(self, args: dict, now: float) -> None:
        nav = self.cfg.navigation
        w = min(abs(float(args.get("angular", 0.5))), nav.max_angular_rps)
        w = w if str(args.get("direction", "left")).lower() != "right" else -w
        dur = max(0.5, min(float(args.get("duration", 3.0)), 15.0))
        self.controller.command(WAIT, now)           # stop following while arcing
        self._circular_cmd = (self._circular_lin, w)
        self._circular_until = now + dur
        self.get_logger().info(f"voice: circular {'left' if w > 0 else 'right'} {dur:.1f} s")

    def _on_select(self, msg: String) -> None:
        pt = parse_point(msg.data)
        if pt is None or self.ownership is None:
            self.get_logger().info(f"select_target ignored: '{msg.data}'")
            return
        with self._lock:
            ok = self.ownership.select(pt[0], pt[1], time.monotonic())
        self.get_logger().info(f"select_target {pt}: " + ("claiming the clicked person" if ok else
                               f"ignored, robot is {self.ownership.state.value} (release first)"))

    def _on_voice(self, msg: String) -> None:
        action = parse_voice(msg.data)
        if action is None:
            self.get_logger().debug(f"voice: ignored '{msg.data}'")
            return
        now = time.monotonic()
        with self._lock:
            own = self.ownership
            if own is not None and action in ("follow", "enroll") and own.state.value == "NO_OWNER":
                own.arm(now)                     # "hey robot, follow me" + raise a hand
                self.get_logger().info("voice: claim armed - raise an open palm")
                return
            if own is not None and action == "enroll":
                self.get_logger().info(f"voice: enroll ignored, robot is {own.state.value} "
                                       "(the owner must release first)")
                return
            if action == "circular":
                if (own is not None and own.state.value != "OWNED") or self.perception is None \
                        or self.perception.model is None:
                    self.get_logger().info("voice: circular ignored, no owner enrolled")
                    return
                self._start_circular(parse_voice_args(msg.data), now)
                return
            if self._circular_until and action in ("wait", "release", "follow"):
                self._circular_until = 0.0              # cancel the arc immediately
                self.backend.set_velocity(0.0, 0.0)
            if action == "follow":
                self.controller.command(FOLLOW, now)
            elif action == "wait":
                self.controller.command(WAIT, now)
            elif action == "release":
                if self.ownership is not None:
                    self.ownership.release(now, "voice")
                elif self.perception is not None:
                    self.perception.clear_target()
                self.controller.machine.reset()
                self.controller.stop(now)
            elif action == "enroll" and self.perception is not None:
                if self.ownership is not None:
                    self.ownership.release(now, "voice enroll")
                self.controller.machine.reset()
                self.controller.stop(now)
                self.perception.start_calibration()
        self.get_logger().info(f"voice: {action} -> {self.controller.state.value}")

    def _publish_marker(self) -> None:
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "followme", 0, Marker.CYLINDER, Marker.ADD
        d = 2 * self.controller.safety_radius
        m.scale.x, m.scale.y, m.scale.z = d, d, 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 0.25
        m.pose.orientation.w = 1.0
        self.marker_pub.publish(m)

    # -- services ---------------------------------------------------------------
    def _srv_calibrate(self, _req, res):
        if self.perception is None:
            res.success, res.message = False, "no CameraInfo yet"
            return res
        with self._lock:
            self.perception.start_calibration()
        res.success, res.message = True, "calibrating: target alone in view, walk near/far and turn"
        return res

    def _srv_save(self, _req, res):
        if self.perception is None or self.perception.model is None:
            res.success, res.message = False, "no calibrated target"
            return res
        self.perception.model.save(self.target_model_path)
        res.success, res.message = True, f"saved {self.target_model_path}"
        return res

    def _command(self, gamma: str, res):
        with self._lock:
            self.controller.command(gamma, time.monotonic())
            res.success, res.message = True, f"state {self.controller.state.value}"
        return res

    def _srv_follow(self, _req, res):
        return self._command(FOLLOW, res)

    def _srv_wait(self, _req, res):
        return self._command(WAIT, res)

    def _srv_reset(self, _req, res):
        with self._lock:
            if self.ownership is not None:
                self.ownership.release(time.monotonic(), "reset service")
            elif self.perception is not None:
                self.perception.clear_target()
            self.controller.machine.reset()
            self.controller.stop(time.monotonic())
        res.success, res.message = True, "calibration dropped, robot stopped"
        return res

    def destroy_node(self):
        self._running = False
        self._frame_event.set()                         # wake the processing thread ...
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)                    # ... and let it finish before destroying
        try:
            self.backend.cancel_goal()
            self.backend.set_velocity(0.0, 0.0)
        except Exception:  # noqa: BLE001 - shutting down
            pass
        if getattr(self, "_singleton_lock_held", False):
            release_singleton()
            self._singleton_lock_held = False
        return super().destroy_node()


def obstacle_points(depth_m, masks, intr, T, stride: int = 4, z_min: float = 0.04, z_max: float = 0.8,
                    r_max: float = 3.0, z_offset: float = 0.0) -> np.ndarray:
    """Object masks + aligned depth -> (N, 2) floor points in base_link. Points at any height
    between z_min and z_max count (a chair seat above the lidar plane is still an obstacle)."""
    out = []
    for m in masks:
        rows = np.flatnonzero(m.any(axis=1))
        if rows.size == 0:
            continue
        cols = np.flatnonzero(m.any(axis=0))
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        vs, us = np.nonzero(m[y0:y1:stride, x0:x1:stride])
        vs, us = vs * stride + y0, us * stride + x0
        z = depth_m[vs, us].astype(np.float64)
        good = np.isfinite(z) & (z > 0.15) & (z < r_max + 1.0)
        if not good.any():
            continue
        P = intr.deproject(us[good].astype(np.float64), vs[good].astype(np.float64), z[good])
        B = P @ T[:3, :3].T + T[:3, 3]
        B[:, 2] += z_offset
        keep = (B[:, 2] > z_min) & (B[:, 2] < z_max) & (np.hypot(B[:, 0], B[:, 1]) < r_max)
        if keep.any():
            out.append(B[keep, :2])
    return np.vstack(out) if out else np.zeros((0, 2))


def front_min_points(pts: np.ndarray, sector: float) -> Optional[float]:
    """Closest obstacle point ahead (x > 0, |bearing| <= sector), from the base origin."""
    if pts is None or len(pts) == 0:
        return None
    a = np.arctan2(pts[:, 1], pts[:, 0])
    m = (pts[:, 0] > 0) & (np.abs(a) <= sector)
    return float(np.min(np.hypot(pts[m, 0], pts[m, 1]))) if m.any() else None


def points_to_scan(pts: np.ndarray, stamp, frame_id: str, r_max: float = 3.0,
                   half_fov: float = math.radians(40), inc: float = math.radians(1.0)):
    """Obstacle points -> LaserScan (nearest point per 1 deg, inf = free) for planners/costmaps."""
    n = int(round(2 * half_fov / inc)) + 1
    ranges = np.full(n, np.inf)
    if pts is not None and len(pts):
        a = np.arctan2(pts[:, 1], pts[:, 0])
        d = np.hypot(pts[:, 0], pts[:, 1])
        k = np.round((a + half_fov) / inc).astype(int)
        ok = (k >= 0) & (k < n) & (pts[:, 0] > 0)
        np.minimum.at(ranges, k[ok], d[ok])
    msg = LaserScan()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.angle_min, msg.angle_max, msg.angle_increment = -half_fov, half_fov, inc
    msg.time_increment, msg.scan_time = 0.0, 0.0
    msg.range_min, msg.range_max = 0.1, float(r_max)
    msg.ranges = [float(r) for r in ranges]
    return msg


def draw_obstacles(img: np.ndarray, result) -> None:
    """Orange boxes for camera obstacles (chairs ...) on the overlay."""
    for o in getattr(result, "obstacles", None) or []:
        x1, y1, x2, y2 = [int(v) for v in o.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(img, o.name, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1,
                    cv2.LINE_AA)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = FollowMeNode()
    except SystemExit:
        rclpy.shutdown()
        raise
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
