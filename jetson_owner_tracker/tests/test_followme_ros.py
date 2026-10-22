"""Tests for the ROS 2 wrapper without a ROS installation.

Conversions are tested directly; the node is exercised end to end against
minimal stand-ins for rclpy / message types (it checks our wiring, not ROS).
"""

import math
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "followme_ros"))

from followme_ros.conversions import (depth_to_meters, image_to_bgr, quaternion_to_matrix,  # noqa: E402
                                      transform_to_matrix, yaw_from_matrix, yaw_to_quaternion)


def test_image_conversions():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 255                                        # red in RGB
    padded = np.concatenate([rgb.reshape(2, 9), np.zeros((2, 3), np.uint8)], 1)   # step > width*3
    msg = NS(encoding="rgb8", height=2, width=3, step=12, data=padded.tobytes())
    assert image_to_bgr(msg)[0, 0].tolist() == [0, 0, 255]
    d = np.array([[1000, 0], [2500, 65535]], np.uint16)
    m = NS(encoding="16UC1", height=2, width=2, step=4, is_bigendian=0, data=d.tobytes())
    np.testing.assert_allclose(depth_to_meters(m), [[1.0, 0.0], [2.5, 65.535]], rtol=1e-6)
    f = np.array([[1.5, np.nan]], np.float32)
    m = NS(encoding="32FC1", height=1, width=2, step=8, is_bigendian=0, data=f.tobytes())
    assert depth_to_meters(m).tolist() == [[1.5, 0.0]]


def test_transform_helpers():
    q = yaw_to_quaternion(0.7)
    R = quaternion_to_matrix(*q)
    assert yaw_from_matrix(R) == pytest.approx(0.7)
    t = NS(translation=NS(x=1, y=2, z=3), rotation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))
    T = transform_to_matrix(t)
    assert T[:3, 3].tolist() == [1, 2, 3] and yaw_from_matrix(T) == pytest.approx(0.7)


# ---------------------------------------------------------------- node stubs
class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)

    def get_subscription_count(self):
        return 1


class _Param:
    def __init__(self, v):
        self.value = v


class _Node:
    def __init__(self, name):
        self._params, self.pubs, self.services = {}, {}, {}

    def declare_parameter(self, n, v):
        self._params[n] = _Param(OVERRIDES.get(n, v))

    def get_parameter(self, n):
        return self._params[n]

    def create_publisher(self, typ, topic, qos):
        self.pubs[topic] = _Pub()
        return self.pubs[topic]

    def create_subscription(self, *a, **k):
        return None

    def create_service(self, typ, name, cb):
        self.services[name] = cb

    def create_timer(self, *a):
        return None

    def add_on_set_parameters_callback(self, cb):
        self._param_cb = cb

    def get_logger(self):
        return NS(info=print, warn=print, error=lambda *a, **k: print(*a), fatal=print)

    def get_clock(self):
        return NS(now=lambda: NS(to_msg=lambda: 0))

    def destroy_node(self):
        pass


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _msg_class(name, **defaults):
    def init(self, **kw):
        for k, v in defaults.items():
            setattr(self, k, v() if callable(v) else v)
        self.__dict__.update(kw)
    return type(name, (), {"__init__": init})


OVERRIDES = {}
TF_T = np.eye(4)


def _install_stubs(monkeypatch):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    header = lambda: _Msg(frame_id="", stamp=0)                      # noqa: E731
    vec = lambda: _Msg(x=0.0, y=0.0, z=0.0)                          # noqa: E731
    quat = lambda: _Msg(x=0.0, y=0.0, z=0.0, w=1.0)                  # noqa: E731
    pose = lambda: _Msg(position=vec(), orientation=quat())          # noqa: E731
    rclpy = mod("rclpy", ok=lambda: True, init=lambda args=None: None, spin=lambda n: None, shutdown=lambda: None)
    rclpy.time = mod("rclpy.time", Time=type("Time", (), {"__init__": lambda s, *a, **k: None,
                                                          "from_msg": staticmethod(lambda m: 0)}))
    rclpy.duration = mod("rclpy.duration", Duration=lambda **k: None)
    mod("rclpy.node", Node=_Node)
    mod("rclpy.action", ActionClient=lambda *a: NS(server_is_ready=lambda: False))
    mod("rclpy.qos", qos_profile_sensor_data=None)
    mod("geometry_msgs")
    mod("geometry_msgs.msg", PoseStamped=_msg_class("PoseStamped", header=header, pose=pose),
        Twist=_msg_class("Twist", linear=vec, angular=vec),
        Point=_msg_class("Point", x=0.0, y=0.0, z=0.0),
        PointStamped=_msg_class("PointStamped", header=header, point=vec),
        Pose=_msg_class("Pose", position=vec, orientation=quat),
        PoseArray=_msg_class("PoseArray", header=header, poses=list))
    mod("sensor_msgs")
    mod("sensor_msgs.msg", CameraInfo=object, Image=_msg_class("Image", header=header), LaserScan=object)
    mod("std_msgs")
    mod("std_msgs.msg", String=_msg_class("String"), Float32=_msg_class("Float32"))
    mod("std_srvs")
    mod("std_srvs.srv", Trigger=object)
    mod("rcl_interfaces")
    mod("rcl_interfaces.msg", SetParametersResult=_msg_class("SetParametersResult", successful=False))
    mod("visualization_msgs")
    mod("visualization_msgs.msg", Marker=_msg_class("Marker"))
    sync = NS(registerCallback=lambda cb: None)
    mod("message_filters", Subscriber=lambda *a, **k: None, ApproximateTimeSynchronizer=lambda *a, **k: sync)
    mod("nav2_msgs")
    mod("nav2_msgs.action", NavigateToPose=NS(Goal=_msg_class("Goal", pose=None)))

    class TfEx(Exception):
        pass

    class Buffer:
        def lookup_transform(self, target, source, when, timeout=None):
            q = NS(x=0.0, y=0.0, z=0.0, w=1.0)
            if source == "cam_optical":                  # optical -> map: x fwd = z, y left = -x
                from followme.pipeline import OPTICAL_TO_BASE
                return NS(transform=_mat_to_tf(TF_T @ OPTICAL_TO_BASE))
            return NS(transform=NS(translation=NS(x=0, y=0, z=0), rotation=q))

    mod("tf2_ros", Buffer=Buffer, TransformException=TfEx, TransformListener=lambda *a, **k: None)


def _mat_to_tf(T):
    R = T[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return NS(translation=NS(x=T[0, 3], y=T[1, 3], z=T[2, 3]), rotation=NS(x=x, y=y, z=z, w=w))


def test_node_end_to_end_with_stubs(monkeypatch):
    _install_stubs(monkeypatch)
    sys.modules.pop("followme_ros.followme_node", None)
    import followme_ros.followme_node as fn
    from followme.pipeline import FollowMePerception
    from followme.reid import build_feature_extractor
    from followme.segmentation import StaticSegmenter
    from test_followme_pipeline import INTR, OTHER, TARGET, render

    seg = StaticSegmenter()

    def fake_build(cfg, device, intr, path):
        cfg.reid.backend = "histogram"
        cfg.calibration.num_samples = 40
        cfg.gestures.enabled = False
        return FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), intr)

    monkeypatch.setattr(fn, "build_perception", fake_build)
    OVERRIDES["web_port"] = 0
    node = fn.FollowMeNode()
    node._on_info(NS(k=[INTR.fx, 0, INTR.cx, 0, INTR.fy, INTR.cy, 0, 0, 1], width=640, height=480))
    rng = np.random.default_rng(0)

    def feed(people):
        frame, depth, insts, _ = render(people, rng)
        seg.next = insts
        c = _Msg(encoding="bgr8", height=480, width=640, step=640 * 3, data=frame.tobytes(),
                 header=_Msg(frame_id="cam_optical", stamp=0))
        d = _Msg(encoding="16UC1", height=480, width=640, step=640 * 2, is_bigendian=0,
                 data=(depth * 1000).astype(np.uint16).tobytes(), header=c.header)
        node._on_frames(c, d)
        t0 = time.time()
        while node._latest is not None and time.time() - t0 < 5:
            time.sleep(0.005)
        time.sleep(0.12)                                   # let the worker finish this frame

    res = NS(success=None, message="")
    assert node.services["~/calibrate"](None, res).success
    k = 0
    while node.perception.mode.value == "CALIBRATING" and k < 200:
        z = 1.5 + 2.5 * (0.5 + 0.5 * math.sin(0.3 * k))
        feed([(TARGET, 0.0, z)])
        k += 1
    assert node.perception.mode.value == "IDENTIFYING"
    node.services["~/follow"](None, NS(success=None, message=""))
    for _ in range(5):
        feed([(TARGET, 0.4, 3.0), (OTHER, -0.8, 2.5)])
    state = node.pubs["~/state"].msgs[-1].data
    assert state.startswith("FOLLOW")
    goal = [c for c in node.backend.calls if c[0] == "goal"][-1][1]
    assert goal.x == pytest.approx(3.0, abs=0.15) and goal.y == pytest.approx(-0.4, abs=0.15)   # map frame
    tp = node.pubs["~/target_pose"].msgs[-1]
    assert tp.header.frame_id == node.goal_frame and tp.pose.position.x == pytest.approx(goal.x)
    assert node.pubs["~/debug_image"].msgs[-1].encoding == "bgr8"
    node.services["~/wait"](None, NS(success=None, message=""))
    assert node.controller.state.value == "WAIT"
    node.destroy_node()


def test_parse_voice():
    from followme_ros.followme_node import parse_voice
    assert parse_voice('{"action": "follow"}') == "follow"
    assert parse_voice('{"command": "STOP"}') == "wait"
    assert parse_voice("FOLLOW") == "follow"
    assert parse_voice("release owner") == "release"
    assert parse_voice('{"action": "new_owner"}') == "enroll"
    assert parse_voice('{"action": "dance"}') is None
    assert parse_voice("not json {") is None


def test_node_cmd_vel_mode_and_voice(monkeypatch):
    _install_stubs(monkeypatch)
    sys.modules.pop("followme_ros.followme_node", None)
    OVERRIDES.clear()
    OVERRIDES.update({"enable_navigation": True, "control_mode": "cmd_vel", "publish_debug_image": False, "web_port": 0})
    try:
        import followme_ros.followme_node as fn
        from followme.pipeline import FollowMePerception
        from followme.reid import build_feature_extractor
        from followme.segmentation import StaticSegmenter
        from test_followme_pipeline import INTR, TARGET, render

        seg = StaticSegmenter()

        def fake_build(cfg, device, intr, path):
            cfg.reid.backend = "histogram"
            cfg.calibration.num_samples = 40
            cfg.gestures.enabled = False
            return FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), intr)

        monkeypatch.setattr(fn, "build_perception", fake_build)
        node = fn.FollowMeNode()
        assert isinstance(node.backend, fn.CmdVelBackend)
        node._on_info(NS(k=[INTR.fx, 0, INTR.cx, 0, INTR.fy, INTR.cy, 0, 0, 1], width=640, height=480))
        rng = np.random.default_rng(0)

        def feed(people):
            frame, depth, insts, _ = render(people, rng)
            seg.next = insts
            c = _Msg(encoding="bgr8", height=480, width=640, step=640 * 3, data=frame.tobytes(),
                     header=_Msg(frame_id="cam_optical", stamp=0))
            d = _Msg(encoding="16UC1", height=480, width=640, step=640 * 2, is_bigendian=0,
                     data=(depth * 1000).astype(np.uint16).tobytes(), header=c.header)
            node._on_frames(c, d)
            t0 = time.time()
            while node._latest is not None and time.time() - t0 < 5:
                time.sleep(0.005)
            time.sleep(0.12)

        node._on_voice(NS(data='{"action": "enroll_owner"}'))          # voice starts calibration
        k = 0
        while node.perception.mode.value != "IDENTIFYING" and k < 200:
            feed([(TARGET, 0.0, 1.5 + 2.5 * (0.5 + 0.5 * math.sin(0.3 * k)))])
            k += 1
        node._on_voice(NS(data='{"action": "follow"}'))
        for _ in range(8):
            feed([(TARGET, 0.4, 3.0)])                                     # 3 m ahead, slightly right
        assert node.controller.state.value == "FOLLOW"
        twists = node.pubs["/cmd_vel"].msgs
        last = twists[-1]
        assert 0.0 < last.linear.x <= 0.3 and last.angular.z < 0          # forward, turning right
        assert node.pubs["/robot_state"].msgs[-1].data == "FOLLOWING"
        assert node.pubs["/owner_distance"].msgs[-1].data == pytest.approx(3.0, abs=0.15)
        feed([(TARGET, 0.0, 0.8)])                                         # owner walks up to 0.8 m
        assert node.pubs["/cmd_vel"].msgs[-1].linear.x == 0.0              # stops (hard stop / safety)
        assert node.pubs["/followme/owner_position"].msgs                   # camera 3-D owner point
        # camera loses the owner, the lidar fusion still reports him at 3 m, 1 m to the left
        fused = _Msg(header=_Msg(frame_id="cam_optical", stamp=0), point=_Msg(x=-1.0, y=0.0, z=3.0))
        node.perception.kf.reset()                                          # camera track expired
        node._on_fused(fused)
        node._on_voice(NS(data='{"action": "follow"}'))
        node._on_fused(fused)
        feed([])
        assert node.lidar_assist and node.controller.state.value == "FOLLOW"
        assert node.pubs["/cmd_vel"].msgs[-1].angular.z > 0                 # steering left, toward him
        node._on_voice(NS(data="STOP"))                                    # /robot_state_cmd style
        assert node.controller.state.value == "WAIT"
        assert (node.pubs["/cmd_vel"].msgs[-1].linear.x, node.pubs["/cmd_vel"].msgs[-1].angular.z) == (0.0, 0.0)
        node._on_voice(NS(data='{"action": "circular", "direction": "right", "duration": 1, "angular": 2.0}'))
        assert node._circular_step(time.monotonic())                        # arcing
        tw = node.pubs["/cmd_vel"].msgs[-1]
        assert tw.linear.x == pytest.approx(0.1) and tw.angular.z == pytest.approx(-0.8)   # right, clamped
        node._on_voice(NS(data='{"action": "stop"}'))                     # cancels immediately
        assert node._circular_until == 0.0 and node.pubs["/cmd_vel"].msgs[-1].linear.x == 0.0
        node.backend.moving, node.backend.last_cmd_t = True, time.monotonic() - 1.0
        node._watchdog()                                                   # stalled -> zero
        assert node.pubs["/cmd_vel"].msgs[-1].linear.x == 0.0 and not node.backend.moving
        node.destroy_node()
    finally:
        OVERRIDES.clear()


def test_node_refuses_to_start_alongside_a_live_instance(monkeypatch):
    """The bug this guards against: two followme_node processes silently racing on
    /cmd_vel, with no error anywhere, while every param get/set and physical test
    becomes unreliable because it is unclear which process is actually driving."""
    _install_stubs(monkeypatch)
    sys.modules.pop("followme_ros.followme_node", None)
    import followme_ros.followme_node as fn
    from followme_ros.singleton import SingletonResult

    monkeypatch.setattr(fn, "check_singleton", lambda: SingletonResult(False, 4242))
    released = []
    monkeypatch.setattr(fn, "release_singleton", lambda: released.append(True))
    try:
        fn.FollowMeNode()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "4242" in str(exc)
    assert not released           # never acquired: must not release someone else's lock


def test_eye_expression_mapping():
    from followme_ros.followme_node import eye_expression
    assert eye_expression("WAIT", "NO_OWNER") == "idle"
    assert eye_expression("WAIT", "CLAIMING") == "thinking"
    assert eye_expression("FOLLOW", "OWNED") == "following"
    assert eye_expression("WAIT", "OWNED") == "stopped"
    assert eye_expression("SEARCH", "OWNED") == "circular"
    assert eye_expression("STEADY", None) == "unknown"
    assert eye_expression("WAIT", "NO_OWNER", armed=True) == "listening"


def test_bridge_helpers():
    from followme_ros.followme_node import bridge_state, parse_point, parse_voice, parse_voice_args
    assert parse_voice('{"action": "wake_trigger"}') == "enroll"
    assert parse_voice('{"action": "go_to"}') == "follow"
    assert parse_voice('{"action": "idle"}') == "wait"
    assert parse_voice('{"action": "circular", "direction": "right"}') == "circular"
    assert parse_voice_args('{"action": "circular", "duration": 4}')["duration"] == 4
    assert parse_point('{"x": 320, "y": 200}') == (320.0, 200.0)
    assert parse_point('{"u": 0.5, "v": 0.4}') == (0.5, 0.4)
    assert parse_point("[10, 20]") == (10.0, 20.0) and parse_point("12,34") == (12.0, 34.0)
    assert parse_point("nonsense") is None
    assert bridge_state("WAIT", "NO_OWNER", False, False) == "IDLE"
    assert bridge_state("WAIT", "NO_OWNER", True, False) == "SEARCHING"
    assert bridge_state("WAIT", "CLAIMING", False, False) == "ENROLLING"
    assert bridge_state("FOLLOW", "OWNED", False, False) == "FOLLOWING"
    assert bridge_state("WAIT", "OWNED", False, False) == "OWNED"


def test_front_distance():
    from followme_ros.followme_node import front_distance
    d = np.full((480, 640), 3.0, np.float32)
    assert front_distance(d) == pytest.approx(3.0)
    d[150:300, 250:400] = 0.6                       # someone right in front
    assert front_distance(d) == pytest.approx(0.6)
    d2 = np.full((480, 640), 3.0, np.float32)
    d2[400:, :] = 0.5                               # floor near the bottom is ignored
    assert front_distance(d2) == pytest.approx(3.0)
    assert front_distance(None) is None


def test_mirror_x_and_angular_sign_are_actually_live_settable(monkeypatch):
    """Regression test for the bug where 'ros2 param set angular_sign ...' reported success
    but the running node kept using the value passed at startup."""
    _install_stubs(monkeypatch)
    sys.modules.pop("followme_ros.followme_node", None)
    OVERRIDES.clear()
    OVERRIDES.update({"enable_navigation": True, "control_mode": "cmd_vel", "publish_debug_image": False,
                      "angular_sign": 1.0})
    try:
        import followme_ros.followme_node as fn
        from followme.pipeline import FollowMePerception
        from followme.reid import build_feature_extractor
        from followme.segmentation import StaticSegmenter
        from test_followme_pipeline import INTR

        seg = StaticSegmenter()

        def fake_build(cfg, device, intr, path):
            cfg.reid.backend = "histogram"
            cfg.calibration.num_samples = 40
            cfg.gestures.enabled = False
            return FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), intr)

        monkeypatch.setattr(fn, "build_perception", fake_build)
        node = fn.FollowMeNode()
        node._on_info(NS(k=[INTR.fx, 0, INTR.cx, 0, INTR.fy, INTR.cy, 0, 0, 1], width=640, height=480))

        # sanity: params start at the declared defaults
        assert node.backend.angular_sign == 1.0
        assert node.cfg.localization.mirror_x is False

        from rcl_interfaces.msg import SetParametersResult

        result = node._param_cb([NS(name="angular_sign", value=-1.0), NS(name="mirror_x", value=True)])
        assert isinstance(result, SetParametersResult) and result.successful
        # this is the actual bug we are guarding against: the live value must reach the
        # running objects immediately, not just the (mocked) parameter server
        assert node.backend.angular_sign == -1.0
        assert node.cfg.localization.mirror_x is True

        result2 = node._param_cb([NS(name="mirror_x", value=False)])
        assert result2.successful and node.cfg.localization.mirror_x is False
        node.destroy_node()
    finally:
        OVERRIDES.clear()


def test_lidar_front_min_sector_and_mounting():
    import math as m
    from followme_ros.followme_node import lidar_front_min
    n = 360
    amin, inc = -m.pi, 2 * m.pi / n
    r = np.full(n, 5.0)
    r[180] = 0.4                                   # straight ahead (angle 0) at 0.4 m
    assert lidar_front_min(r, amin, inc, 0.1, 12.0) == pytest.approx(0.4)
    r2 = np.full(n, 5.0)
    r2[180 + 90] = 0.3                             # 90 deg to the left: outside a 45 deg sector
    assert lidar_front_min(r2, amin, inc, 0.1, 12.0) == pytest.approx(5.0)
    r3 = np.full(n, 5.0)
    r3[0] = 0.4                                    # directly BEHIND in the laser frame ...
    assert lidar_front_min(r3, amin, inc, 0.1, 12.0) == pytest.approx(5.0)
    # ... but in FRONT of the robot if the lidar is mounted rotated by 180 deg
    assert lidar_front_min(r3, amin, inc, 0.1, 12.0, laser_yaw=m.pi) == pytest.approx(0.4)
    assert lidar_front_min(np.full(n, np.inf), amin, inc, 0.1, 12.0) is None


def test_owner_side_text_is_from_the_robots_point_of_view():
    from followme_ros.followme_node import owner_side_text
    assert "ROBOT'S LEFT" in owner_side_text(0.5) and "your RIGHT" in owner_side_text(0.5)
    assert "ROBOT'S RIGHT" in owner_side_text(-0.5) and "your LEFT" in owner_side_text(-0.5)
    assert "STRAIGHT" in owner_side_text(0.01)
    assert "not seen" in owner_side_text(None)


def test_circular_obeys_hard_stop_and_owner_is_persisted(monkeypatch, tmp_path):
    _install_stubs(monkeypatch)
    sys.modules.pop("followme_ros.followme_node", None)
    OVERRIDES.clear()
    OVERRIDES.update({"enable_navigation": True, "control_mode": "cmd_vel", "publish_debug_image": False,
                      "web_port": 0, "target_model_path": str(tmp_path / "owner.npz")})
    try:
        import followme_ros.followme_node as fn
        from followme.calibration import TargetModel, fit_target_model
        from followme.config import CalibrationConfig

        node = fn.FollowMeNode()
        # --- S1: the voice arc may steer but must not drive forward while a stop is active
        node._circular_cmd, node._circular_until = (0.1, 0.4), time.monotonic() + 5
        node.backend.allow_forward = False
        assert node._circular_step(time.monotonic())
        tw = node.pubs["/cmd_vel"].msgs[-1]
        assert tw.linear.x == 0.0 and tw.angular.z == pytest.approx(0.4)
        node.backend.allow_forward = True
        node._circular_step(time.monotonic())
        assert node.pubs["/cmd_vel"].msgs[-1].linear.x == pytest.approx(0.1)
        # --- R1: claimed owner is saved, released owner file is removed
        rng = np.random.default_rng(0)
        model = fit_target_model(rng.normal(0, 1, (30, 8)), CalibrationConfig())
        node.perception = NS(model=model)
        node._persist_owner("claimed")
        assert (tmp_path / "owner.npz").is_file()
        assert TargetModel.load(tmp_path / "owner.npz").dim == 8
        node._persist_owner("released")
        assert not (tmp_path / "owner.npz").is_file()
        node.destroy_node()
    finally:
        OVERRIDES.clear()


def test_steer_test_recommendations():
    from followme_ros.steer_test import recommended
    assert recommended(True, None) == {"ANGULAR_SIGN": "1.0"}
    assert recommended(False, None) == {"ANGULAR_SIGN": "-1.0"}
    assert recommended(None, True) == {"MIRROR_X": "false"}
    assert recommended(None, False) == {"MIRROR_X": "true"}
    assert recommended(True, False) == {"ANGULAR_SIGN": "1.0", "MIRROR_X": "true"}
