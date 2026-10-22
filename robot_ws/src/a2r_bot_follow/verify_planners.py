#!/usr/bin/env python3
"""Offline verification: run the REAL node code with stubbed rclpy/msgs.
Scenarios per planner: clear path, obstacle ahead (must swerve, keep speed),
boxed-in (must reverse-escape, never permanently stall)."""
import sys, os, types, math, time
import numpy as np

# ── Stub ROS modules ─────────────────────────────────────────────────
def mod(name):
    m = types.ModuleType(name); sys.modules[name] = m; return m

class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)

class Vector3(_Msg):
    def __init__(self): self.x = self.y = self.z = 0.0
class Twist(_Msg):
    def __init__(self): self.linear = Vector3(); self.angular = Vector3()
class Header(_Msg):
    def __init__(self): self.stamp = None; self.frame_id = ""
class Point(_Msg):
    def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
class Quaternion(_Msg):
    def __init__(self): self.x = self.y = self.z = 0.0; self.w = 1.0
class Pose(_Msg):
    def __init__(self): self.position = Point(); self.orientation = Quaternion()
class PoseStamped(_Msg):
    def __init__(self): self.header = Header(); self.pose = Pose()
class MapMeta(_Msg):
    def __init__(self):
        self.resolution = 0.0; self.width = self.height = 0
        self.origin = Pose()
class OccupancyGrid(_Msg):
    def __init__(self): self.header = Header(); self.info = MapMeta(); self.data = []
class Path(_Msg):
    def __init__(self): self.header = Header(); self.poses = []
class LaserScan(_Msg):
    def __init__(self):
        self.header = Header()
        self.angle_min = -math.pi; self.angle_increment = math.pi/180
        self.range_min = 0.1; self.range_max = 8.0; self.ranges = []
class Imu(_Msg):
    def __init__(self):
        self.header = Header(); self.linear_acceleration = Vector3()
        self.angular_velocity = Vector3(); self.orientation_covariance = [0.0]*9
class BatteryState(_Msg):
    def __init__(self): self.header = Header(); self.voltage = 0.0; self.present = False
class String(_Msg):
    def __init__(self, data=""): self.data = data
class Float32(_Msg):
    def __init__(self, data=0.0): self.data = data
class Bool(_Msg):
    def __init__(self, data=False): self.data = data
class Color:
    def __init__(self): self.r = self.g = self.b = self.a = 0.0
class Scale:
    def __init__(self): self.x = self.y = self.z = 0.0
class Lifetime:
    def __init__(self): self.nanosec = 0
class Marker(_Msg):
    ARROW=0; CUBE=1; SPHERE=2; LINE_STRIP=4; LINE_LIST=5
    SPHERE_LIST=7; TEXT_VIEW_FACING=9; ADD=0; DELETE=2
    def __init__(self):
        self.header = Header(); self.ns=""; self.id=0; self.type=0; self.action=0
        self.pose = Pose(); self.scale = Scale(); self.color = Color()
        self.lifetime = Lifetime(); self.points = []; self.text = ""
class MarkerArray(_Msg):
    def __init__(self, markers=None): self.markers = markers or []

geom = mod("geometry_msgs"); gm = mod("geometry_msgs.msg")
gm.Twist, gm.Point, gm.PoseStamped = Twist, Point, PoseStamped
navm = mod("nav_msgs"); nm = mod("nav_msgs.msg")
nm.OccupancyGrid, nm.Path = OccupancyGrid, Path
senm = mod("sensor_msgs"); sm = mod("sensor_msgs.msg")
class PointField(_Msg):
    FLOAT32 = 7
    def __init__(self, name="", offset=0, datatype=7, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count
class PointCloud2(_Msg):
    def __init__(self):
        self.header = Header(); self.height = self.width = 0; self.fields = []
        self.is_bigendian = False; self.point_step = self.row_step = 0
        self.data = b""; self.is_dense = True
sm.LaserScan, sm.Imu, sm.BatteryState = LaserScan, Imu, BatteryState
sm.PointCloud2, sm.PointField = PointCloud2, PointField
stdm = mod("std_msgs"); stm = mod("std_msgs.msg")
stm.String, stm.Float32, stm.Bool = String, Float32, Bool
vizm = mod("visualization_msgs"); vm = mod("visualization_msgs.msg")
vm.Marker, vm.MarkerArray = Marker, MarkerArray

rclpy = mod("rclpy")
rnode = mod("rclpy.node"); rqos = mod("rclpy.qos")
class _Param:
    def __init__(self, v): self.value = v
class _Logger:
    def info(self, m, **k): pass
    def warn(self, m, **k): pass
    def error(self, m, **k): print("  [ERR]", m)
class _Clock:
    def now(self):
        class T:
            def to_msg(self): return None
        return T()
class StubNode:
    def __init__(self, name):
        self._params = {}
        self.published = {}
    def declare_parameter(self, n, v): self._params[n] = _Param(v)
    def get_parameter(self, n): return self._params[n]
    def create_subscription(self, *a, **k): return None
    def create_publisher(self, typ, topic, qos):
        outer = self
        class P:
            def publish(self, msg): outer.published[topic] = msg
        return P()
    def create_timer(self, *a, **k): return None
    def create_service(self, *a, **k): return None
    def get_logger(self): return _Logger()
    def get_clock(self): return _Clock()
    def destroy_node(self): pass
    def add_on_set_parameters_callback(self, cb): self._param_cb = cb
rnode.Node = StubNode
class _E:  # enum stubs
    RELIABLE = BEST_EFFORT = KEEP_LAST = TRANSIENT_LOCAL = VOLATILE = 1
rqos.QoSProfile = lambda **k: None
rqos.ReliabilityPolicy = _E; rqos.HistoryPolicy = _E; rqos.DurabilityPolicy = _E
rqos.qos_profile_sensor_data = None   # the planner imports it (it used to crash without)
rclpy.node = rnode; rclpy.qos = rqos
rclpy.init = lambda **k: None; rclpy.ok = lambda: True
rclpy.spin = lambda n: None; rclpy.shutdown = lambda: None

# ── Stub rcl_interfaces (the parameter-callback return type) ─────────
rcli = mod("rcl_interfaces"); rclim = mod("rcl_interfaces.msg")
class SetParametersResult:
    def __init__(self, successful=True, reason=""):
        self.successful = successful; self.reason = reason
rclim.SetParametersResult = SetParametersResult
rcli.msg = rclim

# ── Import the real node ─────────────────────────────────────────────
# Resolve relative to THIS file so the script runs from anywhere, which
# matters because it is invoked both from the workspace root and from the
# package directory.
import importlib
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "a2r_bot_follow"))
lpb = importlib.import_module("smart_planner_node")

def make_node():
    n = lpb.LocalPlannerBridgeNode.__new__(lpb.LocalPlannerBridgeNode)
    StubNode.__init__(n, "x")
    # replicate __init__ but with dry_run forced on: easiest is call real init
    return n

def build_node():
    n = lpb.LocalPlannerBridgeNode.__new__(lpb.LocalPlannerBridgeNode)
    StubNode.__init__(n, "local_planner_bridge")
    lpb.LocalPlannerBridgeNode.__init__.__wrapped__ if False else None
    return n

# Instead: set dry_run default by patching declare: simplest — call real __init__
# after forcing HAS_SERIAL False so it goes dry.
lpb.HAS_SERIAL = False

def new_node():
    n = lpb.LocalPlannerBridgeNode("stub") if False else None
    n = lpb.LocalPlannerBridgeNode.__new__(lpb.LocalPlannerBridgeNode)
    StubNode.__init__(n, "local_planner_bridge")
    lpb.LocalPlannerBridgeNode.__init__(n)
    return n
# LocalPlannerBridgeNode.__init__ calls super().__init__("local_planner_bridge")
# which re-inits StubNode — fine.

def scan_from_obstacles(obstacles, lyaw=0.0):
    """obstacles: list of (x, y, radius) in base frame → 360° scan msg."""
    msg = LaserScan()
    ranges = []
    for deg in range(360):
        a = -math.pi + math.radians(deg)
        best = 7.9
        for ox, oy, orad in obstacles:
            # ray-circle intersection
            dx, dy = math.cos(a), math.sin(a)
            fx, fy = -ox, -oy
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - orad * orad
            disc = b * b - 4 * c
            if disc >= 0:
                t = (-b - math.sqrt(disc)) / 2
                if 0.05 < t < best:
                    best = t
        ranges.append(best)
    msg.ranges = ranges
    return msg

def set_cmd(n, v, w):
    t = Twist(); t.linear.x = v; t.angular.z = w
    n._cmd_cb(t)

def out(n):
    return n._out_v * n._max_mps / 1.0, n._out_s * n._max_steer

# All eight. RRT joined the original seven; it is the only sampling planner
# in the set, so it is also the only one whose output varies between runs on
# identical input — the seeding below keeps this script reproducible.
PLANNERS = ["_plan_dwa", "_plan_apf", "_plan_teb", "_plan_vfh",
            "_plan_pure_pursuit", "_plan_fgm", "_plan_lattice", "_plan_rrt"]

import random as _random


def _seeded(n, pl, fn, *a):
    """Call a planner, seeding the RNG first so RRT is reproducible."""
    if pl == "_plan_rrt":
        _random.seed(12345)
        n._rrt_tree = None
    return fn(*a)

print("=" * 72)
print("SCENARIO A — clear path, desired 0.6 m/s straight")
print("=" * 72)
for pl in PLANNERS:
    n = new_node()
    n._scan_cb(scan_from_obstacles([]))
    v, s = _seeded(n, pl, getattr(n, pl), 0.6, 0.0)
    ok = v >= n._min_drive - 1e-6 and abs(s) < math.radians(8)
    print(f"  {pl:20s} v={v:5.2f} m/s steer={math.degrees(s):+6.1f}°  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, pl

print("=" * 72)
print("SCENARIO B — obstacle (r=0.15 m) 1.0 m dead ahead, desired straight")
print("=" * 72)
for pl in PLANNERS:
    n = new_node()
    n._scan_cb(scan_from_obstacles([(1.0, 0.0, 0.15)]))
    v, s = _seeded(n, pl, getattr(n, pl), 0.6, 0.0)
    moving = v >= n._min_drive - 1e-6
    swerves = abs(s) > math.radians(4)
    ok = moving and swerves
    print(f"  {pl:20s} v={v:5.2f} m/s steer={math.degrees(s):+6.1f}°  "
          f"{'PASS' if ok else 'FAIL'} (moving={moving}, swerves={swerves})")
    assert ok, pl

print("=" * 72)
print("SCENARIO C — wall across the front at 0.55 m (boxed): planner must")
print("             yield stop → stall manager must trigger REVERSE escape")
print("=" * 72)
wall = [(0.55, y, 0.08) for y in np.arange(-1.4, 1.45, 0.12)]
for pl_idx, pl in enumerate(PLANNERS):
    n = new_node()
    n._scan_cb(scan_from_obstacles(wall))
    v, s = _seeded(n, pl, getattr(n, pl), 0.6, 0.0)
    planner_stops = abs(v) < 1e-3
    # Drive the full _plan_loop through the REAL dispatch. The original
    # version of this script monkey-patched _plan_dwa, because dispatch was a
    # comment-swap and there was no other way in. Dispatch is now a parameter
    # backed by self._planner_fn, so the test can just point it at the planner
    # under test — which means it exercises the same code path the robot does.
    n2 = new_node()
    n2._planner_name = pl.replace("_plan_", "")
    n2._planner_fn = getattr(n2, pl)
    n2._scan_cb(scan_from_obstacles(wall))
    set_cmd(n2, 0.6, 0.0)
    # The escape is a THREE-POINT TURN, not a plain reverse:
    #
    #     FOLLOW  --stall-->  PRESTEER  --servo swung-->  REVERSE  -->  FOLLOW
    #
    # The pre-steer phase exists because an Ackermann car reversing with the
    # wheels still pointed forward just backs into whatever it was already
    # stuck against. It swings the servo to lock FIRST, holds while the servo
    # physically travels, and only then applies reverse throttle.
    #
    # An earlier version of this script expected FOLLOW to go straight to
    # REVERSE and reported a failure when it did not. It was the test that
    # was out of date, not the planner.
    escaped = False
    presteered = False
    t0 = time.monotonic()
    n2._plan_loop()                      # pass 1: blocked, stall timer arms
    n2._stall_since = t0 - 10.0          # simulate stall_time elapsed
    n2._last_rev_end = t0 - 10.0
    set_cmd(n2, 0.6, 0.0)
    n2._plan_loop()                      # pass 2: should enter PRESTEER
    presteered = (n2._state == lpb.ST_PRESTEER)
    if presteered:
        # Simulate the servo having finished its swing.
        n2._presteer_until = time.monotonic() - 0.01
        set_cmd(n2, 0.6, 0.0)
        n2._plan_loop()                  # pass 3: should enter REVERSE
        if n2._state == lpb.ST_REVERSE:
            vv, ss = out(n2)
            escaped = vv < -0.05
    ok = planner_stops and presteered and escaped
    print(f"  {pl:20s} stops={str(planner_stops):5s} presteer={str(presteered):5s} "
          f"reverse={str(escaped):5s}  {'PASS' if ok else 'FAIL'}")
    assert ok, pl

print("=" * 72)
print("SCENARIO D — narrow 0.95 m corridor: must keep driving (no paralysis)")
print("=" * 72)
corridor = [(-0.5 + 0.15 * i, 0.48, 0.05) for i in range(20)] + \
           [(-0.5 + 0.15 * i, -0.48, 0.05) for i in range(20)]
n = new_node()
n._scan_cb(scan_from_obstacles(corridor))
v, s = n._plan_dwa(0.6, 0.0)
ok = v >= n._min_drive - 1e-6
print(f"  DWA in corridor      v={v:5.2f} m/s steer={math.degrees(s):+6.1f}°  "
      f"{'PASS' if ok else 'FAIL'}")
assert ok

print("=" * 72)
print("SCENARIO E — watchdog & threshold: zero cmd → STOP; stale cmd → STOP")
print("=" * 72)
n = new_node()
n._scan_cb(scan_from_obstacles([]))
set_cmd(n, 0.0, 0.0)               # owner within threshold
n._plan_loop()
v, s = out(n)
print(f"  zero cmd  → out v={v:.2f}  {'PASS' if abs(v) < 1e-6 else 'FAIL'}")
assert abs(v) < 1e-6
set_cmd(n, 0.6, 0.0)
n._last_cmd_t = time.monotonic() - 5.0   # stale
n._plan_loop()
v, s = out(n)
print(f"  stale cmd → out v={v:.2f}  {'PASS' if abs(v) < 1e-6 else 'FAIL'}")
assert abs(v) < 1e-6

print("=" * 72)
print("SCENARIO F — speed law sanity: cost 89 must NOT collapse speed")
print("=" * 72)
n = new_node()
v = n._speed_law(0.6, 89)
print(f"  des=0.6, max_cost=89 → v={v:.2f} m/s (floor {n._min_drive})  "
      f"{'PASS' if v >= n._min_drive else 'FAIL'}")
assert v >= n._min_drive

print("\nALL SCENARIOS PASSED — all 8 planners avoid, none stalls, escape works")
