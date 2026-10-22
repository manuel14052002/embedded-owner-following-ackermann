#!/usr/bin/env python3
"""
verify_geometry.py — the bugs this patch set fixes, as tests.
Runs the REAL node code with the same ROS stubs verify_planners.py uses.
No ROS needed:  python3 verify_geometry.py
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_code = open(os.path.join(_HERE, "verify_planners.py")).read()
_code = _code[:_code.index('print("=" * 72)\nprint("SCENARIO A')]
exec(compile(_code, "verify_planners_stubs", "exec"))          # noqa: S102

sys.path.insert(0, os.path.join(_HERE, "a2r_bot_follow"))
from ackermann import Ackermann                                 # noqa: E402
import collision_monitor_node as cmn                            # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def head(t):
    print("=" * 72)
    print(t)
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════
head("1 — measured geometry and the Ackermann model")
car = Ackermann()
rr, rc, ro = car.radii()
check("wheelbase is the measured 0.280 m", abs(car.L - 0.280) < 1e-9, f"L={car.L}")
check("turning radius rear-axle 0.485 m", abs(rr - 0.485) < 0.002, f"{rr:.3f} m")
check("turning radius base_link 0.505 m", abs(rc - 0.505) < 0.002, f"{rc:.3f} m")
check("outer front corner sweeps 0.73 m", abs(ro - 0.732) < 0.005, f"{ro:.3f} m")

s, x, y, yaw = car.poses(car.max_steer, 2 * math.pi * rc, 0.01)
closes = math.hypot(x[-1], y[-1]) < 2e-3 and abs(yaw[-1] - 2 * math.pi) < 2e-3
check("full-lock circle closes exactly", closes,
      f"end=({x[-1]:.4f},{y[-1]:.4f})")
d = np.hypot(x + car.lr, y - rr)
check("base_link stays on its ICR circle", np.ptp(d) < 1e-3, f"spread {np.ptp(d):.2e} m")

s, x, y, yaw = car.poses(car.max_steer, -0.5, 0.01)
check("reverse on left lock swings the nose clockwise",
      x[-1] < -0.4 and yaw[-1] < -0.5, f"yaw {math.degrees(yaw[-1]):+.0f}°")

# The wheels cannot be at full lock instantly; that is why tight arcs start late.
_, _, _, yaw_ramp = car.poses(car.max_steer, 0.6, 0.01,
                              steer0=-car.max_steer, slew_per_m=math.radians(90) / 0.28)
_, _, _, yaw_inst = car.poses(car.max_steer, 0.6, 0.01)
check("servo slew makes a lock-to-lock arc turn less",
      yaw_ramp[-1] < yaw_inst[-1] - 0.2,
      f"{math.degrees(yaw_ramp[-1]):.0f}° vs {math.degrees(yaw_inst[-1]):.0f}°")

# ══════════════════════════════════════════════════════════════════════
head("2 — the car can always back away from something it stopped against")
core = cmn.CollisionCore()
wall = np.array([[0.235, yy] for yy in np.linspace(-1, 1, 81)])   # inside the margin band
v_f, _, st_f, _ = core.filter(0.3, 0.0, wall, True)
core.stopped = False
v_r, _, st_r, d_r = core.filter(-0.3, 0.0, wall, True)
check("forward into the near wall is stopped", abs(v_f) < 1e-9, st_f)
check("reverse away from it is allowed", v_r < -1e-9, f"{st_r}, clear {d_r}")

# ══════════════════════════════════════════════════════════════════════
head("3 — planner and collision monitor agree about what is drivable")
n = new_node()                                                   # noqa: F821
# Gap the body only just fits through: 0.40 m clear, body is 0.31 m wide.
gap = [(1.0, y, 0.05) for y in list(np.arange(0.20, 1.2, 0.10))
       + list(np.arange(-1.2, -0.19, 0.10))]
n._scan_cb(scan_from_obstacles(gap))                             # noqa: F821
disagree = 0
for steer in np.linspace(-n._max_steer, n._max_steer, 21):
    _, c, _ = n._rollout(0.5, float(steer), ramp=False)
    planner_ok = c < n._lethal
    mon = cmn.CollisionCore(max_look=n._horizon_m)
    d = mon.collision_distance(n._scan_pts, 0.5, float(steer), n._horizon_m)
    monitor_ok = d >= n._hard_dist(0.5)
    if planner_ok and not monitor_ok:
        disagree += 1
check("planner never proposes an arc the monitor would veto", disagree == 0,
      f"{disagree}/21 arcs disagreed")

# ══════════════════════════════════════════════════════════════════════
head("4 — obstacles near the bumper are visible (scan_min_range was 0.40 m)")
n = new_node()                                                   # noqa: F821
n._scan_cb(scan_from_obstacles([(0.32, 0.0, 0.05)]))             # noqa: F821
seen = n._scan_pts is not None and len(n._scan_pts) > 0
_, c, _ = n._rollout(0.3, 0.0, ramp=False)
check("a box 32 cm ahead is in the scan at all", seen,
      f"{0 if n._scan_pts is None else len(n._scan_pts)} points")
check("...and driving straight at it is lethal", c >= n._lethal, f"cost {c}")

n = new_node()                                                   # noqa: F821
n._scan_cb(scan_from_obstacles([]))                              # noqa: F821
check("the robot's own body is NOT mistaken for an obstacle",
      len(n._scan_pts) == 0, f"{len(n._scan_pts)} self points")

# ══════════════════════════════════════════════════════════════════════
head("5 — a collision_monitor STOP starts the escape (search mode)")
n = new_node()                                                   # noqa: F821
n._scan_cb(scan_from_obstacles([(0.55, y, 0.08)                  # noqa: F821
                                for y in np.arange(-1.4, 1.45, 0.12)]))
now = __import__("time").monotonic()
n._cm_cb(String(data='{"state": "STOP", "in_v": 0.28, "out_v": 0.0}'))   # noqa: F821
n._cm_since = now - 1.0
n._cm_t = now
check("monitor STOP of a real command counts as blocked", n._cm_blocked(now))
n._cm_cb(String(data='{"state": "PASS", "in_v": 0.28, "out_v": 0.28}'))  # noqa: F821
check("...and clears the moment it lets us go", not n._cm_blocked(now))

# The Jetson pausing mid-manoeuvre used to cancel the escape: search mode
# publishes zeros between arcs, which is exactly when the car must reverse.
n = new_node()                                                   # noqa: F821
n._scan_cb(scan_from_obstacles([(0.55, y, 0.08)                  # noqa: F821
                                for y in np.arange(-1.4, 1.45, 0.12)]))
n._state = 1          # ST_REVERSE
n._rev_steer = 0.4
n._rev_until = __import__("time").monotonic() + 5.0
n._des_v, n._des_w = 0.0, 0.0                    # Jetson went quiet
n._last_cmd_t = __import__("time").monotonic()
n._pet_tokens, n._pet_t = {"SEARCH"}, __import__("time").monotonic()
n._plan_loop()
check("escape continues while the Jetson is silent in SEARCH",
      n._state == 1 and n._out_v < -1e-3, f"state={n._state} v={n._out_v:.2f}")

# An explicit stop still ends it.
n._pet_tokens = {"IDLE"}
n._plan_loop()
check("an explicit IDLE still ends the escape", n._state == 0, f"state={n._state}")

# ══════════════════════════════════════════════════════════════════════
head("6 — the escape never reverses into unseen space")
n = new_node()                                                   # noqa: F821
# Wall ahead, and a wall close behind that the lidar CAN see.
obs = [(0.5, y, 0.08) for y in np.arange(-1.4, 1.45, 0.12)] + \
      [(-0.6, y, 0.08) for y in np.arange(-1.4, 1.45, 0.12)]
n._scan_cb(scan_from_obstacles(obs))                             # noqa: F821
n._hold_since = __import__("time").monotonic() - 10.0
n._begin_escape(__import__("time").monotonic())
backed = getattr(n, "_rev_dist", 0.0)
check("reverse distance respects the wall behind", n._state == 0 or backed <= 0.45,
      f"chose {backed:.2f} m")
check("escape distances are capped for a rear-blind lidar",
      max(n._esc_dists) <= 1.2, f"max {max(n._esc_dists)} m")

print()
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL GEOMETRY / ESCAPE CHECKS PASSED")
