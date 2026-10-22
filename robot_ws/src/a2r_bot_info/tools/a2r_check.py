#!/usr/bin/env python3
"""
a2r_check.py - cross-file consistency check for the A2R_bot workspace
======================================================================
Many bugs in this project were two files disagreeing: a topic nobody
published, a yaml value silently overriding a new code default, two inputs on
one topic. This reads the configs, URDF, firmware Config.h and the planner
code and checks every invariant the pipeline depends on.

    python3 a2r_check.py [path/to/a2r_bot_ws]        exit code 1 if anything FAILs
"""
import math, os, re, sys
try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

WS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/A2R_BOT/a2r_bot_ws"))
S = os.path.join(WS, "src")
fails = warns = 0

def load(rel):
    p = os.path.join(S, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return yaml.safe_load(f) or {}

def node(doc, name):
    return ((doc or {}).get(name) or {}).get("ros__parameters", {})

def text(rel):
    p = os.path.join(S, rel)
    return open(p).read() if os.path.exists(p) else ""

def define(src, name):
    m = re.search(rf"^#define\s+{name}\s+([-0-9.]+)f?", src, re.M)
    return float(m.group(1)) if m else None

def default(src, name):
    m = re.search(rf'declare_parameter\(\s*"{name}"\s*,\s*([-0-9.]+)', src)
    return float(m.group(1)) if m else None

def res(ok, msg, level="FAIL"):
    global fails, warns
    if ok:
        print(f"  ok    {msg}")
    elif level == "WARN":
        warns += 1; print(f"  WARN  {msg}")
    else:
        fails += 1; print(f"  FAIL  {msg}")

def same(label, vals, tol=1e-6, level="FAIL"):
    known = {k: v for k, v in vals.items() if v is not None}
    if not known:
        return res(False, f"{label}: not found anywhere", "WARN")
    ref = next(iter(known.values()))
    ok = all(abs(v - ref) <= tol for v in known.values())
    res(ok, f"{label}: " + ", ".join(f"{k}={v:g}" for k, v in known.items()), level)

P = load("a2r_bot_follow/config/params.yaml")
FU = load("a2r_bot_sensor_fusion/config/fusion.yaml")
DE = load("a2r_bot_description/config/description.yaml")
AR = load("a2r_bot_arduino/config/arduino.yaml")
if P is None:
    sys.exit(f"no a2r_bot_follow/config/params.yaml under {S}")
pl, cm, mux, fm, of = (node(P, n) for n in ("local_planner_bridge", "collision_monitor", "cmd_mux",
                                            "follow_manager", "owner_fusion"))
br = node(AR, "serial_bridge_node")
cfg = text("a2r_bot_arduino/firmware/a2r_bot_firmware/Config.h")
ack = text("a2r_bot_follow/a2r_bot_follow/ackermann.py")
xac = text("a2r_bot_description/urdf/common.xacro")
plc = text("a2r_bot_follow/a2r_bot_follow/smart_planner_node.py")
xa = lambda n: (lambda m: float(m.group(1)) if m else None)(re.search(rf'name="{n}"\s+value="([-0-9.]+)"', xac))
ak = lambda n: (lambda m: float(m.group(1)) if m else None)(re.search(rf"^{n}\s*=\s*([-0-9.]+)", ack, re.M))

print(f"A2R_bot consistency check - {WS}\n")
print("== geometry (planner, monitor, EKF, URDF and ackermann.py must agree)")
same("wheelbase m", {"planner": pl.get("wheelbase_m"), "fusion": node(FU, "ekf_state_node").get("wheelbase_m"),
                     "description": node(DE, "wheel_state_node").get("wheelbase_m"),
                     "urdf": xa("wheelbase"), "ackermann.py": ak("WHEELBASE_M")}, 1e-3)
same("max steer deg", {"planner": pl.get("max_steer_deg"), "monitor": cm.get("max_steer_deg"),
                       "fusion": node(FU, "ekf_state_node").get("max_steer_deg"),
                       "bridge": br.get("max_steer_deg"), "ackermann.py": ak("MAX_STEER_DEG")}, 1e-6)
for k, a in (("footprint_front_m", "FOOT_FRONT_M"), ("footprint_rear_m", "FOOT_REAR_M"),
             ("footprint_half_width_m", "FOOT_HALF_W_M")):
    same(k, {"planner": pl.get(k), "monitor": cm.get(k), "ackermann.py": ak(a)}, 1e-3)
same("lidar_offset_x", {"planner": pl.get("lidar_offset_x"), "monitor": cm.get("lidar_offset_x"),
                        "ackermann.py": ak("LIDAR_X_M")}, 1e-3)
pm, mm = pl.get("footprint_margin_m"), cm.get("margin_m")
res(pm is not None and mm is not None and pm > mm,
    f"planner footprint_margin_m ({pm}) > collision_monitor margin_m ({mm}) - else the monitor vetoes planner paths")

print("\n== topics (every subscriber must have the publisher it expects)")
inputs = dict((x.split(":")[0], x.split(":")[1:]) for x in str(mux.get("inputs", "")).split(",") if ":" in x)
topics = [v[0] for v in inputs.values()]
res(len(topics) == len(set(topics)), f"mux inputs on distinct topics: {', '.join(topics)}")
res("/cmd_vel" not in topics, "Jetson tracker /cmd_vel is NOT a mux input (it is m/s, the mux carries -1..1)")
res(cm.get("input_topic") == mux.get("output_topic"),
    f"monitor input {cm.get('input_topic')} == mux output {mux.get('output_topic')}")
res(cm.get("output_topic", "/local_planner/cmd_out") == br.get("drive_topic", "/local_planner/cmd_out"),
    f"monitor output {cm.get('output_topic')} == bridge drive_topic {br.get('drive_topic')}")
res(inputs.get("navigation", [None])[0] == pl.get("nav_cmd_topic", "/cmd_vel_nav"),
    f"mux navigation {inputs.get('navigation', [None])[0]} == planner nav topic {pl.get('nav_cmd_topic', '/cmd_vel_nav')}")
res(inputs.get("escape", [None])[0] == pl.get("escape_cmd_topic", "/cmd_vel_escape"),
    f"mux escape {inputs.get('escape', [None])[0]} == planner escape topic {pl.get('escape_cmd_topic', '/cmd_vel_escape')}")
scans = {"planner": pl.get("scan_topic"), "monitor": cm.get("scan_topic"), "owner_fusion": of.get("scan_topic")}
res(len({v for v in scans.values() if v}) == 1 and "/scan/fused" not in scans.values(),
    "scan topics agree and are published: " + ", ".join(f"{k}={v}" for k, v in scans.items()))
res(fm.get("goal_topic", "/follow/goal") == pl.get("goal_topic", "/follow/goal"),
    f"follow_manager goal {fm.get('goal_topic', '/follow/goal')} == planner goal {pl.get('goal_topic', '/follow/goal')}")

print("\n== command chain (fractions -1..1 from planner to firmware)")
lo, hi = br.get("min_throttle"), br.get("max_throttle")
same("ESC window low",  {"bridge min_throttle": lo, "firmware ESC_THR_LO_FRAC": define(cfg, "ESC_THR_LO_FRAC")}, 1e-6)
same("ESC window high", {"bridge max_throttle": hi, "firmware ESC_THR_HI_FRAC": define(cfg, "ESC_THR_HI_FRAC")}, 1e-6)
mps, cap = pl.get("max_speed_mps"), pl.get("max_speed_cmd")
res(cap is not None and hi is not None and cap <= hi + 1e-9, f"planner max_speed_cmd {cap} <= bridge max_throttle {hi}")
md = pl.get("min_drive_speed")
if md and mps and lo:
    res(md / mps >= lo - 1e-3, f"planner min_drive {md} m/s = {md/mps:.3f} >= bridge min_throttle {lo} (else the bridge lifts it)", "WARN")
res(abs((cm.get("min_drive_speed") or 0) - (md or 0)) < 1e-6,
    f"monitor min_drive_speed {cm.get('min_drive_speed')} == planner {md}", "WARN")
fl, fh, rl, rh = (define(cfg, n) for n in ("ESC_FWD_LO_US", "ESC_FWD_HI_US", "ESC_REV_LO_US", "ESC_REV_HI_US"))
lim = define(cfg, "ESC_FWD_LIMIT_US")
if None not in (fl, fh, rl, rh):
    res(fl <= fh and rl >= rh, f"ESC anchors monotonic: fwd {fl:g}->{fh:g} us, rev {rl:g}->{rh:g} us")
    res(lim is None or lim < 1790, f"ESC_FWD_LIMIT_US {lim} below the bench 'excessively fast' 1790 us")
    res(1500 - rl > 25 and fl - 1500 > 25, "both anchors outside the ESC neutral band (+-25 us)")

print("\n== timing (the chain must outlive its own gaps)")
send_dt = 1.0 / 50.0                                     # planner _send_loop
for name, v in inputs.items():
    to = float(v[2]) if len(v) > 2 else 0.0
    res(to >= 5 * send_dt, f"mux {name} timeout {to:g} s >= 5 planner frames ({5*send_dt:g} s)",
        "FAIL" if to < 2 * send_dt else "WARN")
bt = br.get("cmd_timeout_s"); fsm = define(cfg, "CMD_TIMEOUT_MS") or define(cfg, "FAILSAFE_MS")
if bt is not None and fsm is not None:
    res(bt * 1000 <= fsm + 1, f"bridge cmd_timeout {bt} s <= firmware failsafe {fsm:g} ms (bridge zeroes first)", "WARN")
st = cm.get("scan_timeout_s")
res(st is not None and st >= 0.25, f"monitor scan_timeout_s {st} >= 2.5 lidar frames at 10 Hz")
res(cm.get("ignore_stale_scan") in (False, None), "collision_monitor ignore_stale_scan is OFF (true disables ALL obstacle stops)")
esc_d, brk, gap = pl.get("esc_reverse_delay_s"), define(cfg, "ESC_BRAKE_STILL_MS"), define(cfg, "ESC_NEUTRAL_GAP_MS")
if None not in (esc_d, brk, gap):
    res(esc_d * 1000 >= brk + gap - 1, f"planner esc_reverse_delay_s {esc_d} covers the firmware handshake {brk + gap:g} ms")

print("\n== yaml keys the code actually reads (a typo is silently ignored by ROS)")
_F = "a2r_bot_follow/a2r_bot_follow/"          # v2.5: every node, not just the planner
for nm, sec, code in (("local_planner_bridge", pl, plc),
                      ("follow_manager", node(P, "follow_manager"), text(_F + "follow_manager_node.py")),
                      ("collision_monitor", node(P, "collision_monitor"), text(_F + "collision_monitor_node.py")),
                      ("owner_fusion", node(P, "owner_fusion"), text(_F + "owner_fusion_node.py")),
                      ("cmd_mux", node(P, "cmd_mux"), text(_F + "cmd_mux_node.py"))):
    unknown = [k for k in sec if f'"{k}"' not in code and f"'{k}'" not in code]   # v2.5: either quote style
    res(not unknown, f"{nm}: every params.yaml key is declared in the code" + (f" - unknown: {unknown}" if unknown else ""), "WARN")

print("\n== firmware battery gauge (log: 15-19 % shown at a resting 45 %)")
_bat = text("a2r_bot_arduino/firmware/a2r_bot_firmware/Battery.cpp")
_ino = text("a2r_bot_arduino/firmware/a2r_bot_firmware/a2r_bot_firmware.ino")
res("(pack_v + sag)" in _bat, "sag compensated per sample BEFORE the EMA (else stale sag after every stop)", "WARN")
res(define(cfg, "BATTERY_FALL_LIMIT_PPS") is not None, "gauge fall rate-limited while loaded (a sag spike is not lost charge)", "WARN")
res("Drive::escPulseUs()" in _ino, "battery load = ESC pulse, not the 0.14-0.20 command window", "WARN")

print(f"\n{fails} FAIL, {warns} WARN")
sys.exit(1 if fails else 0)
