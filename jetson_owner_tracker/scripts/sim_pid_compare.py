"""PID vs pure pursuit, closed loop, with realistic disturbances (servo trim, noise, delay)."""
import math, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sim_steering as S
from followme.direct_control import DirectFollowController
from followme.localization import Goal2D

L, DMAX, DT = 0.28, math.radians(30), 0.05

def run(path, law, trim_deg=0.0, noise_deg=0.0, delay=2, gains=(1.0, 0.0, 0.0), seed=0, steps=2400):
    rng = np.random.default_rng(seed)
    cfg = S.nav_cfg()
    pp = DirectFollowController(cfg, 0.5)
    x = y = yaw = 0.0; dw = 0.0
    integ, prev_a, d_filt = 0.0, None, 0.0
    vis, bear, q = [], [], []
    v = 0.0
    for k in range(steps):
        t = k * DT
        px, py = path(t)
        cx, cy = x + S.CAM_X * math.cos(yaw), y + S.CAM_X * math.sin(yaw)
        dx, dy = px - cx, py - cy
        f = math.cos(yaw) * dx + math.sin(yaw) * dy; l = -math.sin(yaw) * dx + math.cos(yaw) * dy
        b = math.atan2(l, f); r = math.hypot(f, l)
        visible = f > 0.3 and abs(b) < S.HFOV and r < 6
        q.append((b + math.radians(noise_deg) * rng.standard_normal(), r) if visible else None)
        m = q.pop(0) if len(q) > delay else None
        if m is None:
            v, delta = 0.0, None; pp.reset(); integ = 0.0; prev_a = None
        else:
            a, d = m
            goal = Goal2D(d * math.cos(a), d * math.sin(a), a, d)
            vv, ww = pp.command(goal, t)
            v = vv
            if law == "pp":
                delta = pp.delta
            else:
                kp, ki, kd = gains
                if v > 0:
                    integ = max(-0.5, min(0.5, integ + a * DT))         # anti-windup
                da = 0.0 if prev_a is None else (a - prev_a) / DT
                d_filt = 0.7 * d_filt + 0.3 * da                       # filtered derivative
                prev_a = a
                delta = max(-DMAX, min(DMAX, kp * a + ki * integ + kd * d_filt))
        if v > 0 and delta is not None:
            dw = max(-DMAX, min(DMAX, delta)) + math.radians(trim_deg)  # servo trim error
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        yaw += v * math.tan(dw) / L * DT
        vis.append(visible); bear.append(abs(b))
    bb = np.degrees(np.array(bear)[np.array(vis)])
    return 100 * np.mean(vis), float(np.mean(bb)) if bb.size else 99, float(np.percentile(bb, 95)) if bb.size else 99

scen = {"walk + 90deg turn": S.straight_then_turn, "side step 1 m": S.side_step, "weaving walk": S.zigzag,
        "straight walk away": lambda t: (2.0 + 0.04 * t, 0.0)}
laws = {"pure pursuit (current)": ("pp", (0, 0, 0)), "P   (kp=1.0)": ("pid", (1.0, 0, 0)),
        "PD  (kp=1.0, kd=0.15)": ("pid", (1.0, 0, 0.15)), "PI  (kp=1.0, ki=0.3)": ("pid", (1.0, 0.3, 0)),
        "PID (1.0, 0.3, 0.15)": ("pid", (1.0, 0.3, 0.15))}
for dist_name, dist in (("ideal", dict()), ("servo trim 3deg + noise 1.5deg + 2-frame delay",
                                                 dict(trim_deg=3.0, noise_deg=1.5))):
    print(f"\n######## disturbances: {dist_name}   (visible %, mean |bearing| deg, 95th pct |bearing| deg)")
    for sname, path in scen.items():
        print(f"== {sname}")
        for lname, (law, g) in laws.items():
            vis, mb, p95 = run(path, law, gains=g, **dist)
            print(f"   {lname:24s} {vis:6.1f} %  {mb:5.1f}  {p95:5.1f}")
