#!/usr/bin/env python3
"""
verify_learning_patch.py
------------------------
Usage (in the package directory):
    python3 verify_learning_patch.py multicue_owner.py.bak multicue_owner.py.new

Test 1  LEGACY EQUIVALENCE: original and patched run the same 1500-step random
        session (enroll, score, gallery updates, lock changes, occlusions).
        Every returned dict and every profile field must be IDENTICAL.
Test 2  SHADOW EQUIVALENCE: leg cue computed but not voting — decisions identical.
Test 3  EXTENDED: enrollment refuses to finish before 120 s, finishes after.
Test 4  EXTENDED: 5-min active learning at 0.2 Hz, then 0.05 Hz drift phase.
Test 5  LEG CUE: owner at a different distance still matches (distance-invariance);
        different leg shape scores lower; clipped legs abstain.
No ROS, no camera, no GPU needed. Writes nothing.
"""
import importlib.util
import os
import sys
import time
import numpy as np

orig_path, new_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(new_path)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path,
            loader=importlib.machinery.SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


for k in ("A2RBOT_LEARNING", "A2RBOT_LEG_CUE"):
    os.environ.pop(k, None)
O = load(orig_path, "mc_orig")
N = load(new_path, "mc_new")
import a2rbot_learning_patch as L


def fake_reid(frame, bbox):            # deterministic, depends on crop content
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    rng = np.random.default_rng(int(crop.mean() * 1000) % (2**31))
    e = rng.standard_normal(512).astype(np.float32); return e / np.linalg.norm(e)


def fake_face_det(frame, bbox):
    return None if int(bbox[0]) % 3 else np.zeros((80, 80, 3), np.uint8)


def fake_face_emb(crop):
    e = np.ones(128, np.float32); return e / np.linalg.norm(e)


def session(m, seed, pass_depth):
    rng = np.random.default_rng(seed)
    outs = []
    def frame_and_bbox():
        f = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        x1 = int(rng.integers(0, 400)); y1 = int(rng.integers(0, 150))
        b = (x1, y1, x1 + int(rng.integers(60, 230)), y1 + int(rng.integers(150, 320)))
        d = rng.integers(500, 4000, (480, 640)).astype(np.uint16)
        return f, b, d
    for _ in range(20):
        f, b, d = frame_and_bbox()
        kw = {"depth_frame": d} if pass_depth else {}
        m.enroll(f, b, float(rng.uniform(0.8, 3.5)), **kw)
    outs.append(m.finalize_enrollment())
    for i in range(1500):
        f, b, d = frame_and_bbox()
        dist = None if i % 7 == 0 else float(rng.uniform(0.5, 6))
        others = [b, (b[0] + 20, b[1], b[2] + 20, b[3])] if i % 5 == 0 else None
        kw = {"depth_frame": d} if pass_depth else {}
        r = m.score(f, b, dist, reid_emb=fake_reid(f, b) if i % 2 else None,
                    depth_roi=d[b[1]:b[3], b[0]:b[2]], other_dets=others, **kw)
        r = dict(r); r.pop("legs", None)
        outs.append(r)
        if i % 11 == 0:
            m.set_owned(bool(i % 2))
        if i % 3 == 0:
            res = dict(r); res["score"] = 0.9 if i % 6 == 0 else r["score"]
            m.update_galleries(f, b, res, **kw)
    p = m.profile
    outs.append([np.round(p.reid, 6).tolist(), np.round(p.color, 6).tolist(),
                 len(p.reid_gallery), len(p.color_gallery), p.height_ratio,
                 p.enroll_distance, p.enroll_bbox_h, p.body_width_ratio])
    return outs


ok = True
def check(name, cond, info=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")
    ok &= bool(cond)

# ── Test 1 ────────────────────────────────────────────────────────────────
for faces in (False, True):
    fk = dict(face_detector=fake_face_det, face_embed_fn=fake_face_emb) if faces else {}
    a = session(O.MultiCueMatcher(reid_fn=fake_reid, **fk), 7, False)
    b = session(N.MultiCueMatcher(reid_fn=fake_reid, **fk), 7, False)
    diff = sum(x != y for x, y in zip(a, b))
    check(f"1 legacy equivalence (faces={faces})", a == b and len(a) == len(b),
          f"- {len(a)} outputs, {diff} differ")

# ── Test 2 ────────────────────────────────────────────────────────────────
a = session(O.MultiCueMatcher(reid_fn=fake_reid), 11, False)
b = session(N.MultiCueMatcher(reid_fn=fake_reid, leg_mode="shadow"), 11, True)
check("2 shadow mode leaves every decision unchanged", a == b)

# ── Test 3 ────────────────────────────────────────────────────────────────
m = N.MultiCueMatcher(reid_fn=fake_reid, learning_mode="extended")
f = np.full((480, 640, 3), 120, np.uint8); bb = (200, 60, 330, 420)
m.start_enrollment()
for _ in range(800):
    m.enroll(f, bb, 1.6)
check("3a not finished at t~0 s with 800 frames", m.finalize_enrollment() is False
      and not m.profile.enrolled)
m._enrollment._start_ts -= 121          # simulate 121 s elapsed
check("3b finished after 120 s", m.finalize_enrollment() is True and m.profile.enrolled,
      f"- gallery seeded with {len(m.profile.reid_gallery)}")
m2 = N.MultiCueMatcher(reid_fn=fake_reid, learning_mode="extended")
m2.enroll(f, bb, 1.6)                   # caller forgot start_enrollment()
check("3c auto-starts if start_enrollment() not called", m2._enrollment.started)
for _ in range(5):
    m2.enroll(f, bb, 9.0)               # out of range
check("3d out-of-range depth frames rejected", m2._enrollment.n_accepted == 1)

# ── Test 4 ────────────────────────────────────────────────────────────────
lr = L.ContinuousLearner()
t = [0.0]; real = time.monotonic
L.time.monotonic = lambda: t[0]
ups_active = 0
while lr.learned_s < 300:
    lr.observe(); ups_active += lr.should_update(0.9); t[0] += 1 / 15
ups_drift = 0
for _ in range(15 * 300):
    lr.observe(); ups_drift += lr.should_update(0.9); t[0] += 1 / 15
t[0] += 50                               # owner lost 50 s
before = lr.learned_s; lr.observe()
L.time.monotonic = real
check("4a 300 s active phase", ups_active in (60, 61), f"- {ups_active} updates (expect 60)")
check("4b drift phase continues learning", ups_drift in (15, 16), f"- {ups_drift} in next 300 s (expect 15)")
check("4c lost time not counted as learning", abs(lr.learned_s - before) < 1e-9)
check("4d score < 0.82 never learns", not L.ContinuousLearner().should_update(0.81))

# ── Test 5 ────────────────────────────────────────────────────────────────
def person(z_mm, bbox, leg_gap=0.3, W=640, H=480):
    d = np.full((H, W), 6000, np.uint16)               # background 6 m
    x1, y1, x2, y2 = bbox; w = x2 - x1
    d[y1:int(y1 + 0.55 * (y2 - y1)), x1:x2] = z_mm      # torso
    g0 = x1 + int(w * (0.5 - leg_gap / 2)); g1 = x1 + int(w * (0.5 + leg_gap / 2))
    d[int(y1 + 0.55 * (y2 - y1)):y2, x1:g0] = z_mm      # left leg
    d[int(y1 + 0.55 * (y2 - y1)):y2, g1:x2] = z_mm      # right leg
    return d
anc, _ = L.compute_leg_profile(person(1500, (240, 60, 400, 440)), (240, 60, 400, 440), 1.5)
p2, n2 = L.compute_leg_profile(person(3000, (280, 150, 360, 340)), (280, 150, 360, 340), 3.0)
s_same, _ = L.leg_similarity(p2, anc, n2)
p3, n3 = L.compute_leg_profile(person(1500, (240, 60, 400, 440), 0.0), (240, 60, 400, 440), 1.5)
s_diff, _ = L.leg_similarity(p3, anc, n3)
clip, _ = L.compute_leg_profile(person(1500, (240, 60, 400, 480)), (240, 60, 400, 480), 1.5)
check("5a same legs at 3 m vs 1.5 m anchor", s_same > 0.8, f"- s={s_same:.3f}")
check("5b different leg shape scores lower", s_diff < s_same, f"- s={s_diff:.3f}")
check("5c legs cut off at image bottom -> abstain", clip is None)

print("\nALL PASSED - safe to install" if ok else "\nFAILURES - DO NOT INSTALL")
sys.exit(0 if ok else 1)
