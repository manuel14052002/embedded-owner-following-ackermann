"""leg_signature.py - learn the OWNER'S LEGS from the Pi lidar (/scan), in SHADOW mode.

Used by lidar_owner_keeper.py only when started with  -p leg_signature:=shadow .
It never changes what the keeper decides or publishes: it watches, learns, predicts, and
grades its own predictions against the camera.

LEARN    while the camera confirms the owner on a lidar cluster (keeper source
         "camera+lidar") and no other person track is within 1 m, the owner's leg
         features are learned: per-leg width (calf/shin) and leg separation (stance/stride).
         2 samples/s, a sliding 5-minute memory (600 samples).
PREDICT  when the keeper abstains (ambiguous / merged / waiting_camera), score the owner
         hypothesis and every person track near it against the learned legs
         (Eq.-1-style normalised distance, same form as FollowMe's Re-ID model) and record
         which track the legs say is the owner - or abstain if the margin is small.
GRADE    when the camera next confirms the owner (within 10 s), the prediction is graded
         correct / wrong. These counts decide whether the cue may ever vote.

Every threshold below is [UNK]: chosen, not measured. The CSV this writes is the measurement.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from typing import Optional

import numpy as np

SEL_RADIUS_M = 0.45        # points within this of a track belong to it
MAX_RANGE_M = 6.0          # beyond this, legs are 1-3 beams: no shape information
WINDOW_S = 2.0             # per-track feature window (about one gait cycle and more)
LEARN_HZ = 2.0
MEMORY = 600               # [DER] 600 / 2 Hz = 300 s = 5 min of learned leg views
MIN_MODEL = 20             # samples before the model predicts anything
LEG_MIN_W, LEG_MAX_W = 0.03, 0.25
SIG_FLOOR = np.array([0.010, 0.030])   # m: width, separation  [UNK]
DECIDE_MAX_D = 3.0         # best candidate must be at least this plausible   [UNK]
DECIDE_MARGIN = 0.5        # ... and this much better than the runner-up      [UNK]
GRADE_WITHIN_S = 10.0
ABSTAIN_SOURCES = ("ambiguous", "merged", "waiting_camera")


def leg_features(pts: np.ndarray, ranges: np.ndarray, angle_inc: float) -> Optional[tuple]:
    """Points of ONE person (angle-ordered) -> (mean leg width, leg separation) or None.
    Width is corrected by one beam spacing (an extent of k points spans k-1 gaps)."""
    if len(pts) < 3:
        return None
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    rr = np.maximum(ranges[:-1], ranges[1:])
    cuts = np.flatnonzero(step > np.maximum(0.04, 2.5 * rr * abs(angle_inc))) + 1
    segs = [s for s in np.split(np.arange(len(pts)), cuts) if len(s) >= 2]
    legs = []
    for s in segs:
        q = pts[s]
        w = float(np.linalg.norm(q.max(0) - q.min(0))) + float(np.median(ranges[s])) * abs(angle_inc)
        if LEG_MIN_W <= w <= LEG_MAX_W:
            legs.append((len(s), w, q.mean(0)))
    if not legs:
        return None
    legs.sort(key=lambda x: -x[0])
    legs = legs[:2]
    w = float(np.mean([x[1] for x in legs]))
    sep = float(np.linalg.norm(legs[0][2] - legs[1][2])) if len(legs) == 2 else 0.0
    return w, sep


def window_signature(buf) -> Optional[np.ndarray]:
    if len(buf) < 4:
        return None
    a = np.array([(w, s) for _, w, s in buf])
    return np.array([np.median(a[:, 0]), np.percentile(a[:, 1], 90)])


class LegSignatureShadow:
    def __init__(self, log_dir: str = "~/tracker_logs") -> None:
        self.samples: deque = deque(maxlen=MEMORY)
        self.buf: dict[int, deque] = {}
        self._last_learn = -math.inf
        self._last_next_id = 0
        self.pending: Optional[dict] = None
        self.counts = dict(learned=0, predictions=0, abstained=0, correct=0, wrong=0, ungradable=0)
        d = os.path.expanduser(log_dir)
        os.makedirs(d, exist_ok=True)
        self._fh = open(os.path.join(d, time.strftime("legsig_%Y%m%d_%H%M%S.csv")), "w", newline="")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(["t", "event", "owner_id", "chosen_id", "truth_id", "d_owner",
                            "d_best_rival", "n_model", "mu_w", "mu_sep", "candidates"])

    # -- model -----------------------------------------------------------------------------
    def reset(self) -> None:
        """New owner (ownership changed): forget everything."""
        self.samples.clear()
        self.buf.clear()
        self.pending = None

    @property
    def ready(self) -> bool:
        return len(self.samples) >= MIN_MODEL

    def _model(self):
        a = np.stack(self.samples)
        return a.mean(0), np.maximum(a.std(0), SIG_FLOOR)

    def distance(self, sig: np.ndarray) -> float:
        mu, sd = self._model()
        z = (sig - mu) / sd
        return float(np.sqrt(np.mean(z * z)))

    # -- per scan ----------------------------------------------------------------------------
    def step(self, now: float, pts_laser: np.ndarray, T_tl: np.ndarray, keeper, out,
             angle_inc: float) -> None:
        if keeper._next_id < self._last_next_id:          # keeper.reset(): track ids restart
            self.buf.clear()
            self.pending = None
        self._last_next_id = keeper._next_id
        if len(pts_laser) == 0:
            return
        pw = (T_tl[:2, :2] @ pts_laser.T).T + T_tl[:2, 3]
        rng = np.linalg.norm(pts_laser, axis=1)
        sensor = T_tl[:2, 3]
        live = set()
        for t in keeper.tracks:
            observed = bool(t.hist) and t.hist[-1][0] == now      # really measured, not coasting
            if not observed or t.hits < 3 or float(np.linalg.norm(t.pos - sensor)) > MAX_RANGE_M:
                continue
            m = np.linalg.norm(pw - t.pos, axis=1) <= SEL_RADIUS_M
            f = leg_features(pw[m], rng[m], angle_inc)
            live.add(t.id)
            if f is None:
                continue
            b = self.buf.setdefault(t.id, deque())
            b.append((now, f[0], f[1]))
            while b and now - b[0][0] > WINDOW_S:
                b.popleft()
        for tid in [k for k in self.buf if k not in live and not any(t.id == k for t in keeper.tracks)]:
            del self.buf[tid]

        o = keeper.owner
        src = out.source if out is not None else ""
        # LEARN
        if o is not None and src == "camera+lidar" and now - self._last_learn >= 1.0 / LEARN_HZ:
            alone = all(float(np.linalg.norm(t.pos - o.pos)) > 1.0 for t in keeper.tracks
                        if t is not o and t.person and not keeper._frag(t) and now - t.t_seen <= 0.3)
            sig = window_signature(self.buf.get(o.id, ()))
            if alone and sig is not None:
                self.samples.append(sig)
                self._last_learn = now
                self.counts["learned"] += 1
        # GRADE
        if self.pending is not None:
            if src.startswith("camera") and o is not None:
                p, truth = self.pending, o.id
                if p["chosen"] is None:
                    ev = "abstained"
                elif truth not in p["cands"]:
                    ev = "ungradable"
                else:
                    ev = "correct" if p["chosen"] == truth else "wrong"
                self.counts[ev] += 1
                self._log(now, ev, p["owner"], p["chosen"], truth, p["d_owner"], p["d_rival"], p["cands"])
                self.pending = None
            elif now - self.pending["t"] > GRADE_WITHIN_S:
                self.counts["ungradable"] += 1
                self.pending = None
        # PREDICT
        if self.pending is None and o is not None and src in ABSTAIN_SOURCES and self.ready:
            cands = {}
            for t in keeper.tracks:
                if (t is o or (t.person and not keeper._frag(t)
                               and float(np.linalg.norm(t.pos - o.pos)) <= 1.5)):
                    sig = window_signature(self.buf.get(t.id, ()))
                    if sig is not None:
                        cands[t.id] = self.distance(sig)
            if len(cands) >= 2:
                order = sorted(cands, key=cands.get)
                best, second = order[0], order[1]
                ok = cands[best] <= DECIDE_MAX_D and cands[second] - cands[best] >= DECIDE_MARGIN
                rivals = [v for k, v in cands.items() if k != o.id]
                self.pending = dict(t=now, owner=o.id, chosen=best if ok else None, cands=list(cands),
                                    d_owner=cands.get(o.id, float("nan")),
                                    d_rival=min(rivals) if rivals else float("nan"))
                self.counts["predictions"] += 1

    def _log(self, now, ev, owner, chosen, truth, d_o, d_r, cands) -> None:
        mu = self._model()[0] if self.ready else (float("nan"), float("nan"))
        self._csv.writerow([f"{now:.3f}", ev, owner, chosen, truth, f"{d_o:.3f}", f"{d_r:.3f}",
                            len(self.samples), f"{mu[0]:.4f}", f"{mu[1]:.4f}",
                            " ".join(map(str, cands))])
        self._fh.flush()

    def status(self) -> str:
        mu = self._model()[0] if self.ready else None
        legs = f"w={mu[0]*100:.1f}cm sep={mu[1]*100:.0f}cm" if mu is not None else "learning"
        c = self.counts
        return (f"legsig[{legs} n={len(self.samples)}] pred={c['predictions']} ok={c['correct']} "
                f"wrong={c['wrong']} abst={c['abstained']} ungr={c['ungradable']}")
