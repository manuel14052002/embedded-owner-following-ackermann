#!/usr/bin/env python3
"""Lidar owner keeper: the lidar holds the owner for the whole time the robot is owned.

Runs on the Jetson next to FollowMe (ROS 2 Humble, ROS_DOMAIN_ID shared with the Pi).

    camera (FollowMe /followme/owner_camera)  --anchor-->  lidar leg cluster  --track-->
    /owner_fused_position  ->  FollowMe lidar-assist  ->  /followme/owner_position  ->  Pi

Rules
  * camera sees the owner      -> the lidar cluster next to the camera point IS the owner
                                  (camera always wins: a drifted lidar track is corrected)
  * camera does not            -> follow that cluster with the lidar alone, NO time limit
                                  while the robot is OWNED, up to the lidar's full range
  * two clusters equally close -> abstain (publish nothing); short crossings are resolved
                                  by motion, overlaps longer than ambiguity_max_s need the camera
  * camera veto                -> the tracked cluster is inside the camera view, the camera
                                  sees a person there but not the owner for veto_s -> drop
  * ownership released / new claim -> track reset

Camera <-> lidar alignment
  python3 lidar_owner_keeper.py --calibrate
  Robot NOT owned and standing still, exactly ONE person in the camera view walking slowly
  at 1-4 m across the image. Fits the 2-D offset (yaw, x, y) between where the camera puts
  that person and where the lidar sees them, saves it to align_file; the keeper applies it.
  While running, the status topic reports the live alignment error (rms, range and bearing bias).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

DEFAULT_ALIGN_FILE = os.path.expanduser("~/.ros/a2rbot_cam_lidar_align.yaml")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # owner_path_planner.py sits beside this file


# =========================================================================== core (no ROS)

def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def scan_to_points(ranges: Sequence[float], angle_min: float, angle_increment: float,
                   range_min: float, range_max: float, min_range_m: float = 0.45
                   ) -> tuple[np.ndarray, np.ndarray]:
    """LaserScan -> (N, 2) points and their ranges, angle-ordered. Points inside
    ``min_range_m`` (the robot's own body) and invalid returns are dropped."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= max(range_min, min_range_m)) & (r <= range_max)
    r, a = r[ok], a[ok]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1), r


@dataclass
class Cluster:
    centroid: np.ndarray        # (2,)
    width: float
    n: int
    kind: str = "person"        # person | post | furniture | other  (only "person" can be the owner)

    @property
    def person(self) -> bool:
        return self.kind == "person"


def cluster_points(pts: np.ndarray, ranges: np.ndarray, angle_increment: float,
                   gap_min: float = 0.12, gap_beams: float = 3.0, gap_max: float = 0.35,
                   merge: float = 0.45, leg_width: float = 0.25, leg_pair: float = 0.6,
                   post_width: float = 0.06, post_range: float = 4.0,
                   min_pts_near: int = 3, min_pts_far: int = 2,
                   far_m: float = 4.0, min_width: float = 0.04, max_width: float = 1.0,
                   chair_leg_max: float = 0.08, chair_span: float = 0.75, chair_min_legs: int = 3,
                   blob_min: float = 0.18, blob_max: float = 0.55) -> list[Cluster]:
    """Clusters at any range, each labelled by SHAPE (only "person" can become the owner).

    Neighbouring points belong together if closer than a gap that grows with range
    (``gap_beams`` beam spacings), so far-away legs (few, widely spaced points) still form
    a cluster. A LEG is a segment ``post_width``..``leg_width`` wide; two legs up to
    ``leg_pair`` apart are one person (a walking stride spreads them ~0.5 m); a person can
    also be one blob ``blob_min``..``blob_max`` wide (legs together, long coat). Thinner
    segments are POSTS (chair and table legs) and never pair with anything. FURNITURE:
    ``chair_min_legs`` or more thin segments (< ``chair_leg_max``) within ``chair_span``
    (a 40 x 50 cm chair, a table end) - a group made only of such segments is not a person,
    even if two of its legs are thick enough to look like a pair. Far clusters may have
    only ``min_pts_far`` points."""
    if len(pts) < 2:
        return []
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    rr = np.maximum(ranges[:-1], ranges[1:])
    thr = np.clip(gap_beams * rr * abs(angle_increment), gap_min, gap_max)
    cuts = np.flatnonzero(step > thr) + 1
    idx = np.split(np.arange(len(pts)), cuts)
    segs = [i for i in idx if len(i) >= 1]
    # the scan wraps around: join last and first segment if they touch
    if len(segs) > 1:
        a, b = pts[segs[-1][-1]], pts[segs[0][0]]
        if np.linalg.norm(a - b) <= np.clip(gap_beams * max(ranges[segs[-1][-1]], ranges[segs[0][0]])
                                            * abs(angle_increment), gap_min, gap_max):
            segs[0] = np.concatenate([segs[-1], segs[0]])
            segs.pop()

    def seg_width(ix: np.ndarray) -> float:
        q = pts[ix]
        return float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))

    widths = [seg_width(s) for s in segs]
    centres = [pts[s].mean(axis=0) for s in segs]
    groups: list[list[int]] = []                 # segment indices per group
    legs: list[int] = []                         # number of leg-sized segments in each group
    posts: list[bool] = []
    for si, s in enumerate(segs):
        c, w = centres[si], widths[si]
        is_post = w < post_width and float(np.median(ranges[s])) < post_range
        is_leg = w <= leg_width and not is_post
        best, best_d = None, math.inf
        for i, g in enumerate(groups):
            if is_post or posts[i]:
                continue                         # furniture stands alone
            if is_leg and legs[i] >= 2:
                continue                         # a person has two legs: a third one is someone else
            gi = np.concatenate([segs[k] for k in g])
            lim = leg_pair if (is_leg and legs[i] == 1 and seg_width(gi) <= leg_width) else merge
            d = float(np.linalg.norm(pts[gi].mean(axis=0) - c))
            if d < lim and d < best_d:
                best, best_d = i, d
        if best is None:
            groups.append([si])
            legs.append(int(is_leg))
            posts.append(is_post)
        else:
            groups[best].append(si)
            legs[best] += int(is_leg)
    # chair / table pattern: several thin legs close together
    thin = [k for k in range(len(segs)) if widths[k] < chair_leg_max]
    chairy = set()
    if len(thin) >= chair_min_legs:
        C = np.array([centres[k] for k in thin])
        D = np.hypot(C[:, None, 0] - C[None, :, 0], C[:, None, 1] - C[None, :, 1])
        for a_i, k in enumerate(thin):
            if int((D[a_i] <= chair_span).sum()) >= chair_min_legs:
                chairy.add(k)
    out = []
    for gi_, g in enumerate(groups):
        ix = np.concatenate([segs[k] for k in g])
        p = pts[ix]
        width = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
        need = min_pts_far if float(np.median(ranges[ix])) >= far_m else min_pts_near
        if not (len(ix) >= need and min_width <= width <= max_width):
            continue
        if posts[gi_]:
            kind = "post"
        elif all(k in chairy for k in g):
            kind = "furniture"
        elif legs[gi_] >= 1 or blob_min <= width <= blob_max:
            kind = "person"
        else:
            kind = "other"
        out.append(Cluster(p.mean(axis=0), width, len(ix), kind))
    return out


@dataclass
class KeeperConfig:
    assoc_gate_m: float = 0.6           # camera point -> lidar cluster ...
    assoc_gate_per_m: float = 0.05      # ... plus this per metre of range
    track_gate_m: float = 0.35          # lidar: cluster near a track's prediction (+ own speed * dt)
    max_speed_mps: float = 1.8
    ambiguity_margin_m: float = 0.25    # an unexplained runner-up this close -> abstain
    ambiguity_max_s: float = 3.0        # held longer than this: wait for the camera
    ambiguity_policy: str = "predict"   # predict: conflicts with furniture resolved by motion,
                                        #          conflicts with people need the camera | camera: always
    merge_radius_m: float = 0.5         # a hidden track this close to the owner's cluster = merged
    merged_width_m: float = 0.8         # owner cluster suddenly this wide = someone joined it
    guard_radius_m: float = 1.0         # another moving person this close: keep tracking, publish nothing
    safe_separation_m: float = 0.7      # ... came closer than this: the camera must re-confirm the owner
    moving_speed_mps: float = 0.3       # a track that walked faster than this in the last 3 s is a
                                        # moving person; still ones (furniture, standing people) are not
    camera_fresh_s: float = 0.4
    coast_publish_s: float = 0.5        # keep publishing the prediction through short misses
    lost_s: float = 2.0                 # no matching cluster this long -> track dropped
    lidar_only_max_s: float = 0.0       # 0 = no limit while the owner is assigned
    veto_s: float = 2.0                 # camera sees a person at the tracked spot, not the owner -> drop
    veto_radius_m: float = 0.5
    fov_half_deg: float = 34.5          # D435 colour HFOV / 2
    fov_margin_deg: float = 5.0
    veto_max_range_m: float = 6.0
    vel_window_s: float = 1.0           # velocity = line fit over this much history (averages the gait)
    always_publish: bool = True         # a2r_always_path: never go silent while the owner track exists -
                                        # someone close / merged / ambiguous: keep following the owner's own
                                        # track (its motion), the camera corrects it when it sees the owner


@dataclass
class KeeperOutput:
    pos: Optional[np.ndarray]           # owner position to publish (track frame) or None
    source: str                         # camera | camera+lidar | lidar | coast | merged | ambiguous |
                                        # encounter | waiting_camera | lost | vetoed | timeout | none
    event: str = ""


@dataclass
class _Track:
    id: int
    pos: np.ndarray
    t_seen: float
    width: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    hist: deque = field(default_factory=lambda: deque(maxlen=40))
    moving_until: float = -math.inf      # walked at > moving_speed within the last 3 s
    parent: Optional[int] = None         # born next to this track and never left it: its other leg
    pscore: float = 1.0                  # running share of person-shaped observations; furniture
                                         # (low pscore) never merges with or contests the owner
    furniture_n: int = 0                 # non-person observations while standing still
    moving_flag: bool = False            # copy of "walked recently", kept by the keeper

    @property
    def person(self) -> bool:
        if self.furniture_n >= 10 and not self.moving_flag:
            return False                 # long-standing furniture stays furniture until it moves
        return self.pscore >= 0.5


class OwnerKeeper:
    """Every lidar cluster gets a short-term track (global nearest neighbour), so a crossing is
    two tracks meeting and separating, not one track sliding onto the wrong person. The camera
    says WHICH track is the owner; the lidar keeps that track while the owner is on their own.
    Close encounters with other people are sat out silently and need the camera afterwards."""

    def __init__(self, cfg: Optional[KeeperConfig] = None) -> None:
        self.cfg = cfg or KeeperConfig()
        self.residuals: deque = deque(maxlen=400)       # (camera_xy, lidar_xy, sensor_xy)
        self._person = np.zeros(0, bool)
        self.counters = {"camera_corrected": 0, "ambiguous": 0, "merged": 0, "encounter": 0,
                         "lost": 0, "vetoed": 0, "timeout": 0, "needed_camera": 0}
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []
        self._next_id = 1
        self.owner_id: Optional[int] = None
        self.t_camera = -math.inf
        self.source = "none"
        self.amb_since: Optional[float] = None
        self.tainted = False
        self.enc_since: Optional[float] = None
        self.enc_min = math.inf
        self.veto_since: Optional[float] = None
        self.need_camera = False
        self._owner_width = 0.35

    # -- read-outs ---------------------------------------------------------------------------
    @property
    def owner(self) -> Optional[_Track]:
        return next((t for t in self.tracks if t.id == self.owner_id), None)

    @property
    def pos(self) -> Optional[np.ndarray]:
        o = self.owner
        return None if o is None else o.pos

    # -- tracks --------------------------------------------------------------------------------
    def _pred(self, t: _Track, now: float) -> tuple[np.ndarray, float]:
        """Constant-velocity prediction; the search radius grows with the track's OWN speed
        (+0.6 m/s for a change of pace), not with the fastest possible walker."""
        h = min(max(0.0, now - t.t_seen), 1.0)
        grow = min(self.cfg.max_speed_mps, float(np.linalg.norm(t.vel)) + 0.6)
        return t.pos + t.vel * h, self.cfg.track_gate_m + grow * h

    def _move(self, t: _Track, p, now: float, width: Optional[float] = None) -> None:
        c = self.cfg
        p = np.asarray(p, np.float64)
        t.hist.append((now, float(p[0]), float(p[1])))
        while t.hist and now - t.hist[0][0] > c.vel_window_s:
            t.hist.popleft()
        if len(t.hist) >= 3 and t.hist[-1][0] - t.hist[0][0] >= 0.3:
            h = np.array(t.hist)
            tt = h[:, 0] - h[:, 0].mean()
            den = float(np.sum(tt * tt))
            if den > 1e-9:
                v = np.array([np.sum(tt * (h[:, 1] - h[:, 1].mean())), np.sum(tt * (h[:, 2] - h[:, 2].mean()))]) / den
                sp = float(np.linalg.norm(v))
                if sp > c.max_speed_mps:
                    v *= c.max_speed_mps / sp
                t.vel = v
                if sp > c.moving_speed_mps and h[-1, 0] - h[0, 0] >= 0.9 * c.vel_window_s and t.hits >= 10:
                    t.moving_until = now + 3.0
        t.pos, t.t_seen = p.copy(), now
        if width is not None:
            t.width = 0.7 * t.width + 0.3 * width
        t.hits += 1

    def _coast(self, t: _Track, pred: np.ndarray, now: float) -> None:
        t.pos, t.t_seen = pred.copy(), now          # keeps its pre-conflict velocity and history

    def _new(self, p, now: float, width: float) -> _Track:
        t = _Track(self._next_id, np.asarray(p, np.float64).copy(), now, width)
        t.hist.append((now, float(t.pos[0]), float(t.pos[1])))
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _associate(self, now: float, cents: np.ndarray, widths: np.ndarray):
        """Greedy global nearest neighbour. Every track except the owner is updated here;
        the owner's match is returned for the owner logic to judge."""
        preds = [self._pred(t, now) for t in self.tracks]
        pairs = []
        pers = self._person
        for i, (p, g) in enumerate(preds):
            if len(cents):
                d = np.linalg.norm(cents - p, axis=1)
                ok = d <= g
                if self.tracks[i].id == self.owner_id:
                    ok &= pers                                      # the owner is never a chair
                pairs += [(float(d[j]), i, j) for j in np.flatnonzero(ok)]
        pairs.sort()
        t_of_c: dict[int, int] = {}
        c_of_t: dict[int, int] = {}
        for _, i, j in pairs:
            if i in c_of_t or j in t_of_c:
                continue
            c_of_t[i], t_of_c[j] = j, i
        merged_into: dict[int, int] = {}            # hidden track -> cluster hiding it
        if t_of_c:
            js = list(t_of_c)
            for i, (p, _) in enumerate(preds):
                if (i not in c_of_t and self.tracks[i].hits >= 5 and self.tracks[i].person
                        and not self._frag(self.tracks[i])):        # a person, not a hidden chair leg
                    d = np.linalg.norm(cents[js] - p, axis=1)
                    k = int(np.argmin(d))
                    if d[k] <= self.cfg.merge_radius_m:
                        merged_into[i] = js[k]
        owner_i = next((i for i, t in enumerate(self.tracks) if t.id == self.owner_id), None)
        j_owner = c_of_t.get(owner_i) if owner_i is not None else None
        for i, j in merged_into.items():            # someone hidden in the owner's blob stays known
            if j == j_owner and i != owner_i:
                self._coast(self.tracks[i], preds[i][0], now)
        for i, j in c_of_t.items():
            if i == owner_i:
                continue
            t = self.tracks[i]
            if t.hits >= 5 and widths[j] > 2.0 * t.width + 0.15:   # it merged with something: hold it
                self._coast(t, preds[i][0], now)
            else:
                self._move(t, cents[j], now, float(widths[j]))
                pj = bool(self._person[j]) if j < len(self._person) else True
                t.pscore = 0.8 * t.pscore + 0.2 * float(pj)
                t.moving_flag = now < t.moving_until
                if not pj and not t.moving_flag:
                    t.furniture_n += 1
        est = [(t.id, preds[i][0]) for i, t in enumerate(self.tracks) if t.hits >= 3]
        for j in range(len(cents)):
            if j not in t_of_c:
                t_of_c[j] = len(self.tracks)
                t = self._new(cents[j], now, float(widths[j]))
                t.pscore = float(self._person[j]) if j < len(self._person) else 1.0
                if est:
                    dd = [float(np.linalg.norm(cents[j] - p)) for _, p in est]
                    k = int(np.argmin(dd))
                    if dd[k] < 0.6:
                        t.parent = est[k][0]                    # a piece of that person (split leg)
        return owner_i, c_of_t, t_of_c, merged_into, preds

    def _frag(self, t: _Track) -> bool:
        """A split-off piece of the owner (their other leg), not somebody else."""
        return self.owner_id is not None and t.parent == self.owner_id

    def _owner_pieces(self, now: float, cents: np.ndarray) -> None:
        """Per-scan shape is noisy: a half-hidden leg can measure like a chair leg. Near the owner's
        prediction, a non-person piece counts as the owner's unless it sits on a track that has been
        furniture for a while (static, >= 10 scans, mostly non-person)."""
        o = self.owner
        if o is None or not len(cents) or self._person.all():
            return
        pred, gate = self._pred(o, now)
        near = np.linalg.norm(cents - pred, axis=1) <= max(gate, 0.45)
        visible = np.flatnonzero(near & self._person)          # the owner's visible leg(s) this scan
        if not len(visible):
            return                                             # owner not seen: nothing is reinterpreted
        furn = [self._pred(t, now)[0] for t in self.tracks
                if t.id != self.owner_id and t.hits >= 10 and not t.person and now >= t.moving_until]
        for k in np.flatnonzero(near & ~self._person):
            beside = np.min(np.linalg.norm(cents[visible] - cents[k], axis=1)) <= 0.6
            if beside and not any(np.linalg.norm(cents[k] - f) < 0.15 for f in furn):
                self._person[k] = True

    def _merge_owner_legs(self, now: float, cents: np.ndarray, widths: np.ndarray):
        """A long stride can split the owner into two leg clusters. Two leg-sized pieces near the
        owner's prediction that no other known person explains are merged back into one."""
        o = self.owner
        if o is None or len(cents) < 2:
            return cents, widths
        pred, gate = self._pred(o, now)
        d = np.linalg.norm(cents - pred, axis=1)
        # pieces near the owner; a thin "post" here is usually the owner's half-hidden other leg
        # (a real chair leg is explained by its own long-standing track below and stays apart)
        cand = [int(k) for k in np.flatnonzero(d <= max(gate, 0.6)) if widths[k] <= 0.45]
        if len(cand) < 2:
            return cents, widths
        others = [self._pred(t, now)[0] for t in self.tracks
                  if t.id != self.owner_id and t.hits >= 5 and not self._frag(t) and now - t.t_seen <= 0.5]
        cand = [k for k in cand if not any(np.linalg.norm(p - cents[k]) < min(self.cfg.track_gate_m, d[k])
                                           for p in others)]
        if len(cand) < 2:
            return cents, widths
        cand.sort(key=lambda k: d[k])
        a, b = cand[0], cand[1]
        if not (self._person[a] or self._person[b]):
            return cents, widths                                # two thin pieces: furniture, not legs
        sep = float(np.linalg.norm(cents[a] - cents[b]))
        span = sep + 0.5 * float(widths[a] + widths[b])
        if sep > 0.7 or span > self.cfg.merged_width_m:
            return cents, widths
        keep = [k for k in range(len(cents)) if k not in (a, b)]
        self._person = np.append(self._person[keep], True)
        return (np.vstack([cents[keep], (0.5 * (cents[a] + cents[b]))[None]]),
                np.append(widths[keep], span))

    def _drop(self, reason: str) -> KeeperOutput:
        self.counters[reason] = self.counters.get(reason, 0) + 1
        self.tracks = [t for t in self.tracks if t.id != self.owner_id]
        self.reset_owner_state()
        self.owner_id = None
        self.source = reason
        return KeeperOutput(None, reason, reason)

    def reset_owner_state(self) -> None:
        self.amb_since, self.tainted = None, False
        self.enc_since, self.enc_min = None, math.inf
        self.veto_since, self.need_camera = None, False

    def in_camera_view(self, p: np.ndarray, cam_pose) -> bool:
        if cam_pose is None:
            return False
        origin, yaw = cam_pose
        d = np.asarray(p, float) - np.asarray(origin, float)
        rng = float(np.linalg.norm(d))
        if not 0.3 <= rng <= self.cfg.veto_max_range_m:
            return False
        off = abs(wrap(math.atan2(d[1], d[0]) - yaw))
        return off <= math.radians(self.cfg.fov_half_deg - self.cfg.fov_margin_deg)

    # -- main -----------------------------------------------------------------------------------
    def update(self, now: float, centroids, camera_xy=None, persons_xy=None, cam_pose=None,
               sensor_xy=(0.0, 0.0), widths=None, person=None) -> KeeperOutput:
        """One lidar scan, all points in the tracking frame.

        centroids  (N, 2) lidar clusters (widths (N,) optional)
        person     (N,) bool: shape says person (only these can be the owner); None = all
        camera_xy  owner point from the camera (fresh only) or None
        persons_xy (M, 2) every person the camera sees (fresh) or None if unknown
        cam_pose   (origin_xy, optical-axis yaw) of the camera, for the veto
        sensor_xy  lidar position (the camera gate grows with range from it)
        """
        n = len(np.asarray(centroids, np.float64).reshape(-1, 2))
        self._person = np.ones(n, bool) if person is None else np.asarray(person, bool).reshape(n)
        out = self._update(now, centroids, camera_xy, persons_xy, cam_pose, sensor_xy, widths)
        by_id = {t.id: t for t in self.tracks}
        for t in self.tracks:
            if t.parent is not None:
                par = by_id.get(t.parent)
                if par is None or float(np.linalg.norm(t.pos - par.pos)) > 0.8:
                    t.parent = None                                 # it walked off: a person of its own
        if out.source == "waiting_camera" and out.event == "needed_camera":
            self.counters["needed_camera"] += 1
        self.tracks = [t for t in self.tracks if t.id == self.owner_id or now - t.t_seen <= self.cfg.lost_s]
        return out

    def _update(self, now, centroids, camera_xy, persons_xy, cam_pose, sensor_xy, widths) -> KeeperOutput:
        c = self.cfg
        cents = np.asarray(centroids, np.float64).reshape(-1, 2)
        widths = np.full(len(cents), 0.3) if widths is None else np.asarray(widths, np.float64)
        sensor = np.asarray(sensor_xy, np.float64)
        self._owner_pieces(now, cents)
        cents, widths = self._merge_owner_legs(now, cents, widths)
        owner_i, c_of_t, t_of_c, merged_into, preds = self._associate(now, cents, widths)

        # ---- camera sees the owner: it decides which track is the owner
        if camera_xy is not None:
            cam = np.asarray(camera_xy, np.float64)
            gate = c.assoc_gate_m + c.assoc_gate_per_m * float(np.linalg.norm(cam - sensor))
            event, j = "", None
            if len(cents) and self._person.any():
                d = np.where(self._person, np.linalg.norm(cents - cam, axis=1), np.inf)
                order = np.argsort(d)
                if d[order[0]] <= gate:
                    j = int(order[0])
                    if len(d) == 1 or d[order[1]] > gate:          # unique -> alignment sample
                        self.residuals.append((cam.copy(), cents[j].copy(), sensor.copy()))
            prev = self.owner
            if j is not None:
                tr = self.tracks[t_of_c[j]]
                if prev is not None and tr.id != prev.id and self.source in ("lidar", "coast") \
                        and float(np.linalg.norm(prev.pos - cents[j])) > gate:
                    event = "camera_corrected"
                    self.counters[event] += 1
                if prev is not None and tr.id != prev.id:
                    if owner_i is None or owner_i not in c_of_t:   # old hypothesis was only coasting: drop it
                        self.tracks = [t for t in self.tracks if t.id != prev.id]
                    elif float(np.linalg.norm(prev.pos - cents[j])) < 0.8:
                        prev.parent = tr.id                        # it was their other leg
                self.owner_id = tr.id
                for t2 in self.tracks:                             # split pieces of this same person
                    if t2 is not tr and (tr.parent == t2.id or t2.parent == tr.id) \
                            and float(np.linalg.norm(t2.pos - cents[j])) < 0.8:
                        t2.parent = tr.id
                tr.parent = None
                if tr is prev or tr.t_seen < now:                  # not yet moved this scan
                    self._move(tr, cents[j], now, float(widths[j]))
                self._owner_width = tr.width
                self.source = "camera+lidar"
            else:                                                   # the lidar cannot see them now
                if prev is None:
                    prev = self._new(cam, now, self._owner_width)
                    self.owner_id = prev.id
                else:
                    self._move(prev, cam, now)
                self.source = "camera"
            self.t_camera = now
            self.reset_owner_state()
            return KeeperOutput(self.owner.pos.copy(), self.source, event)

        o = self.owner
        if o is None:
            return KeeperOutput(None, self.source if self.source in ("lost", "vetoed", "timeout") else "none")
        if self.need_camera and not c.always_publish:              # a2r_always_path
            o.t_seen = now                                          # keep the hypothesis, say nothing
            return KeeperOutput(None, "waiting_camera")
        if c.lidar_only_max_s > 0 and now - self.t_camera > c.lidar_only_max_s:
            return self._drop("timeout")
        pred, gate = preds[owner_i]
        j = c_of_t.get(owner_i)

        # ---- merged with someone: coast on the owner's own motion, stay silent until they separate
        partners: list = []
        if j is not None:
            partners = [k for k, v in merged_into.items() if v == j and k != owner_i]
        elif owner_i in merged_into:
            partners = [t_of_c[merged_into[owner_i]]]
        widened = j is not None and widths[j] > c.merged_width_m and self._owner_width < c.merged_width_m - 0.1
        if partners or widened:
            return self._hold(o, pred, now, "merged", partners if partners else [None])

        if j is None:
            dt = now - o.t_seen
            if dt > c.lost_s:
                return self._drop("lost")
            self.source = "coast"
            clean = self.enc_since is None and self.amb_since is None and not self.tainted
            return KeeperOutput(pred.copy() if dt <= c.coast_publish_s and (clean or c.always_publish)
                                else None, "coast")                 # a2r_always_path

        # ---- another known person expects to be about where the owner's cluster is: abstain
        d = np.linalg.norm(cents - pred, axis=1)
        for k, t in enumerate(self.tracks):
            if k == owner_i or t.hits < 5 or not t.person or self._frag(t) or now - t.t_seen > 0.5:
                continue
            dk = float(np.linalg.norm(preds[k][0] - cents[j])) if k < len(preds) else math.inf
            if dk < c.track_gate_m and dk < d[j] + c.ambiguity_margin_m:
                return self._hold(o, pred, now, "ambiguous", [k])
        # ---- an unexplained cluster almost as close as the owner's (often its own other leg)
        explained = {cj for ti, cj in c_of_t.items()
                     if ti != owner_i and (self.tracks[ti].hits >= 3 or self._frag(self.tracks[ti]))}
        explained |= {cj for cj, ti in t_of_c.items() if ti < len(self.tracks) and self._frag(self.tracks[ti])}
        rivals = [k for k in range(len(cents)) if k != j and k not in explained and d[k] <= gate
                  and self._person[k]]
        if rivals and min(d[k] for k in rivals) - d[j] < c.ambiguity_margin_m:
            return self._hold(o, pred, now, "ambiguous", [])

        # ---- clean match
        if self.amb_since is not None:
            self.amb_since = None
            if self.tainted or c.ambiguity_policy == "camera":
                self.tainted, self.need_camera = False, True
                return KeeperOutput(None, "waiting_camera", "needed_camera")
        self._move(o, cents[j], now, float(widths[j]))
        self._owner_width = 0.9 * self._owner_width + 0.1 * float(widths[j])

        # ---- another moving person close by: keep tracking silently
        near = [float(np.linalg.norm(t.pos - o.pos)) for t in self.tracks
                if t is not o and not self._frag(t) and now < t.moving_until and t.hits >= 3
                and now - t.t_seen <= 0.3]
        near = [x for x in near if x < c.guard_radius_m]
        if near:
            event = ""
            if self.enc_since is None:
                self.enc_since, event = now, "encounter"
                self.counters["encounter"] += 1
            self.enc_min = min(self.enc_min, min(near))
            self.source = "encounter"
            return KeeperOutput(o.pos.copy() if c.always_publish else None, "encounter", event)  # a2r_always_path
        if self.enc_since is not None:
            close = self.enc_min < c.safe_separation_m
            self.enc_since, self.enc_min = None, math.inf
            if close:
                self.need_camera = True
                return KeeperOutput(None, "waiting_camera", "needed_camera")
        self.source = "lidar"

        # ---- camera veto: it can see the tracked spot, sees a person, but not the owner
        if persons_xy is not None and self.in_camera_view(o.pos, cam_pose):
            pp = np.asarray(persons_xy, np.float64).reshape(-1, 2)
            seen = len(pp) > 0 and float(np.min(np.linalg.norm(pp - o.pos, axis=1))) <= c.veto_radius_m
            if seen:
                if self.veto_since is None:
                    self.veto_since = now
                if now - self.veto_since >= c.veto_s:
                    return self._drop("vetoed")
            else:
                self.veto_since = None
        else:
            self.veto_since = None
        return KeeperOutput(o.pos.copy(), "lidar")

    def _hold(self, o: _Track, pred: np.ndarray, now: float, why: str, partners: list) -> KeeperOutput:
        """Coast on the owner's own (pre-conflict) motion and publish nothing. A conflict with a
        moving person (or an unknown blob) taints the track: the camera must re-confirm."""
        event = ""
        if self.amb_since is None:
            self.amb_since, event = now, why
            self.counters[why] += 1
        if any(k is None or now < self.tracks[k].moving_until for k in partners):
            self.tainted = True
        if now - self.amb_since > self.cfg.ambiguity_max_s:
            self.need_camera = True
        self._coast(o, pred, now)
        self.source = why
        return KeeperOutput(pred.copy() if self.cfg.always_publish else None, why, event)  # a2r_always_path

    # -- alignment monitor ------------------------------------------------------------------------
    def alignment_stats(self) -> dict:
        """Live camera->lidar error from unique camera+lidar matches."""
        if not self.residuals:
            return {"n": 0}
        cam = np.array([r[0] for r in self.residuals])
        lid = np.array([r[1] for r in self.residuals])
        sen = np.array([r[2] for r in self.residuals])
        res = lid - cam
        rc, rl = np.linalg.norm(cam - sen, axis=1), np.linalg.norm(lid - sen, axis=1)
        bc = np.arctan2(cam[:, 1] - sen[:, 1], cam[:, 0] - sen[:, 0])
        bl = np.arctan2(lid[:, 1] - sen[:, 1], lid[:, 0] - sen[:, 0])
        db = np.array([wrap(x) for x in (bl - bc)])
        return {"n": len(res), "rms_m": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
                "range_bias_m": float(np.mean(rl - rc)), "bearing_bias_deg": float(np.degrees(np.mean(db)))}


# ---------------------------------------------------------------- robot motion (no odometry)

def _kabsch_T(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rigid transform mapping points a -> b (least squares)."""
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    x, y = a - ac, b - bc
    yaw = math.atan2(float(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])),
                     float(np.sum(x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1])))
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:2, 2] = bc - T[:2, :2] @ ac
    return T


class ScanOdometry:
    """Robot motion from the lidar alone (point-to-line ICP against a keyframe), for robots
    without an odom TF. Tracks are kept in this slowly drifting 'world' frame, so furniture
    stays still while the robot drives and turns, and only people move."""

    def __init__(self, max_pts: int = 800, iters: int = 20, max_step_m: float = 0.15,
                 max_step_rad: float = 0.35, key_dist_m: float = 0.3, key_rot_rad: float = 0.2) -> None:
        self.max_pts, self.iters = max_pts, iters
        self.max_step_m, self.max_step_rad = max_step_m, max_step_rad
        self.key_dist_m, self.key_rot_rad = key_dist_m, key_rot_rad
        self.pose = np.eye(3)                 # world <- current laser
        self.key_pose = np.eye(3)             # world <- keyframe laser
        self.rel = np.eye(3)                  # keyframe <- current (last estimate)
        self.delta = np.eye(3)                # last scan-to-scan motion (constant-velocity guess)
        self._key = None                      # keyframe points, normals, kd-tree
        self.failures = 0

    @staticmethod
    def _normals(pts: np.ndarray, tree) -> tuple[np.ndarray, np.ndarray]:
        _, nb = tree.query(pts, k=min(6, len(pts)))
        q = pts[nb] - pts[nb].mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", q, q)
        w, v = np.linalg.eigh(cov)
        line = w[:, 0] < 0.15 * np.maximum(w[:, 1], 1e-12)       # neighbours lie on a line
        return v[:, :, 0], line

    def _set_key(self, pts: np.ndarray) -> None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        n, line = self._normals(pts, tree)
        self._key = (pts, n, line, tree)
        self.key_pose = self.pose.copy()
        self.rel = np.eye(3)

    def update(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if len(pts) > self.max_pts:
            pts = pts[np.linspace(0, len(pts) - 1, self.max_pts).astype(int)]
        if len(pts) < 30:
            return self.pose
        if self._key is None:
            self._set_key(pts)
            return self.pose
        kp, kn, kline, tree = self._key
        T = self.rel @ self.delta             # guess: keyframe <- current
        ok = False
        for _ in range(self.iters):
            q = pts @ T[:2, :2].T + T[:2, 2]
            d, idx = tree.query(q)
            thr = min(0.5, max(0.08, 3.0 * float(np.median(d))))
            m = (d < thr) & kline[idx]
            if m.sum() < 30:
                ok = False
                break
            n = kn[idx[m]]
            r = np.sum(n * (q[m] - kp[idx[m]]), axis=1)          # point-to-line distances
            J = np.column_stack([n[:, 0], n[:, 1], n[:, 1] * q[m, 0] - n[:, 0] * q[m, 1]])
            wgt = 1.0 / (1.0 + (r / 0.05) ** 2)                  # Cauchy: people and clutter count less
            A = J.T @ (J * wgt[:, None])
            b = -(J.T @ (r * wgt))
            try:
                dx, dy, da = np.linalg.solve(A + 1e-9 * np.eye(3), b)
            except np.linalg.LinAlgError:
                ok = False
                break
            c, s_ = math.cos(da), math.sin(da)
            dT = np.array([[c, -s_, dx], [s_, c, dy], [0.0, 0.0, 1.0]])
            T = dT @ T
            ok = True
            if math.hypot(dx, dy) < 1e-5 and abs(da) < 1e-5:
                break
        step = np.linalg.inv(self.rel) @ T                       # previous scan <- current
        if not ok or math.hypot(*step[:2, 2]) > self.max_step_m or \
                abs(math.atan2(step[1, 0], step[0, 0])) > self.max_step_rad:
            step = self.delta.copy()                             # implausible fit: motion continued
            T = self.rel @ step
            self.failures += 1
        self.delta, self.rel = step, T
        self.pose = self.key_pose @ T
        if math.hypot(*T[:2, 2]) > self.key_dist_m or abs(math.atan2(T[1, 0], T[0, 0])) > self.key_rot_rad:
            self._set_key(pts)
        return self.pose


# ---------------------------------------------------------------- camera <-> lidar alignment

@dataclass
class Alignment:
    """2-D rigid correction applied to camera points in the LIDAR frame: p' = R(yaw) p + t."""
    yaw: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    meta: dict = field(default_factory=dict)

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float64)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = p[..., 0], p[..., 1]
        out = p.copy()
        out[..., 0] = c * x - s * y + self.tx
        out[..., 1] = s * x + c * y + self.ty
        return out

    def save(self, path: str) -> None:
        import yaml
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"yaw_deg": round(math.degrees(self.yaw), 4), "tx_m": round(self.tx, 4),
                            "ty_m": round(self.ty, 4), **self.meta}, f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Alignment":
        import yaml
        if not path or not os.path.isfile(path):
            return cls()
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(math.radians(float(d.get("yaw_deg", 0.0))), float(d.get("tx_m", 0.0)),
                   float(d.get("ty_m", 0.0)), {k: v for k, v in d.items() if k not in ("yaw_deg", "tx_m", "ty_m")})


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray, iters: int = 4, trim: float = 2.5
                 ) -> tuple[Alignment, np.ndarray, float, float]:
    """Least-squares yaw + translation mapping src -> dst, with outlier trimming.
    Returns (alignment, inlier mask, rms before, rms after)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 3:
        raise ValueError("need at least 3 pairs")
    keep = np.ones(len(src), bool)
    rms_before = float(np.sqrt(np.mean(np.sum((dst - src) ** 2, axis=1))))
    al = Alignment()
    for _ in range(iters):
        s, d = src[keep], dst[keep]
        sc, dc = s.mean(axis=0), d.mean(axis=0)
        a, b = s - sc, d - dc
        yaw = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                         float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
        c_, s_ = math.cos(yaw), math.sin(yaw)
        t = dc - np.array([c_ * sc[0] - s_ * sc[1], s_ * sc[0] + c_ * sc[1]])
        al = Alignment(yaw, float(t[0]), float(t[1]))
        r = np.linalg.norm(al.apply(src) - dst, axis=1)
        new = r <= max(trim * float(np.median(r[keep])), 0.05)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    rms_after = float(np.sqrt(np.mean(np.linalg.norm(al.apply(src[keep]) - dst[keep], axis=1) ** 2)))
    return al, keep, rms_before, rms_after


def spread_ok(pts: np.ndarray, sensor=(0.0, 0.0), min_bearing_deg: float = 20.0,
              min_range_m: float = 1.0) -> tuple[bool, float, float]:
    """Calibration needs people at different bearings and distances."""
    d = np.asarray(pts, np.float64) - np.asarray(sensor, np.float64)
    if len(d) < 2:
        return False, 0.0, 0.0
    b = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    r = np.linalg.norm(d, axis=1)
    bs, rs = float(b.max() - b.min()), float(r.max() - r.min())
    return bs >= min_bearing_deg and rs >= min_range_m, bs, rs


def parse_state(text: str) -> Optional[bool]:
    """FollowMe ~/state 'FOLLOW|IDENTIFYING|OWNED' -> owned? (None if unparsable)."""
    parts = (text or "").split("|")
    if len(parts) < 3:
        return None
    owner, mode = parts[2].strip(), parts[1].strip()
    if owner in ("-", ""):
        return mode == "IDENTIFYING"
    return owner == "OWNED"


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_to_matrix(tf) -> np.ndarray:
    r, t = tf.transform.rotation, tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(r.x, r.y, r.z, r.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def apply_T(T: np.ndarray, p) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, np.float64) + T[:3, 3]


# =========================================================================== ROS node

def build_node(calibrate: bool = False, calib_pairs: int = 150, calib_seconds: float = 90.0):
    import rclpy
    from geometry_msgs.msg import Point, PointStamped, PoseArray, PoseStamped
    from nav_msgs.msg import Odometry, Path
    from rcl_interfaces.msg import ParameterDescriptor
    try:
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:                                            # debug markers are optional
        Marker = MarkerArray = None
    from owner_path_planner import GoalManager, OwnerPathPlanner, PlannerConfig
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class KeeperNode(Node):
        def __init__(self) -> None:
            super().__init__("lidar_owner_keeper_calib" if calibrate else "lidar_owner_keeper")
            p = self.declare_parameter
            p("scan_topic", "/scan")
            p("camera_topic", "/followme/owner_position")   # FollowMe: camera detections of the owner
            p("persons_topic", "/followme/persons")
            p("state_topic", "/followme/state")
            p("fused_topic", "/owner_fused_position")
            p("status_topic", "/owner_keeper/status")
            p("ego_motion", "icp")                      # icp: robot motion from the lidar | odom: TF | none
            p("fixed_frame", "odom")                    # only for ego_motion:=odom
            p("align_file", DEFAULT_ALIGN_FILE)
            p("min_range_m", 0.45)                      # robot body
            p("camera_veto", True)
            # ---- global path to the owner (same process: one scan match, one clustering per scan)
            p("path_topic", "/owner/path")             # nav_msgs/Path, lidar frame, look-ahead points
            p("owner_status_topic", "/owner/status")   # JSON: mode, plan, distances, timing, odom check
            p("legs_topic", "/owner/legs")             # MarkerArray: person / post / furniture (debug)
            p("odom_topic", "/odometry/filtered")      # only for the odometry sign check
            p("base_frame", "base_link")
            p("planner_enabled", True)
            dyn0 = ParameterDescriptor(dynamic_typing=True)
            for n_, v_ in (("plan_rate_hz", 7.0), ("lost_goal_s", 10.0), ("standoff_m", 1.2), ("spacing_m", 0.3),
                           ("grid_size_m", 10.0), ("grid_res_m", 0.08), ("r_lethal_m", 0.20),
                           ("post_width_m", 0.06), ("chair_leg_max_m", 0.08), ("leg_pair_m", 0.6),
                           ("leg_width_m", 0.25)):
                p(n_, v_, dyn0)
            kc = KeeperConfig()
            dyn = ParameterDescriptor(dynamic_typing=True)     # accept 0 as well as 0.0 on the CLI
            for name, val in vars(kc).items():
                p(name, val, dyn)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.g = g
            self.keeper = OwnerKeeper(KeeperConfig(**{n: type(v)(g(n)) for n, v in vars(kc).items()}))
            self.align = Alignment.load(g("align_file"))
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=True)
            self.cam_msg, self.cam_t = None, -math.inf
            self.persons_msg, self.persons_t = None, -math.inf
            self.owned: Optional[bool] = None
            self.last_z = 0.0
            self.track_frame: Optional[str] = None
            self.odo = ScanOdometry()
            self._warned: set = set()
            self._last_status = 0.0
            self._last_log = 0.0
            self._t_lidar_only = None
            self.pairs: list = []
            self.t_start = time.monotonic()
            self.shape_kw = dict(post_width=float(g("post_width_m")), chair_leg_max=float(g("chair_leg_max_m")),
                                 leg_pair=float(g("leg_pair_m")), leg_width=float(g("leg_width_m")))
            self.planner = self.goals = None
            if bool(g("planner_enabled")) and not calibrate:
                self.planner = OwnerPathPlanner(PlannerConfig(
                    size_m=float(g("grid_size_m")), res_m=float(g("grid_res_m")),
                    r_lethal_m=float(g("r_lethal_m")), standoff_m=float(g("standoff_m")),
                    spacing_m=float(g("spacing_m"))))
                self.goals = GoalManager(lost_goal_s=float(g("lost_goal_s")))
            self._plan_dt = 1.0 / max(0.5, float(g("plan_rate_hz")))
            self._plan_t = -math.inf
            self._legs_t = -math.inf
            self._base_l = None                         # base_link origin in the lidar frame (static)
            self.odom_yaw, self.odom_t = None, -math.inf
            self._prev_icp = self._prev_odom = None
            self._oc = [0.0, 0.0, 0.0]                  # sum d_icp*d_odom, sum |..|, turned (rad)
            self.odom_verdict = "unknown"
            self.path_pub = self.create_publisher(Path, g("path_topic"), 10)
            self.ostatus_pub = self.create_publisher(String, g("owner_status_topic"), 10)
            self.legs_pub = self.create_publisher(MarkerArray, g("legs_topic"), 1) if MarkerArray else None
            if g("odom_topic"):
                self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 20)
            self.pub = self.create_publisher(PointStamped, g("fused_topic"), 10)
            self.status_pub = self.create_publisher(String, g("status_topic"), 10)
            self.create_subscription(PointStamped, g("camera_topic"), self._on_cam, 10)
            # a2r_idguard: the phone (phone_pdr_node) as a second opinion and a way back to the owner
            p("phone_assist", True)
            p("phone_view_topic", "/phone_owner/view")
            p("phone_status_topic", "/phone_owner/status")
            p("phone_gate_m", 1.0)          # camera owner further than max(this, 3 sigma) from the phone: suspect
            p("phone_sigma_max_m", 1.2)     # phone less sure than this: not used
            p("phone_veto_max_s", 6.0)      # never overrule the camera longer than this in a row
            p("phone_seed_m", 0.9)          # lidar person this close to the phone (nobody else near) = owner
            p("phone_seed_scans", 3)
            p("phone_goal_after_s", 1.5)    # keeper silent this long (and no camera): plan to the phone
            # a2r_reacquire: owner lost by the lidar too (behind the robot, 90 deg blind side, doorway):
            # the one person-shaped lidar cluster that turns up where the owner was heading is the owner
            p("reacquire_s", 12.0)          # look this long after the loss
            p("reacquire_scans", 4)         # same spot this many scans in a row
            p("reacquire_max_m", 1.5)       # search radius grows 0.5 m + 0.5 m/s, up to this
            self._rq, self._rq_n, self._rq_p, self._rq_cnt = None, 0, None, 0
            self._rq_birth = {}               # track id -> where it first appeared
            # a2r_owner_memory: the owner's own route (lidar track, last trail_s) - when the owner is lost
            # the robot follows THAT route to where they disappeared, + a bit in their direction
            p("trail_s", 30.0)
            p("trail_extend_m", 1.2)
            from collections import deque as _dq
            self._trail, self._trail_done = _dq(maxlen=200), False
            self._ph = self._ph_st = self._ph_xy = None
            self._ph_veto_since = None
            self._ph_seed_n, self._ph_seed_p = 0, None
            self._ph_cnt = {"veto": 0, "seed": 0, "goal": 0}
            self._last_pos_t = -math.inf
            self.create_subscription(PointStamped, g("phone_view_topic"),
                                     lambda m: setattr(self, "_ph", (m, time.monotonic())), 5)
            self.create_subscription(String, g("phone_status_topic"), self._on_phone_status, 5)
            if calibrate or g("camera_veto"):           # FollowMe computes every person's position only
                self.create_subscription(PoseArray, g("persons_topic"), self._on_persons, 10)   # if asked
            self.create_subscription(String, g("state_topic"), self._on_state, 10)
            self.create_subscription(LaserScan, g("scan_topic"), self._on_scan, qos_profile_sensor_data)
            a = self.align
            self.get_logger().info(
                f"{'CALIBRATION' if calibrate else 'keeper'} up | scan {g('scan_topic')} camera {g('camera_topic')}"
                f" | alignment yaw={math.degrees(a.yaw):+.2f} deg t=({a.tx:+.3f},{a.ty:+.3f}) m"
                f" | lidar-only limit: {'none' if g('lidar_only_max_s') <= 0 else str(g('lidar_only_max_s')) + ' s'}")

        # ---- inputs
        def _on_cam(self, msg) -> None:
            self.cam_msg, self.cam_t = msg, time.monotonic()

        def _on_odom(self, msg) -> None:
            q = msg.pose.pose.orientation
            self.odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.odom_t = time.monotonic()

        def _odom_check(self, now: float, T_tl: np.ndarray) -> None:
            """Does /odometry/filtered turn the same way as the lidar? (scan matching is the reference)"""
            yi = math.atan2(T_tl[1, 0], T_tl[0, 0])
            fresh = self.odom_yaw is not None and now - self.odom_t < 0.3
            if self._prev_icp is not None and fresh and self._prev_odom is not None:
                di, do = wrap(yi - self._prev_icp), wrap(self.odom_yaw - self._prev_odom)
                if abs(di) > 0.004:
                    self._oc[0] += di * do
                    self._oc[1] += abs(di * do)
                    self._oc[2] += abs(di)
                if self._oc[2] >= 1.0 and self._oc[1] > 1e-9:
                    r = self._oc[0] / self._oc[1]
                    v = "OK" if r > 0.5 else ("INVERTED" if r < -0.5 else "UNCLEAR")
                    if v != self.odom_verdict:
                        log = self.get_logger().warn if v != "OK" else self.get_logger().info
                        if getattr(log, "__name__", "") in ('info',):  # a2r_logfix: one call site per severity (rclpy rule)
                            log.__self__.info(f"odom yaw sign check: {v} (agreement {r:+.2f} over {math.degrees(self._oc[2]):.0f} deg of turning)"
                            + (" - /odometry/filtered turns OPPOSITE to the lidar: check the IMU gyro z sign "
                               "(a2r_bot_sensor_fusion imu_filter) and the EKF steering model sign" if v == "INVERTED" else ""))
                        else:
                            log.__self__.warning(f"odom yaw sign check: {v} (agreement {r:+.2f} over {math.degrees(self._oc[2]):.0f} deg of turning)"
                            + (" - /odometry/filtered turns OPPOSITE to the lidar: check the IMU gyro z sign "
                               "(a2r_bot_sensor_fusion imu_filter) and the EKF steering model sign" if v == "INVERTED" else ""))
                    self.odom_verdict = v
                    self._oc = [x * 0.5 for x in self._oc]
            self._prev_icp = yi
            self._prev_odom = self.odom_yaw if fresh else None

        def _base_in_laser(self, laser: str) -> np.ndarray:
            if self._base_l is None:
                T = self._T(laser, str(self.g("base_frame")))
                if T is None:
                    self._warn_once("base", "no TF lidar <- base_link: assuming base_link 0.14 m ahead of the lidar")
                    return np.array([0.14, 0.0, 0.0])
                self._base_l = T[:3, 3].copy()
            return self._base_l

        def _plan_step(self, now, scan, laser, T_tl, owner_w, owned, source) -> None:
            if self.planner is None or now - self._plan_t < self._plan_dt:
                return
            self._plan_t = now
            goal, mode = self.goals.update(now, owner_w, owned)
            robot = apply_T(T_tl, self._base_in_laser(laser))[:2]
            res = None
            if goal is not None:
                res = self.planner.plan(robot, goal)
                if mode == "LAST_KNOWN" and res.status == "reached":
                    self.goals.last, mode = None, "LOST"           # got there, still nobody: stop
            path = Path()
            path.header.stamp, path.header.frame_id = scan.header.stamp, laser
            Tinv = np.linalg.inv(T_tl)
            if res is not None and res.status in ("ok", "reached") and len(res.points):
                P = (Tinv[:2, :2] @ res.points.T).T + Tinv[:2, 3]
                own = Tinv[:2, :2] @ goal + Tinv[:2, 3]
                for i, q in enumerate(P):
                    nxt = P[i + 1] if i + 1 < len(P) else own
                    yaw = math.atan2(nxt[1] - q[1], nxt[0] - q[0]) if np.hypot(*(nxt - q)) > 1e-3 else 0.0
                    ps = PoseStamped()
                    ps.header.stamp, ps.header.frame_id = scan.header.stamp, laser
                    ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
                    ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                    path.poses.append(ps)
            self.path_pub.publish(path)                    # empty path = no target (never a guess)
            st = {"mode": mode, "source": source, "plan": res.status if res else "none",
                  "points": len(path.poses), "odom_yaw": self.odom_verdict}
            if goal is not None:
                st["owner_m"] = round(float(np.hypot(*(goal - robot))), 2)
            if res is not None:
                st.update(route_m=round(res.owner_route_m, 2), path_m=round(res.length_m, 2), ms=round(res.ms, 1))
            self.ostatus_pub.publish(String(data=json.dumps(st)))

        def _legs(self, now, scan, laser, clusters) -> None:
            if self.legs_pub is None or now - self._legs_t < 0.5 or self.legs_pub.get_subscription_count() == 0:
                return
            self._legs_t = now
            ma = MarkerArray()
            col = {"person": (0.1, 0.9, 0.1), "post": (0.6, 0.6, 0.6), "furniture": (0.9, 0.2, 0.2)}
            for i, c in enumerate(clusters):
                m = Marker()
                m.header.stamp, m.header.frame_id = scan.header.stamp, laser
                m.ns, m.id, m.type, m.action = "legs", i, Marker.CYLINDER, Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(c.centroid[0]), float(c.centroid[1]), 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = max(0.06, float(c.width))
                m.scale.z = 0.3
                m.color.r, m.color.g, m.color.b = col.get(c.kind, (0.3, 0.3, 0.9))
                m.color.a = 0.7
                m.lifetime.nanosec = 600_000_000
                ma.markers.append(m)
            self.legs_pub.publish(ma)

        def _on_persons(self, msg) -> None:
            self.persons_msg, self.persons_t = msg, time.monotonic()

        def _on_state(self, msg) -> None:
            owned = parse_state(msg.data)
            if owned is None:
                return
            if owned != self.owned:
                self.keeper.reset()
                self._t_lidar_only = None
                self.get_logger().info(f"ownership {'assigned' if owned else 'free'}: track reset")
            self.owned = owned

        def _warn_once(self, key: str, text: str) -> None:
            if key not in self._warned:
                self._warned.add(key)
                self.get_logger().warn(text)

        def _T(self, target: str, source: str) -> Optional[np.ndarray]:
            if target == source:
                return np.eye(4)
            try:
                return tf_to_matrix(self.tf.lookup_transform(target, source, rclpy.time.Time()))
            except TransformException as exc:
                self._warn_once(f"{target}<-{source}", f"no TF {target} <- {source}: {exc}")
                return None

        def _cam_points_in_laser(self, frame: str, pts3: np.ndarray, laser: str) -> Optional[np.ndarray]:
            T = self._T(laser, frame)
            if T is None:
                return None
            p = (T[:3, :3] @ np.asarray(pts3, np.float64).reshape(-1, 3).T).T + T[:3, 3]
            p[:, :2] = self.align.apply(p[:, :2])
            return p

        # ---- per scan
        def _on_scan(self, scan) -> None:
            now = time.monotonic()
            laser = scan.header.frame_id
            pts, rng = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                      scan.range_min, scan.range_max, float(self.g("min_range_m")))
            clusters = cluster_points(pts, rng, scan.angle_increment, **self.shape_kw)
            if calibrate:
                self._calib_step(now, laser, clusters)
                return
            # tracking frame: robot motion removed (furniture stays still, people move)
            mode = str(self.g("ego_motion"))
            T_tl = None
            if mode == "icp":
                P = self.odo.update(pts)
                T_tl = np.eye(4)
                T_tl[:2, :2], T_tl[:2, 3] = P[:2, :2], P[:2, 2]
            elif mode == "odom":
                T_tl = self._T(str(self.g("fixed_frame")), laser)
            if T_tl is None:
                T_tl = np.eye(4)
                if mode == "odom":
                    self._warn_once("nofixed", f"no TF to '{self.g('fixed_frame')}': robot motion not compensated")
            key = f"{mode}:{laser}"
            if key != self.track_frame:
                self.keeper.reset()
                self.track_frame = key
            cents = np.array([apply_T(T_tl, [c.centroid[0], c.centroid[1], 0.0])[:2] for c in clusters]) \
                if clusters else np.zeros((0, 2))
            sensor = T_tl[:2, 3]
            pm = np.array([c.person for c in clusters], bool)
            if mode == "icp":
                self._odom_check(now, T_tl)
            if self.planner is not None:                    # obstacle grid: everything except the owner
                pw = (T_tl[:2, :2] @ pts.T).T + T_tl[:2, 3] if len(pts) else np.zeros((0, 2))
                self.planner.update_scan(sensor, pw, self.keeper.pos)
            self._legs(now, scan, laser, clusters)

            owned = self.owned
            if owned is None:
                self._warn_once("nostate", "no /followme/state yet: assuming the robot is owned")
                owned = True
            if not owned:
                self.keeper.reset()
                self._status(now, None, "not_owned")
                self._plan_step(now, scan, laser, T_tl, None, False, "not_owned")
                return

            camera_xy = None
            if self.cam_msg is not None and now - self.cam_t <= self.keeper.cfg.camera_fresh_s:
                m = self.cam_msg
                p = self._cam_points_in_laser(m.header.frame_id, [m.point.x, m.point.y, m.point.z], laser)
                if p is not None:
                    camera_xy, self.last_z = apply_T(T_tl, p[0])[:2], float(p[0][2])   # z: laser frame
            persons_xy, cam_pose = None, None
            if self.g("camera_veto") and self.persons_msg is not None and now - self.persons_t <= 0.4:
                m = self.persons_msg
                p3 = [[ps.position.x, ps.position.y, ps.position.z] for ps in m.poses]
                p = self._cam_points_in_laser(m.header.frame_id, p3, laser) if p3 else np.zeros((0, 3))
                Tlc = self._T(laser, m.header.frame_id)
                if p is not None and Tlc is not None:
                    persons_xy = np.array([apply_T(T_tl, q)[:2] for q in p]) if len(p) else np.zeros((0, 2))
                    o = self.align.apply(Tlc[:2, 3])
                    ax = Tlc[:3, :3] @ np.array([0.0, 0.0, 1.0])           # optical axis
                    yaw = math.atan2(ax[1], ax[0]) + self.align.yaw
                    o3 = apply_T(T_tl, [o[0], o[1], 0.0])
                    cam_pose = (o3[:2], yaw + math.atan2(T_tl[1, 0], T_tl[0, 0]))

            ph_xy, ph_r = self._phone_xy(now, laser, T_tl)                       # a2r_idguard
            camera_xy = self._phone_gate(now, camera_xy, ph_xy, ph_r)
            out = self.keeper.update(now, cents, camera_xy, persons_xy, cam_pose, sensor, person=pm)
            out = self._phone_seed(now, cents, pm, camera_xy, ph_xy, ph_r, out)
            out = self._lidar_reseed(now, cents, pm, camera_xy, out)            # a2r_reacquire
            if out.event:
                self.get_logger().info(f"{out.event} ({self.keeper.counters})")
            if out.source in ("lidar", "coast"):
                self._t_lidar_only = self._t_lidar_only or now
            elif out.source.startswith("camera"):
                self._t_lidar_only = None
            if out.pos is not None:                     # published in the lidar frame: FollowMe has its TF
                q = np.linalg.solve(T_tl, np.array([out.pos[0], out.pos[1], 0.0, 1.0]))
                msg = PointStamped()
                msg.header.stamp = scan.header.stamp
                msg.header.frame_id = laser
                msg.point = Point(x=float(q[0]), y=float(q[1]), z=float(self.last_z))
                self.pub.publish(msg)
            self._status(now, out, out.source, sensor)
            goal_w, goal_src = out.pos, out.source                                # a2r_idguard
            if out.pos is not None:
                self._last_pos_t = now
            elif ph_xy is not None and now - self._last_pos_t >= float(self.g("phone_goal_after_s")) \
                    and now - self.cam_t > 1.0:
                goal_w, goal_src = ph_xy, "phone"
                self._ph_cnt["goal"] += 1
            self._trail_add(now, out)                                             # a2r_owner_memory
            if not (goal_w is None and self._trail_step(now, scan, laser, T_tl)):
                self._plan_step(now, scan, laser, T_tl, goal_w, True, goal_src)

        # ---- a2r_idguard: phone assist -----------------------------------------------------
        def _on_phone_status(self, msg) -> None:
            try:
                self._ph_st = (json.loads(msg.data), time.monotonic())
            except ValueError:
                pass

        def _phone_xy(self, now: float, laser: str, T_tl: np.ndarray):
            """Where the phone puts the owner (tracking frame) and how far to trust it."""
            self._ph_xy = None
            if not bool(self.g("phone_assist")) or self._ph is None or self._ph_st is None:
                return None, None
            m, t = self._ph
            st, ts = self._ph_st
            sigma = float(m.point.z)                    # phone_pdr_node puts its sigma in z
            if now - t > 0.6 or now - ts > 2.0 or not st.get("aligned") or not st.get("phone", True) \
                    or not 0.0 <= sigma <= float(self.g("phone_sigma_max_m")):
                return None, None
            T = self._T(laser, m.header.frame_id or "odom")
            if T is None:
                return None, None
            q = apply_T(T_tl, apply_T(T, [m.point.x, m.point.y, 0.0]))[:2]
            self._ph_xy = q
            return q, max(float(self.g("phone_gate_m")), 3.0 * sigma)

        def _phone_gate(self, now: float, camera_xy, ph_xy, ph_r):
            """The camera puts the owner where the phone says they cannot be (a person crossing
            the owner's path taken for the owner). The camera point is ignored - for at most
            phone_veto_max_s in a row - and a lidar track that went along with the camera is let
            go, so _phone_seed() picks up the person the phone is with."""
            if camera_xy is None or ph_xy is None or float(np.linalg.norm(camera_xy - ph_xy)) <= ph_r:
                self._ph_veto_since = None
                return camera_xy
            if self._ph_veto_since is None:
                self._ph_veto_since = now
                self._ph_cnt["veto"] += 1
                self.get_logger().info(f"phone veto: camera owner point {np.linalg.norm(camera_xy - ph_xy):.1f} m "
                                       f"from the phone - camera ignored (max {self.g('phone_veto_max_s')} s)")
            if now - self._ph_veto_since > float(self.g("phone_veto_max_s")):
                return camera_xy                        # long disagreement: the phone may be wrong
            k = self.keeper
            o = k.owner
            if o is not None and float(np.linalg.norm(o.pos - ph_xy)) > ph_r:
                k.tracks = [t for t in k.tracks if t.id != o.id]
                k.reset_owner_state()
                k.owner_id, k.source = None, "lost"
            return None

        def _trail_add(self, now: float, out) -> None:
            """a2r_owner_memory: remember the owner's route (tracking frame), a point every 0.2 m."""
            if self.owned is False:
                self._trail.clear()
                return
            if out.pos is None or out.source in ("coast", "merged", "ambiguous", "phone"):
                return
            p = np.asarray(out.pos, float)
            if not self._trail or float(np.linalg.norm(p - self._trail[-1][1])) >= 0.2:
                self._trail.append((now, p.copy()))
                self._trail_done = False
            while self._trail and now - self._trail[0][0] > float(self.g("trail_s")):
                self._trail.popleft()

        def _trail_step(self, now: float, scan, laser: str, T_tl) -> bool:
            """Owner lost: path = from the trail point nearest the robot along the owner's route to where
            they vanished, continued trail_extend_m in their last direction. True = trail path published."""
            if self._trail_done or len(self._trail) < 2 or now - self._trail[-1][0] > float(self.g("trail_s")):
                return False
            if now - self._plan_t < self._plan_dt:
                return True
            self._plan_t = now
            robot = apply_T(T_tl, self._base_in_laser(laser))[:2]
            P = np.array([q for _, q in self._trail])
            i = int(np.argmin(np.linalg.norm(P - robot, axis=1)))
            P = P[i:]
            d = P[-1] - P[-2] if len(P) >= 2 else P[-1] - robot
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                P = np.vstack([P, P[-1] + d / n * float(self.g("trail_extend_m"))])
            if float(np.linalg.norm(P[-1] - robot)) < 0.4:
                self._trail_done = True                 # reached it: the Pi's search takes over
                return False
            Tinv = np.linalg.inv(T_tl)
            Q = (Tinv[:2, :2] @ P.T).T + Tinv[:2, 3]
            path = Path()
            path.header.stamp, path.header.frame_id = scan.header.stamp, laser
            for k, q in enumerate(Q):
                nx = Q[k + 1] if k + 1 < len(Q) else q + (q - Q[k - 1] if k else 0)
                yaw = math.atan2(nx[1] - q[1], nx[0] - q[0]) if np.hypot(*(nx - q)) > 1e-3 else 0.0
                ps = PoseStamped()
                ps.header.stamp, ps.header.frame_id = scan.header.stamp, laser
                ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                path.poses.append(ps)
            self.path_pub.publish(path)
            self.ostatus_pub.publish(String(data=json.dumps({"mode": "LAST_KNOWN", "source": "trail",
                "plan": "ok", "points": len(path.poses), "odom_yaw": self.odom_verdict,
                "owner_m": round(float(np.linalg.norm(P[-1] - robot)), 2)})))
            return True

        def _lidar_reseed(self, now: float, cents, pm, camera_xy, out):
            """a2r_reacquire: the lidar lost the owner too (walked behind the robot, round a corner, into
            the lidar's blind side). Remember where they were and where they were going; the ONE
            person-shaped cluster that appears within a radius growing with time around that predicted
            spot (nobody else near it), at the same place reacquire_scans scans in a row, becomes the
            owner track again - no camera needed. The camera confirms (or corrects) when it sees them."""
            k = self.keeper
            o = k.owner
            alive = {t.id for t in k.tracks}
            for t in k.tracks:
                self._rq_birth.setdefault(t.id, t.pos.copy())
            for i in [i for i in self._rq_birth if i not in alive]:
                del self._rq_birth[i]
            if o is not None and out.pos is not None:
                self._rq, self._rq_n = (o.pos.copy(), np.array(o.vel, float).copy(), now), 0
                return out
            if (out.pos is not None or camera_xy is not None or self._rq is None or not len(cents)
                    or out.source not in ("none", "lost", "timeout")):
                self._rq_n = 0
                return out
            p0, v, t0 = self._rq
            dt = now - t0
            if dt > float(self.g("reacquire_s")):
                self._rq_n = 0
                return out
            pred = p0 + v * min(dt, 1.5)
            rad = min(0.5 + 0.5 * dt, float(self.g("reacquire_max_m")))
            pmb = np.asarray(pm, bool)
            d = np.where(pmb, np.linalg.norm(cents - pred, axis=1), np.inf)
            j = int(np.argmin(d))
            sep = np.where(pmb, np.linalg.norm(cents - cents[j], axis=1), np.inf)
            sep[j] = np.inf
            other = np.isfinite(sep) & (sep > 0.4) & ((sep < 1.0) | (d < d[j] + 0.8))   # < 0.4 m: other leg
            if not d[j] <= rad or other.any():
                self._rq_n = 0
                return out
            same = self._rq_p is not None and float(np.linalg.norm(cents[j] - self._rq_p)) <= 0.5
            self._rq_n = self._rq_n + 1 if same else 1
            self._rq_p = cents[j].copy()
            if self._rq_n < int(self.g("reacquire_scans")):
                return out
            live = [t for t in k.tracks if t.t_seen >= now - 1e-6 and t.person]
            tr = min(live, key=lambda t: float(np.linalg.norm(t.pos - cents[j])), default=None)
            if tr is None or float(np.linalg.norm(tr.pos - cents[j])) > 0.35:
                return out
            born = self._rq_birth.get(tr.id, tr.pos)
            if float(np.linalg.norm(born - pred)) > rad:
                self._rq_n = 0                                   # walked in from elsewhere: not the owner
                return out
            k.owner_id = tr.id
            tr.parent = None
            k.reset_owner_state()
            k.source = "lidar"
            self._rq_n, self._rq_cnt = 0, self._rq_cnt + 1
            self.get_logger().info(f"lidar reacquired the owner {d[j]:.2f} m from where they were heading "
                                   f"({dt:.1f} s after losing them) - camera will confirm")
            return KeeperOutput(tr.pos.copy(), "lidar", "lidar_reacquired")

        def _phone_seed(self, now: float, cents, pm, camera_xy, ph_xy, ph_r, out):
            """No owner track (lost, or held after someone came close) and the camera does not see
            the owner: the one lidar person next to the phone position becomes the owner track again
            (phone_seed_scans scans in a row). The lidar path then continues without the camera."""
            k = self.keeper
            if (out.pos is not None or camera_xy is not None or ph_xy is None or not len(cents)
                    or out.source not in ("none", "lost", "vetoed", "timeout", "waiting_camera")):
                self._ph_seed_n = 0
                return out
            d = np.where(np.asarray(pm, bool), np.linalg.norm(cents - ph_xy, axis=1), np.inf)
            order = np.argsort(d)
            j = int(order[0])
            unique = len(order) < 2 or d[order[1]] > d[j] + 0.6
            if not (d[j] <= min(ph_r, float(self.g("phone_seed_m"))) and unique):
                self._ph_seed_n = 0
                return out
            same = self._ph_seed_p is not None and float(np.linalg.norm(cents[j] - self._ph_seed_p)) <= 0.4
            self._ph_seed_n = self._ph_seed_n + 1 if same else 1
            self._ph_seed_p = cents[j].copy()
            if self._ph_seed_n < int(self.g("phone_seed_scans")):
                return out
            live = [t for t in k.tracks if t.t_seen >= now - 1e-6]
            tr = min(live, key=lambda t: float(np.linalg.norm(t.pos - cents[j])), default=None)
            if tr is None or float(np.linalg.norm(tr.pos - cents[j])) > 0.35:
                return out
            if k.owner_id is not None and k.owner_id != tr.id:
                k.tracks = [t for t in k.tracks if t.id != k.owner_id]
            k.owner_id = tr.id
            tr.parent = None
            k.reset_owner_state()
            k.source = "phone+lidar"
            self._ph_seed_n = 0
            self._ph_cnt["seed"] += 1
            self.get_logger().info(f"phone seed: the lidar person {d[j]:.2f} m from the phone is the owner again")
            return KeeperOutput(tr.pos.copy(), "phone+lidar", "phone_seed")

        def _status(self, now: float, out, source: str, sensor=(0.0, 0.0)) -> None:
            if now - self._last_status < 0.5:
                return
            self._last_status = now
            txt = source
            if out is not None and out.pos is not None:
                d = out.pos - np.asarray(sensor)
                txt += f" r={np.linalg.norm(d):.2f}m"
            if self._t_lidar_only:
                txt += f" lidar-only {now - self._t_lidar_only:.0f}s"
            a = self.keeper.alignment_stats()
            if a["n"]:
                txt += (f" | align n={a['n']} rms={a['rms_m']:.2f}m range_bias={a['range_bias_m']:+.2f}m"
                        f" bearing_bias={a['bearing_bias_deg']:+.1f}deg")
            k = self.keeper.counters
            txt += f" | corrected={k['camera_corrected']} ambiguous={k['ambiguous']} vetoed={k['vetoed']} lost={k['lost']}"
            if bool(self.g("phone_assist")):                                     # a2r_idguard
                c = self._ph_cnt
                txt += (f" | phone {'ok' if self._ph_xy is not None else '-'} veto={c['veto']} "
                        f"seed={c['seed']} goal={c['goal']}")
            txt += f" | reacquired={self._rq_cnt}"                                  # a2r_reacquire
            self.status_pub.publish(String(data=txt))
            if now - self._last_log > 10.0:
                self._last_log = now
                self.get_logger().info(txt)

        # ---- calibration
        def _calib_step(self, now: float, laser: str, clusters) -> None:
            if self.owned:
                self._warn_once("calib_owned", "release the robot first: calibration needs it standing still")
                return
            if now - self.t_start > calib_seconds or len(self.pairs) >= calib_pairs:
                self._calib_finish()
                return
            if self.persons_msg is None or now - self.persons_t > 0.3 or len(self.persons_msg.poses) != 1:
                return
            ps = self.persons_msg.poses[0].position
            T = self._T(laser, self.persons_msg.header.frame_id)
            if T is None or not clusters:
                return
            cam = apply_T(T, [ps.x, ps.y, ps.z])[:2]
            cents = np.array([c.centroid for c in clusters])
            d = np.linalg.norm(cents - cam, axis=1)
            near = np.flatnonzero(d <= 1.0)
            if len(near) != 1:
                return
            self.pairs.append((cam, cents[near[0]]))
            if len(self.pairs) % 15 == 0:
                ok, bs, rs = spread_ok(np.array([q[1] for q in self.pairs]))
                self.get_logger().info(f"calibration: {len(self.pairs)}/{calib_pairs} pairs, bearing spread "
                                       f"{bs:.0f} deg, range spread {rs:.1f} m" + ("" if ok else "  (walk wider/nearer-farther)"))

        def _calib_finish(self) -> None:
            n = len(self.pairs)
            log = self.get_logger()
            if n < 30:
                log.error(f"calibration: only {n} pairs (need 30+). One person in view, robot free, walk slowly.")
                raise SystemExit(1)
            src = np.array([q[0] for q in self.pairs])
            dst = np.array([q[1] for q in self.pairs])
            ok, bs, rs = spread_ok(dst)
            al, keep, rb, ra = fit_rigid_2d(src, dst)
            log.info(f"calibration: {n} pairs ({int(keep.sum())} inliers), bearing spread {bs:.0f} deg, range spread {rs:.1f} m")
            log.info(f"calibration: error before {rb:.3f} m -> after {ra:.3f} m | correction yaw "
                     f"{math.degrees(al.yaw):+.2f} deg, x {al.tx:+.3f} m, y {al.ty:+.3f} m")
            if not ok:
                log.error("calibration NOT saved: walk across the whole image and between ~1 m and ~4 m, then rerun")
                raise SystemExit(1)
            al.meta = {"pairs": n, "inliers": int(keep.sum()), "rms_before_m": round(rb, 4),
                       "rms_after_m": round(ra, 4), "date": time.strftime("%Y-%m-%d %H:%M")}
            path = self.g("align_file")
            al.save(path)
            log.info(f"calibration saved to {path}; restart the keeper to use it")
            if abs(math.degrees(al.yaw)) > 2.0 or math.hypot(al.tx, al.ty) > 0.05:
                log.warn("large offset: also check the camera pose in the Pi URDF (base_link -> camera_link)")
            raise SystemExit(0)

    return KeeperNode


def main(argv=None) -> None:  # pragma: no cover - needs ROS
    import argparse

    import rclpy
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="camera <-> lidar alignment (robot free, still)")
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--seconds", type=float, default=90.0)
    args = ap.parse_args(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = build_node(args.calibrate, args.pairs, args.seconds)()
    code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = int(e.code or 0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
