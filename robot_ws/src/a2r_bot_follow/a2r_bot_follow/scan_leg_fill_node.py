#!/usr/bin/env python3
"""scan_leg_fill_node - lidar-only chair handling: /scan -> /scan/planner + /scan/merged.

The lidar (31 cm high) does hit chair legs - a few points per leg - but the legs stand
40-50 cm apart, so the planner sees free space BETWEEN them and drives under the seat.
This node finds narrow, leg-like objects in the scan and closes every gap between two of
them that is narrower than ``max_gap_m`` with a virtual wall (ray/segment intersection, so
the geometry is exact). The planner and the collision monitor then see a chair as one solid
obstacle. Table legs (> max_gap apart) stay open, so the robot can still pass under tables.

v2: the OWNER (/follow/owner) is removed from the planner's scan (never an obstacle, never glued
to a chair by a gap wall), and thin posts become 32 cm discs (swivel-chair column -> its whole
five-arm base below the lidar plane; splayed plastic-chair legs -> their feet). The collision
monitor gets /scan/merged, which still contains the owner, so it always stops in front of them.
owner_fusion keeps the raw /scan, so the owner's own legs are tracked exactly as before.

a2r_fix_freeze (Oct 2026): one vectorised cluster pass (legs AND posts), all gap pairs and posts
solved in one batch, depth-1 scan queue, body box cached, TF listener dropped once the mount is
known. ~5x less CPU per scan, same output.
"""

from __future__ import annotations

import array
import math
from typing import Sequence

import numpy as np


def leg_clusters(ranges: Sequence[float], angle_min: float, angle_inc: float, range_min: float,
                 max_range: float = 3.0, break_m: float = 0.08, max_width_m: float = 0.10,
                 min_points: int = 2, leg_ref_m: float = 0.02,
                 adaptive: bool = True) -> list[tuple[float, float]]:
    """Narrow objects (chair/table legs, poles) as centroids (x, y) in the laser frame.
    A cluster is a run of consecutive valid beams whose neighbours are < break_m apart; it is
    leg-like if its extent is <= max_width_m and it has enough points.

    v2.8 - why chairs were driven into:
      * "enough points" was a fixed 2. With the old ~1 deg beams a 2.5 cm leg at 1.5 m
        is ONE beam, so no chair leg was ever found beyond ~0.7 m: by the time the
        gap wall appeared the car was already committed between the legs.
        adaptive=True asks for ~40 % of the beams a leg_ref_m-wide leg would return at
        that range, between 1 and min_points + 2. Far away, at low resolution, one
        beam is enough; close up, at high resolution, a lone mixed pixel is not.
      * a leg stands IN FRONT of what is beside it. A cluster whose neighbour beam is
        nearer than itself is a mixed pixel on an edge (or a leg half hidden behind
        something), not a free-standing leg - rejected. This is what stops the extra
        resolution from turning every table edge into a phantom chair.
      * vectorised (was a Python loop over every beam; there are ~2500 now)."""
    c, _ = leg_clusters_ext(ranges, angle_min, angle_inc, range_min, max_range, break_m, max_width_m,
                            min_points, leg_ref_m, adaptive)
    return [(float(x), float(y)) for x, y in c]


def leg_clusters_ext(ranges, angle_min: float, angle_inc: float, range_min: float,
                     max_range: float = 3.0, break_m: float = 0.08, max_width_m: float = 0.10,
                     min_points: int = 2, leg_ref_m: float = 0.02, adaptive: bool = True,
                     step_m: float = 0.15, search_beams: int = 25, min_need: int = 2):
    """a2r_fix_freeze: leg_clusters without a Python loop over the runs (the Pi spent most of its
    80-150 ms per scan here). Returns (centroids (k, 2), extents (k,)) so ONE pass serves both the
    legs (extent <= max_leg_width) and the posts (extent <= post_max_width). Same rules as v2.8."""
    empty = (np.zeros((0, 2)), np.zeros(0))
    r = np.asarray(ranges, dtype=np.float64)
    n = r.size
    if n == 0:
        return empty
    a = angle_min + angle_inc * np.arange(n)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(r) & (r > max(range_min, 0.05)) & (r <= max_range)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return empty
    rv = r[idx]
    x, y = rv * np.cos(a[idx]), rv * np.sin(a[idx])
    link = (np.diff(idx) == 1) & (np.hypot(np.diff(x), np.diff(y)) <= break_m)
    cuts = np.flatnonzero(~link) + 1
    st = np.r_[0, cuts]
    en = np.r_[cuts - 1, idx.size - 1]
    full_ring = abs(abs(angle_inc) * n - 2 * math.pi) < 2 * abs(angle_inc)
    wrap = (full_ring and len(st) > 1 and idx[0] == 0 and idx[-1] == n - 1
            and math.hypot(x[0] - x[-1], y[0] - y[-1]) <= break_m)
    # per-run sums (cumulative sums: exact enough, no Python loop)
    cx, cy, cr = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(y)], np.r_[0.0, np.cumsum(rv)]
    first, last = st.copy(), en.copy()
    size = (en - st + 1).astype(np.int64)
    sx, sy, sr = cx[en + 1] - cx[st], cy[en + 1] - cy[st], cr[en + 1] - cr[st]
    if wrap:            # 360 deg wrap: the last run continues into the first one
        first[0] = st[-1]
        size[0] += size[-1]
        sx[0] += sx[-1]
        sy[0] += sy[-1]
        sr[0] += sr[-1]
        first, last, size, sx, sy, sr = first[:-1], last[:-1], size[:-1], sx[:-1], sy[:-1], sr[:-1]
    ext = np.hypot(x[last] - x[first], y[last] - y[first])
    rc = sr / size
    if adaptive:
        n_exp = leg_ref_m / np.maximum(rc * abs(angle_inc), 1e-6)
        need = np.minimum(np.maximum(min_need, np.round(0.4 * n_exp)), min_points + 2).astype(np.int64)
    else:
        need = np.full(size.shape, max(int(min_points), int(min_need)))
    keep = (ext <= max_width_m) & (size >= need)
    # a2r_no_phantoms: a free-standing leg stands >= step_m IN FRONT of its surroundings on BOTH
    # sides. Look past dropped beams to the nearest real return on each side (up to search_beams):
    # a wall cut into pieces by dropouts (dark / shiny / grazing) has the same wall right beside each
    # piece - those pieces were ~100 phantom "legs" per scan. Old rule (next beam only) kept them,
    # because the next beam was the dropped one. Nothing found within the window = open space.
    lo = max(range_min, 0.05)
    with np.errstate(invalid="ignore"):
        ok_b = np.isfinite(r) & (r > lo)
    if full_ring:
        rr, okr, off = np.r_[r, r, r], np.r_[ok_b, ok_b, ok_b], n
    else:
        rr, okr, off = np.r_[np.full(n, np.nan), r, np.full(n, np.nan)], \
            np.r_[np.zeros(n, bool), ok_b, np.zeros(n, bool)], n
    ar = np.arange(rr.size)
    prev_ok = np.maximum.accumulate(np.where(okr, ar, -1))                    # last valid index <= i
    next_ok = np.minimum.accumulate(np.where(okr, ar, rr.size)[::-1])[::-1]   # first valid index >= i
    i0, i1 = idx[first] + off - 1, idx[last] + off + 1
    j0, j1 = prev_ok[i0], next_ok[np.minimum(i1, rr.size - 1)]
    f0 = (j0 >= 0) & (i0 - j0 < search_beams)
    f1 = (j1 < rr.size) & (j1 - i1 < search_beams)
    with np.errstate(invalid="ignore"):
        s0 = np.where(f0, np.where(rr[np.clip(j0, 0, rr.size - 1)] >= rc + step_m, 2, 0), 1)
        s1 = np.where(f1, np.where(rr[np.clip(j1, 0, rr.size - 1)] >= rc + step_m, 2, 0), 1)
    keep &= (s0 + s1) >= 3            # step on both sides, or a step on one side + open space
    cen = np.stack([sx[keep] / size[keep], sy[keep] / size[keep]], axis=1)
    return cen, ext[keep]


def fill_leg_gaps(ranges: Sequence[float], angle_min: float, angle_inc: float, range_min: float,
                  max_range: float = 3.0, max_gap_m: float = 0.60, min_gap_m: float = 0.05,
                  **cluster_kw) -> np.ndarray:
    """Close every gap between two leg-like objects that is narrower than max_gap_m: each beam
    that crosses the segment between them gets the segment's distance (if that is closer)."""
    out = np.asarray(ranges, dtype=np.float64).copy()
    legs = leg_clusters(out, angle_min, angle_inc, range_min, max_range, **cluster_kw)
    return fill_leg_gaps_from(out, angle_min, angle_inc, range_min, legs, max_gap_m, min_gap_m)


def _beams_between(a1: float, span: float, angle_min: float, angle_inc: float, n: int) -> np.ndarray:
    """Beam indices from bearing a1 sweeping `span` radians (signed), inclusive."""
    k0 = int(round((a1 - angle_min) / angle_inc))
    k1 = int(round((a1 + span - angle_min) / angle_inc))
    lo, hi = min(k0, k1), max(k0, k1)              # a2r_fix_freeze: the same beams, no np.unique
    return np.arange(lo, hi + 1, dtype=np.int64) % n


def fill_leg_gaps_from(ranges, angle_min: float, angle_inc: float, range_min: float, legs,
                       max_gap_m: float = 0.60, min_gap_m: float = 0.05) -> np.ndarray:
    """Close the gaps between the given leg centroids (laser frame).
    a2r_fix_freeze: the pairs are picked with one distance matrix and every crossing beam of every
    pair is solved in one batch (was: a numpy round per pair). Nearest wall wins, as before."""
    out = np.asarray(ranges, dtype=np.float64).copy()
    n = out.size
    L = np.asarray(legs, dtype=np.float64).reshape(-1, 2)
    if len(L) < 2 or n == 0:
        return out
    d = np.hypot(L[:, None, 0] - L[None, :, 0], L[:, None, 1] - L[None, :, 1])
    i, j = np.nonzero(np.triu((d >= min_gap_m) & (d <= max_gap_m), 1))
    if i.size == 0:
        return out
    x1, y1, x2, y2 = L[i, 0], L[i, 1], L[j, 0], L[j, 1]
    a1 = np.arctan2(y1, x1)
    span = (np.arctan2(y2, x2) - a1 + math.pi) % (2 * math.pi) - math.pi     # signed, shortest way round
    k0 = np.round((a1 - angle_min) / angle_inc).astype(np.int64)
    k1 = np.round((a1 + span - angle_min) / angle_inc).astype(np.int64)
    lo, cnt = np.minimum(k0, k1), np.abs(k1 - k0) + 1
    pid = np.repeat(np.arange(i.size), cnt)
    k = (lo[pid] + np.arange(pid.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)) % n
    beam = angle_min + angle_inc * k
    dx, dy = np.cos(beam), np.sin(beam)
    # ray (t*dx, t*dy) meets segment P1 + u*(P2-P1):  solve 2x2
    X1, Y1 = x1[pid], y1[pid]
    ex, ey = x2[pid] - X1, y2[pid] - Y1
    det = dx * (-ey) - dy * (-ex)
    ok = np.abs(det) >= 1e-9
    det = np.where(ok, det, 1.0)
    t = (X1 * (-ey) - Y1 * (-ex)) / det
    u = (dx * Y1 - dy * X1) / det
    ok &= (t > 0) & (u >= -1e-6) & (u <= 1 + 1e-6)
    return _write_nearest(out, k[ok], t[ok], range_min)


def _write_nearest(out: np.ndarray, k: np.ndarray, t: np.ndarray, floor: float) -> np.ndarray:
    """out[k] = t where the beam had no return (or one <= floor) or t is nearer; nearest t per beam."""
    if k.size == 0:
        return out
    best = np.full(out.size, np.inf)
    np.minimum.at(best, k, t)
    cur = out
    with np.errstate(invalid="ignore"):
        w = np.isfinite(best) & (~np.isfinite(cur) | (cur <= floor) | (best < cur))
    out[w] = best[w]
    return out


def inflate_posts(ranges: np.ndarray, angle_min: float, angle_inc: float, posts, radius: float,
                  body_box=None) -> np.ndarray:
    """Each thin post (x, y) in the laser frame becomes a disc of ``radius`` (ray/circle
    intersection, nearest wins). A swivel-chair column becomes its whole five-arm base, and a
    splayed chair leg covers its foot. Posts the laser is already inside of are left alone, and no
    point is ever written inside the robot's own outline (the collision monitor would stop for good).
    a2r_fix_freeze: all posts in one batch."""
    out = np.asarray(ranges, dtype=np.float64).copy()
    n = out.size
    C = np.asarray(posts, dtype=np.float64).reshape(-1, 2)
    if len(C) == 0 or n == 0:
        return out
    dist = np.hypot(C[:, 0], C[:, 1])
    C, dist = C[dist > radius + 0.02], dist[dist > radius + 0.02]   # too close to tell: never trap the robot
    if len(C) == 0:
        return out
    half = np.arcsin(np.minimum(1.0, radius / dist))
    a0 = np.arctan2(C[:, 1], C[:, 0])
    k0 = np.round((a0 - half - angle_min) / angle_inc).astype(np.int64)
    k1 = np.round((a0 + half - angle_min) / angle_inc).astype(np.int64)
    lo, cnt = np.minimum(k0, k1), np.abs(k1 - k0) + 1
    pid = np.repeat(np.arange(len(C)), cnt)
    k = (lo[pid] + np.arange(pid.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)) % n
    beam = angle_min + angle_inc * k
    cb, sb = np.cos(beam), np.sin(beam)
    cx, cy, dd = C[pid, 0], C[pid, 1], dist[pid]
    b = cx * cb + cy * sb
    disc = b * b - (dd * dd - radius * radius)
    ok = (disc >= 0) & (b > 0)
    t = b - np.sqrt(np.where(ok, disc, 0.0))
    if body_box is not None:
        px, py = t * cb, t * sb
        ok &= ~((body_box[0] < px) & (px < body_box[1]) & (body_box[2] < py) & (py < body_box[3]))
    ok &= t > 0
    return _write_nearest(out, k[ok], t[ok], 0.0)


class LegPersistence:
    """a2r_no_phantoms: a leg/post only counts if one was also seen within ``radius`` in one of the
    last ``keep`` scans - real legs are there scan after scan, noise / dust 'legs' flicker for one.
    Compared in the odom frame when ``T_next`` (3x3 laser->odom for the coming scan) is set, so the
    car's own motion does not break the match; otherwise in the laser frame with a wider radius."""

    def __init__(self, keep: int = 2, radius: float = 0.10, radius_no_odom: float = 0.18):
        from collections import deque
        self.hist = deque(maxlen=keep)
        self.radius, self.radius_no_odom = radius, radius_no_odom
        self.T_next = None

    def __call__(self, cen) -> np.ndarray:
        """cen (k, 2) laser frame -> bool mask of the confirmed ones; remembers this scan."""
        C = np.asarray(cen, dtype=np.float64).reshape(-1, 2)
        T = self.T_next
        W = C if T is None else (C @ T[:2, :2].T + T[:2, 2])
        rad = self.radius if T is not None else self.radius_no_odom
        mask = np.zeros(len(C), bool)
        for H in self.hist:
            if len(H) and len(W):
                d = np.hypot(W[:, None, 0] - H[None, :, 0], W[:, None, 1] - H[None, :, 1])
                mask |= d.min(axis=1) <= rad
        self.hist.append(W)
        return mask


def process_scan(ranges: Sequence[float], angle_min: float, angle_inc: float, range_min: float,
                 owner_xy=None, owner_radius: float = 0.45, body_box=None, post_radius: float = 0.32,
                 post_max_width: float = 0.08, max_range: float = 3.0, max_gap_m: float = 0.60,
                 confirm=None, grouped_post_radius=None, **cluster_kw):
    """Returns (planner_ranges, monitor_ranges), all in the laser frame.
    1. the OWNER (owner_xy, laser frame) is removed: never an obstacle for the planner, never part
       of a chair wall;  2. gaps <= max_gap_m between leg-like objects are closed;
    3. thin posts (<= post_max_width) become discs of post_radius (swivel-chair bases, splayed legs).
    Returns of the robot's own body (body_box = (xmin, xmax, ymin, ymax), laser frame) are never
    treated as legs or posts. The monitor scan = planner scan + the raw owner returns (so the
    collision monitor still stops in front of the owner)."""
    raw = np.asarray(ranges, dtype=np.float64)
    n = raw.size
    a = angle_min + angle_inc * np.arange(n)
    with np.errstate(invalid="ignore"):
        x, y = raw * np.cos(a), raw * np.sin(a)
    work = raw.copy()
    if owner_xy is not None:
        near = np.isfinite(raw) & (np.hypot(x - owner_xy[0], y - owner_xy[1]) < owner_radius)
        work[near] = np.inf
    det = work.copy()
    if body_box is not None:
        bx0, bx1, by0, by1 = body_box
        body = np.isfinite(raw) & (x > bx0) & (x < bx1) & (y > by0) & (y < by1)
        det[body] = np.inf                            # own body: never a leg or a post
    # a2r_fix_freeze: ONE cluster pass; posts are the legs no wider than post_max_width
    leg_w = float(cluster_kw.pop("max_width_m", 0.10))
    cen, ext = leg_clusters_ext(det, angle_min, angle_inc, range_min, max_range,
                                max_width_m=max(leg_w, post_max_width), **cluster_kw)
    if confirm is not None:                       # a2r_no_phantoms: must persist across scans
        m = confirm(cen)
        cen, ext = cen[m], ext[m]
    legs = cen[ext <= leg_w]
    posts = cen[ext <= post_max_width]
    out = fill_leg_gaps_from(work, angle_min, angle_inc, range_min, legs, max_gap_m)
    if grouped_post_radius is not None and len(posts) and len(cen) > 1:
        # a2r_step2: only a LONE post (swivel-chair column) stands for a hidden 32 cm base. A post with
        # another leg within max_gap_m is one leg of a chair/table: the gap walls above already make
        # it solid, and a 32 cm disc on each leg grew a 0.43 m chair to ~1.0 m and closed real
        # passages (2026-10-09 probe: discs filled 2-3x more beams than the chair walls themselves).
        d = np.hypot(posts[:, None, 0] - cen[None, :, 0], posts[:, None, 1] - cen[None, :, 1])
        grouped = ((d > 1e-6) & (d <= max_gap_m)).any(axis=1)
        out = inflate_posts(out, angle_min, angle_inc, posts[~grouped], post_radius, body_box)
        if grouped_post_radius > 0.0:
            out = inflate_posts(out, angle_min, angle_inc, posts[grouped], grouped_post_radius, body_box)
    else:
        out = inflate_posts(out, angle_min, angle_inc, posts, post_radius, body_box)
    monitor = np.where(np.isfinite(raw) & ~np.isfinite(work), raw, out)   # owner back in
    monitor = np.minimum(monitor, np.where(np.isfinite(raw), raw, np.inf))
    return out, monitor


def main() -> None:  # pragma: no cover - needs ROS
    import time

    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener

    class ScanLegFill(Node):
        def __init__(self) -> None:
            super().__init__("scan_leg_fill")
            p = self.declare_parameter
            p("input_topic", "/scan")
            p("planner_topic", "/scan/planner")   # chairs solid, OWNER REMOVED -> local planner
            p("monitor_topic", "/scan/merged")    # chairs solid, owner kept    -> collision monitor
            p("owner_topic", "/follow/owner")     # owner_fusion output, base_link
            p("owner_scrub_topic", "/follow/owner_scrub")   # a2r_pi_update2: owner point for cutting out only
            p("owner_max_age_s", 0.5)
            p("owner_radius_m", 0.45)
            p("max_gap_m", 0.60)                  # chair legs ~0.40-0.50 apart; table legs stay open
            p("max_leg_width_m", 0.10)
            p("post_max_width_m", 0.08)           # thinner than this = post (chair column, chair leg)
            p("post_radius_m", 0.32)              # swivel-chair base radius / splayed-leg foot
            p("grouped_post_radius_m", 0.06)      # a2r_step2: foot of a leg that belongs to a chair (< 0 = old rule)
            p("cluster_break_m", 0.08)
            p("min_points", 2)
            p("adaptive_min_points", True)        # v2.8: points needed scale with range/resolution
            p("leg_ref_width_m", 0.02)            # thinnest leg to expect (m)
            p("depth_step_m", 0.15)               # a2r_no_phantoms: a leg stands this much in front of its surroundings
            p("step_search_beams", 25)            # a2r_no_phantoms: look past this many dropped beams
            p("leg_min_points", 2)                # a2r_no_phantoms: never a leg from a single return
            p("persist_scans", 2)                 # a2r_no_phantoms: also seen in one of the last N scans (0 = off)
            p("persist_radius_m", 0.10)           # a2r_no_phantoms: same leg if within this (odom frame)
            p("odom_topic", "/odometry/filtered") # a2r_no_phantoms: the car's own motion between scans
            p("max_range_m", 3.0)
            p("base_frame", "base_link")
            p("lidar_offset_x", -0.140)           # until TF base_link->laser is known
            p("lidar_offset_y", 0.001)
            p("lidar_yaw_offset", 0.0)
            p("footprint_front_m", 0.215)
            p("footprint_rear_m", -0.210)
            p("footprint_half_width_m", 0.155)
            p("body_margin_m", 0.03)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.g = g
            self.kw = dict(max_range=float(g("max_range_m")), max_gap_m=float(g("max_gap_m")),
                           max_width_m=float(g("max_leg_width_m")), break_m=float(g("cluster_break_m")),
                           min_points=int(g("min_points")), post_radius=float(g("post_radius_m")),
                           adaptive=bool(g("adaptive_min_points")), leg_ref_m=float(g("leg_ref_width_m")),
                           post_max_width=float(g("post_max_width_m")), owner_radius=float(g("owner_radius_m")),
                           step_m=float(g("depth_step_m")), search_beams=int(g("step_search_beams")),
                           min_need=int(g("leg_min_points")))
            _gpr = float(g("grouped_post_radius_m"))      # a2r_step2
            self.kw["grouped_post_radius"] = _gpr if _gpr >= 0.0 else None
            n_p = int(g("persist_scans"))                 # a2r_no_phantoms
            self.persist = LegPersistence(n_p, float(g("persist_radius_m"))) if n_p > 0 else None
            self.odom_pose = None
            if self.persist is not None:
                from nav_msgs.msg import Odometry
                self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 10)
            self.base = g("base_frame")
            self.pose = (float(g("lidar_offset_x")), float(g("lidar_offset_y")), float(g("lidar_yaw_offset")))
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=False)
            self.tf_ok = False
            self.owner = None
            self.owner_t = -1e9
            self.pub_plan = self.create_publisher(LaserScan, g("planner_topic"), qos_profile_sensor_data)
            self.pub_mon = self.create_publisher(LaserScan, g("monitor_topic"), qos_profile_sensor_data)
            self.create_subscription(PointStamped, g("owner_topic"), self._on_owner, 10)
            self.scrub, self.scrub_t, self.bad_frame = None, -1e9, 0      # a2r_pi_update2
            self.create_subscription(PointStamped, g("owner_scrub_topic"), self._on_scrub, 10)
            # a2r_fix_freeze: depth 1 - when the Pi is busy, process the NEWEST scan and drop the rest
            # (depth 5 queued old scans: the planner and the monitor then drove on a stale world)
            latest = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(LaserScan, g("input_topic"), self._on_scan, latest)
            self._bb, self._age = None, float(g("owner_max_age_s"))
            self.stats = dict(n=0, filled=0, owner=0, ms=0.0)
            self.create_timer(10.0, self._report)
            self.get_logger().info(
                f"{g('input_topic')} -> {g('planner_topic')} (owner removed) + {g('monitor_topic')} (owner kept): "
                f"chair-leg gaps <= {self.kw['max_gap_m']:.2f} m closed, posts -> {self.kw['post_radius']:.2f} m discs")

        def _on_owner(self, msg: PointStamped) -> None:
            if msg.header.frame_id and msg.header.frame_id != self.base:
                self.bad_frame += 1                 # a2r_pi_update2: reported, not silent
                return                              # owner_fusion publishes base_link
            self.owner, self.owner_t = (msg.point.x, msg.point.y), time.monotonic()

        def _on_scrub(self, msg: PointStamped) -> None:     # a2r_pi_update2
            if msg.header.frame_id and msg.header.frame_id != self.base:
                self.bad_frame += 1
                return
            self.scrub, self.scrub_t = (msg.point.x, msg.point.y), time.monotonic()

        def _on_odom(self, msg) -> None:                  # a2r_no_phantoms
            q, p_ = msg.pose.pose.orientation, msg.pose.pose.position
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.odom_pose = (p_.x, p_.y, yaw, time.monotonic())

        def _laser_in_odom(self):                         # a2r_no_phantoms: 3x3 laser->odom, or None
            o = self.odom_pose
            if o is None or time.monotonic() - o[3] > 0.5:
                return None
            lx, ly, lyaw = self.pose                      # laser in base_link
            c, s = math.cos(o[2]), math.sin(o[2])
            tx, ty = o[0] + c * lx - s * ly, o[1] + s * lx + c * ly
            ct, st = math.cos(o[2] + lyaw), math.sin(o[2] + lyaw)
            return np.array([[ct, -st, tx], [st, ct, ty], [0.0, 0.0, 1.0]])

        def _laser_pose(self, frame: str):
            if not self.tf_ok:
                try:
                    t = self.tf.lookup_transform(self.base, frame, rclpy.time.Time())
                    q = t.transform.rotation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    self.pose = (t.transform.translation.x, t.transform.translation.y, yaw)
                    self.tf_ok = True
                    self._bb = None                     # body box follows the TF pose
                    self.get_logger().info(f"laser pose from TF: x={self.pose[0]:.3f} y={self.pose[1]:.3f} "
                                           f"yaw={math.degrees(yaw):.1f} deg")
                    try:            # a2r_fix_freeze: static mount - stop parsing every /tf message in Python
                        self.tfl.unregister()
                    except Exception:                   # noqa: BLE001
                        pass
                except TransformException:
                    pass
            return self.pose

        def _to_laser(self, x, y):
            lx, ly, yaw = self.pose
            dx, dy = x - lx, y - ly
            return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)

        def _body_box(self):
            if self._bb is not None:                    # a2r_fix_freeze: computed once per laser pose
                return self._bb
            g, m = self.g, float(self.g("body_margin_m"))
            xs = (float(g("footprint_rear_m")) - m, float(g("footprint_front_m")) + m)
            ys = (-float(g("footprint_half_width_m")) - m, float(g("footprint_half_width_m")) + m)
            c = [self._to_laser(x, y) for x in xs for y in ys]
            self._bb = (min(p[0] for p in c), max(p[0] for p in c), min(p[1] for p in c), max(p[1] for p in c))
            return self._bb

        def _copy(self, scan, ranges):
            out = LaserScan()
            out.header = scan.header
            out.angle_min, out.angle_max, out.angle_increment = scan.angle_min, scan.angle_max, scan.angle_increment
            out.time_increment, out.scan_time = scan.time_increment, scan.scan_time
            out.range_min, out.range_max = scan.range_min, scan.range_max
            out.ranges = array.array("f", np.asarray(ranges, dtype=np.float32).tobytes())   # v2.8: fast path
            # a2r_merge: no intensities - nobody downstream reads them, and they doubled the size
            # of both re-published scans (planner, monitor, follow_manager all deserialise them)
            return out

        def _on_scan(self, scan: LaserScan) -> None:
            t0 = time.perf_counter()
            self._laser_pose(scan.header.frame_id)
            owner = None
            age = self._age
            now_m = time.monotonic()
            src = None                                   # a2r_pi_update2: tracked owner, else scrub point
            if self.owner is not None and now_m - self.owner_t <= age:
                src = self.owner
            if self.scrub is not None and now_m - self.scrub_t <= age and (src is None or self.scrub_t > self.owner_t):
                src = self.scrub
                self.stats["scrub"] = self.stats.get("scrub", 0) + 1
            if src is not None:
                owner = self._to_laser(*src)
                self.stats["owner"] += 1
            plan, mon = process_scan(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                                     owner_xy=owner, body_box=self._body_box(),
                                     confirm=self._confirm(), **self.kw)
            self.pub_plan.publish(self._copy(scan, plan))
            self.pub_mon.publish(self._copy(scan, mon))
            raw = np.asarray(scan.ranges, dtype=np.float64)
            self.stats["filled"] += int(np.sum(np.isfinite(mon) & (~np.isfinite(raw) | (mon < raw - 1e-6))))
            self.stats["n"] += 1
            self.stats["ms"] += 1e3 * (time.perf_counter() - t0)

        def _confirm(self):                               # a2r_no_phantoms
            if self.persist is None:
                return None
            self.persist.T_next = self._laser_in_odom()
            return self.persist

        def _report(self) -> None:
            n = self.stats["n"]
            if n:
                self.get_logger().info(f"{n} scans: {self.stats['filled'] / n:.1f} beams filled per scan "
                                       f"(chairs), owner removed in {100 * self.stats['owner'] / n:.0f} % "
                                       f"(scrub point {100 * self.stats.get('scrub', 0) / n:.0f} %), "
                                       f"{self.stats['ms'] / n:.1f} ms per scan"
                                       + (f", {self.bad_frame} owner msgs in a wrong frame" if self.bad_frame else ""))
            self.stats = dict(n=0, filled=0, owner=0, ms=0.0)      # a2r_pi_update2: + scrub share
            self.bad_frame = 0

    rclpy.init()
    node = ScanLegFill()
    try:
        import gc               # a2r_merge: init objects out of the GC (no full-pass stalls)
        gc.collect()
        gc.freeze()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
