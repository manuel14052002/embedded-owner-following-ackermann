#!/usr/bin/env python3
"""owner_path_planner.py - global path from the robot to the owner (Jetson side, no ROS here).

Used by lidar_owner_keeper.py, in the same process, so the scan is matched (ICP) and
clustered once per scan for both tracking and planning.

    RollingGrid      log-odds occupancy around the robot in the keeper's world frame (robot
                     motion removed by scan matching). Hits mark occupied, the space each beam
                     crosses is marked free, everything slowly fades to unknown, so people who
                     walked away disappear quickly. The owner's own legs are never an obstacle.
    OwnerPathPlanner obstacles grown by the robot's half width (lethal) plus a clearance cost
                     that keeps the route away from walls and chairs; A* (octile, 8-connected,
                     no corner cutting) from the robot to the owner; shortcut smoothing that
                     keeps a safety margin; cut stand-off_m before the owner ALONG the route;
                     resampled into look-ahead points. Unreachable -> no path (never a path
                     into a wall).
    GoalManager      FOLLOW while the owner is tracked; LAST_KNOWN (their last position) for
                     lost_goal_s after they are lost; then LOST (no goal). Reset on release.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SQ2 = math.sqrt(2.0)


@dataclass
class PlannerConfig:
    size_m: float = 10.0            # rolling window edge
    res_m: float = 0.08             # cell size
    r_lethal_m: float = 0.20        # robot half width 0.155 + margin: centre may not come closer
    clear_weight: float = 4.0       # extra cost near obstacles ...
    clear_lambda_m: float = 0.15    # ... decaying with this length
    line_margin_m: float = 0.10     # shortcuts keep at least r_lethal + this from obstacles
    standoff_m: float = 1.2         # stop this far before the owner, measured along the route
    spacing_m: float = 0.30         # look-ahead point spacing
    goal_search_m: float = 0.6      # owner inside an inflated obstacle: nearest free cell within this
    start_clear_m: float = 0.25     # the robot's own cell is never lethal (it is already there)
    owner_clear_m: float = 0.45     # the owner's legs are not an obstacle
    max_expand: int = 40000
    l_occ: float = 0.85
    l_free: float = -0.40
    l_min: float = -2.0
    l_max: float = 3.5
    decay: float = 0.985            # per scan, toward unknown
    occ_thresh: float = 0.6
    max_rays: int = 720             # rays used for free-space carving (all hits are used)
    recenter_m: float = 1.5


@dataclass
class PlanResult:
    points: np.ndarray              # (K, 2) world frame, look-ahead points from the robot on; empty = none
    status: str                     # ok | reached | unreachable | no_goal | start_blocked
    length_m: float = 0.0           # route length robot -> stop point
    owner_route_m: float = 0.0      # route length robot -> owner
    ms: float = 0.0
    expanded: int = 0


class RollingGrid:
    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self.n = int(round(cfg.size_m / cfg.res_m))
        self.res = cfg.res_m
        self.L = np.zeros((self.n, self.n), np.float32)        # [ix, iy]
        self.origin: Optional[np.ndarray] = None                # world xy of cell (0, 0)'s corner

    def reset(self) -> None:
        self.L[:] = 0.0
        self.origin = None

    def _recenter(self, center: np.ndarray) -> None:
        half = 0.5 * self.n * self.res
        if self.origin is None:
            self.origin = center - half
            return
        off = center - (self.origin + half)
        if abs(off[0]) < self.cfg.recenter_m and abs(off[1]) < self.cfg.recenter_m:
            return
        sx, sy = int(round(off[0] / self.res)), int(round(off[1] / self.res))
        if abs(sx) >= self.n or abs(sy) >= self.n:
            self.L[:] = 0.0
        else:
            self.L = np.roll(self.L, (-sx, -sy), axis=(0, 1))
            if sx > 0: self.L[-sx:, :] = 0.0
            elif sx < 0: self.L[:-sx, :] = 0.0
            if sy > 0: self.L[:, -sy:] = 0.0
            elif sy < 0: self.L[:, :-sy] = 0.0
        self.origin = self.origin + np.array([sx, sy]) * self.res

    def cell(self, xy) -> tuple:
        p = (np.asarray(xy, np.float64) - self.origin) / self.res
        return int(math.floor(p[0])), int(math.floor(p[1]))

    def inside(self, c) -> bool:
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def center_of(self, ix, iy) -> np.ndarray:
        return self.origin + (np.array([ix, iy], np.float64) + 0.5) * self.res

    def _flat(self, xy: np.ndarray) -> np.ndarray:
        c = np.floor((xy - self.origin) / self.res).astype(np.int64)
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.n) & (c[:, 1] >= 0) & (c[:, 1] < self.n)
        c = c[ok]
        return np.unique(c[:, 0] * self.n + c[:, 1])

    def update(self, sensor_xy, pts_world: np.ndarray, owner_xy=None) -> None:
        c = self.cfg
        sensor = np.asarray(sensor_xy, np.float64)
        self._recenter(sensor)
        self.L *= c.decay
        pts = np.asarray(pts_world, np.float64).reshape(-1, 2)
        if len(pts):
            step = max(1, len(pts) // c.max_rays)
            P = pts[::step]
            D = P - sensor
            R = np.hypot(D[:, 0], D[:, 1])
            ns = np.maximum(0, np.floor((R - self.res) / self.res)).astype(np.int64)
            tot = int(ns.sum())
            if tot:
                ray = np.repeat(np.arange(len(P)), ns)
                k = np.arange(tot) - np.repeat(np.cumsum(ns) - ns, ns)
                frac = (k * self.res) / np.maximum(R[ray], 1e-6)
                free = sensor + D[ray] * frac[:, None]
                Lf = self.L.reshape(-1)
                Lf[self._flat(free)] += c.l_free
            Lf = self.L.reshape(-1)
            Lf[self._flat(pts)] += c.l_occ + (-c.l_free)              # a hit cell is never also freed
        np.clip(self.L, c.l_min, c.l_max, out=self.L)
        if owner_xy is not None:
            self.clear_disk(owner_xy, c.owner_clear_m)

    def clear_disk(self, xy, r: float) -> None:
        cx, cy = self.cell(xy)
        k = int(math.ceil(r / self.res)) + 1
        x0, x1 = max(0, cx - k), min(self.n, cx + k + 1)
        y0, y1 = max(0, cy - k), min(self.n, cy + k + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ix, iy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        ctr = (np.stack([ix, iy], -1) + 0.5) * self.res + self.origin
        m = np.hypot(ctr[..., 0] - xy[0], ctr[..., 1] - xy[1]) <= r
        sub = self.L[x0:x1, y0:y1]
        sub[m] = np.minimum(sub[m], 0.0)

    def occupied(self) -> np.ndarray:
        return self.L > self.cfg.occ_thresh


class OwnerPathPlanner:
    def __init__(self, cfg: Optional[PlannerConfig] = None) -> None:
        self.cfg = cfg or PlannerConfig()
        self.grid = RollingGrid(self.cfg)
        self._dist = None

    def update_scan(self, sensor_xy, pts_world, owner_xy=None) -> None:
        self.grid.update(sensor_xy, pts_world, owner_xy)
        self._dist = None

    # -- costs ---------------------------------------------------------------------------
    def _fields(self):
        from scipy.ndimage import distance_transform_edt
        c, g = self.cfg, self.grid
        occ = g.occupied()
        if occ.any():
            dist = distance_transform_edt(~occ) * g.res
        else:
            dist = np.full(occ.shape, 1e3)
        lethal = dist < c.r_lethal_m
        cost = 1.0 + c.clear_weight * np.exp(-np.maximum(dist - c.r_lethal_m, 0.0) / c.clear_lambda_m)
        return dist, lethal, cost

    # -- search ------------------------------------------------------------------------------
    @staticmethod
    def _astar(cost: np.ndarray, lethal: np.ndarray, s: tuple, t: tuple, max_expand: int):
        H, W = cost.shape
        cf = cost.ravel().tolist()
        lf = lethal.ravel().tolist()
        si, ti = s[0] * W + s[1], t[0] * W + t[1]
        tr, tc = t
        g = {si: 0.0}
        parent = {si: -1}
        closed = set()
        k2 = SQ2 - 2.0

        def h(i):
            r, cc = divmod(i, W)
            dr, dc = abs(r - tr), abs(cc - tc)
            return (dr + dc) + k2 * (dr if dr < dc else dc)

        heap = [(h(si), si)]
        nbr = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))
        expanded = 0
        while heap:
            _, i = heapq.heappop(heap)
            if i in closed:
                continue
            if i == ti:
                break
            closed.add(i)
            expanded += 1
            if expanded > max_expand:
                return None, expanded
            r, cc = divmod(i, W)
            gi = g[i]
            for dr, dc, step in nbr:
                rr, c2 = r + dr, cc + dc
                if rr < 0 or rr >= H or c2 < 0 or c2 >= W:
                    continue
                j = rr * W + c2
                if lf[j] or j in closed:
                    continue
                if dr and dc and (lf[r * W + c2] or lf[rr * W + cc]):
                    continue                                    # no squeezing past a corner
                ng = gi + step * cf[j]
                if ng < g.get(j, math.inf):
                    g[j] = ng
                    parent[j] = i
                    heapq.heappush(heap, (ng + h(j), j))
        if ti not in parent:
            return None, expanded
        path, i = [], ti
        while i != -1:
            path.append(divmod(i, W))
            i = parent[i]
        return path[::-1], expanded

    def _line_ok(self, a: np.ndarray, b: np.ndarray, dist: np.ndarray) -> bool:
        g = self.grid
        n = max(2, int(np.hypot(*(b - a)) / (0.5 * g.res)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        p = a + (b - a) * t
        c = np.floor((p - g.origin) / g.res).astype(int)
        if (c < 0).any() or (c >= g.n).any():
            return False
        return bool((dist[c[:, 0], c[:, 1]] >= self.cfg.r_lethal_m + self.cfg.line_margin_m).all())

    def _smooth(self, pts: np.ndarray, dist: np.ndarray) -> np.ndarray:
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_ok(pts[i], pts[j], dist):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)

    @staticmethod
    def _cut_and_resample(pts: np.ndarray, keep_len: float, spacing: float) -> np.ndarray:
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        keep_len = min(keep_len, cum[-1])
        s = np.r_[np.arange(0.0, keep_len, spacing), keep_len]
        if len(s) > 1 and s[-1] - s[-2] < 1e-6:
            s = s[:-1]
        x = np.interp(s, cum, pts[:, 0])
        y = np.interp(s, cum, pts[:, 1])
        return np.stack([x, y], axis=1)

    def plan(self, robot_xy, goal_xy, standoff: Optional[float] = None) -> PlanResult:
        t0 = time.perf_counter()
        c, g = self.cfg, self.grid
        standoff = c.standoff_m if standoff is None else standoff
        robot = np.asarray(robot_xy, np.float64)
        goal = np.asarray(goal_xy, np.float64)
        ms = lambda: 1e3 * (time.perf_counter() - t0)  # noqa: E731
        if g.origin is None:
            return PlanResult(np.zeros((0, 2)), "no_goal", ms=ms())
        straight = float(np.hypot(*(goal - robot)))
        if straight <= standoff + 1e-3:
            return PlanResult(robot[None].copy(), "reached", 0.0, straight, ms())
        dist, lethal, cost = self._fields()
        s, t = g.cell(robot), g.cell(goal)
        if not g.inside(s):
            return PlanResult(np.zeros((0, 2)), "start_blocked", ms=ms())
        k = int(math.ceil(c.start_clear_m / g.res))
        lethal[max(0, s[0] - k):s[0] + k + 1, max(0, s[1] - k):s[1] + k + 1] = False
        if not g.inside(t) or lethal[t]:
            t = self._nearest_free(t, lethal, goal)
            if t is None:
                return PlanResult(np.zeros((0, 2)), "unreachable", ms=ms())
        cells, expanded = self._astar(cost, lethal, s, t, c.max_expand)
        if cells is None:
            return PlanResult(np.zeros((0, 2)), "unreachable", ms=ms(), expanded=expanded)
        pts = np.array([g.center_of(ix, iy) for ix, iy in cells])
        pts[0] = robot
        if g.inside(g.cell(goal)) and not np.allclose(pts[-1], goal):
            pts = np.vstack([pts, goal])                         # the owner, not their cell centre
        pts = self._smooth(pts, dist)
        route = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
        if route <= standoff:
            return PlanResult(robot[None].copy(), "reached", 0.0, route, ms(), expanded)
        look = self._cut_and_resample(pts, route - standoff, c.spacing_m)
        return PlanResult(look, "ok", route - standoff, route, ms(), expanded)

    def _nearest_free(self, t, lethal, goal):
        g = self.grid
        k = int(math.ceil(self.cfg.goal_search_m / g.res))
        x0, x1 = max(0, t[0] - k), min(g.n, t[0] + k + 1)
        y0, y1 = max(0, t[1] - k), min(g.n, t[1] + k + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        sub = ~lethal[x0:x1, y0:y1]
        if not sub.any():
            return None
        ix, iy = np.nonzero(sub)
        ix, iy = ix + x0, iy + y0
        ctr = (np.stack([ix, iy], 1) + 0.5) * g.res + g.origin
        d = np.hypot(ctr[:, 0] - goal[0], ctr[:, 1] - goal[1])
        j = int(np.argmin(d))
        return (int(ix[j]), int(iy[j])) if d[j] <= self.cfg.goal_search_m else None


@dataclass
class GoalManager:
    lost_goal_s: float = 10.0
    last: Optional[np.ndarray] = None
    t_last: float = -math.inf
    mode: str = "NO_OWNER"

    def reset(self) -> None:
        self.last, self.t_last, self.mode = None, -math.inf, "NO_OWNER"

    def update(self, now: float, owner_world, owned: bool):
        """-> (goal world xy or None, mode)."""
        if not owned:
            self.reset()
            return None, "NO_OWNER"
        if owner_world is not None:
            self.last, self.t_last = np.asarray(owner_world, np.float64).copy(), now
            self.mode = "FOLLOW"
            return self.last, self.mode
        if self.last is not None and now - self.t_last <= self.lost_goal_s:
            self.mode = "LAST_KNOWN"
            return self.last, self.mode
        self.mode = "LOST"
        return None, self.mode
