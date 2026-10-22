#!/usr/bin/env python3
"""
rrt.py
======
Sampling-based global planners: RRT and RRT*.

These sit alongside A* and Dijkstra in the same `global_planner_node`, behind
the same interface, and are chosen by a parameter.

-----------------------------------------------------------------------------
When a sampling planner beats a grid search, and when it does not
-----------------------------------------------------------------------------
Be honest about this, because the fashionable answer is wrong for a robot
like this one.

On a 2-D occupancy grid of a building, **A\\* is better than RRT nearly
always**. It is complete (if a path exists it finds it), optimal (it finds
the shortest one), deterministic (same map, same path, every time), and on a
1000x1000 grid it runs in tens of milliseconds. RRT is none of those things:
it is only *probabilistically* complete, its paths are typically 10-20%
longer than optimal, and it returns a different answer every run.

RRT earns its place here for three specific reasons:

  1. **Very large or fine maps.** A* cost scales with the number of cells.
     At 2 cm resolution over a 50 x 50 m site that is six million cells and
     A* starts to hurt. RRT's cost scales with the difficulty of the *route*,
     not the size of the map, so it barely notices.

  2. **The state space stops being a grid.** The moment you want to plan in
     (x, y, heading) — which you must, if you want a global path a car can
     follow without a three-point turn at every corner — the grid becomes
     three-dimensional and A* becomes expensive. Sampling does not care.

  3. **Anytime behaviour.** RRT* improves its answer the longer it runs and
     can be stopped whenever. A* gives you nothing at all until it finishes.

So: the default global planner is A*, and RRT is there for the cases above
and for comparison. That ordering is deliberate.

-----------------------------------------------------------------------------
RRT vs RRT*
-----------------------------------------------------------------------------
Plain RRT connects each new node to its nearest neighbour and never revisits
that choice, so the tree it grows is arbitrary and the path is visibly
wandering. RRT* adds two steps:

  * **Choose parent**: connect a new node to whichever nearby node gives the
    lowest total cost from the start, not merely the nearest one.
  * **Rewire**: check whether the new node is a better parent for any of its
    neighbours, and re-parent them if so.

That makes RRT* asymptotically optimal — the path converges to the true
shortest path as samples accumulate. The cost is a neighbourhood query per
sample instead of a nearest query, which roughly doubles the per-sample work.
It is worth it: at equal wall-clock time RRT* paths are markedly shorter, and
the offline test measures exactly this.

The neighbourhood radius shrinks as the tree grows, following the standard
result

    r = min(gamma * (log(n) / n) ** (1/d), step)

with d = 2 for a planar problem. A fixed radius either does too little work
early or too much late.
"""

import math
import random

import numpy as np

from a2r_bot_slam.planners.grid import path_length


# ======================================================================
def _sample(grid, gx, gy, goal_bias, rng):
    """Draw a sample in world coordinates, biased towards the goal."""
    if rng.random() < goal_bias:
        return gx, gy
    x = grid.origin_x + rng.random() * grid.width * grid.resolution
    y = grid.origin_y + rng.random() * grid.height * grid.resolution
    return x, y


def _steer(fx, fy, tx, ty, step):
    """Move from (fx,fy) towards (tx,ty) by at most `step`."""
    d = math.hypot(tx - fx, ty - fy)
    if d <= step:
        return tx, ty
    return fx + (tx - fx) / d * step, fy + (ty - fy) / d * step


# ======================================================================
def plan_rrt(grid, start_world, goal_world,
             max_iterations=6000,
             step_m=0.30,
             goal_bias=0.08,
             goal_tolerance_m=0.25,
             seed=None,
             smooth_passes=120,
             **_ignored):
    """
    Plain RRT.  Returns (path_world, stats) or (None, stats).

    `path_world` is a list of (x, y) in the map frame.
    """
    rng = random.Random(seed)
    sx, sy = start_world
    gx, gy = goal_world

    if not grid.is_free_world(sx, sy):
        return None, {'reason': 'start occupied', 'iterations': 0}
    if not grid.is_free_world(gx, gy):
        return None, {'reason': 'goal occupied', 'iterations': 0}

    xs = [sx]
    ys = [sy]
    parent = [-1]

    goal_idx = -1
    it = 0

    for it in range(1, int(max_iterations) + 1):
        tx, ty = _sample(grid, gx, gy, goal_bias, rng)

        ax = np.asarray(xs)
        ay = np.asarray(ys)
        d2 = (ax - tx) ** 2 + (ay - ty) ** 2
        ni = int(np.argmin(d2))

        nx, ny = _steer(xs[ni], ys[ni], tx, ty, step_m)
        if not grid.line_of_sight_world(xs[ni], ys[ni], nx, ny):
            continue

        xs.append(nx)
        ys.append(ny)
        parent.append(ni)

        if math.hypot(nx - gx, ny - gy) <= goal_tolerance_m:
            if grid.line_of_sight_world(nx, ny, gx, gy):
                xs.append(gx)
                ys.append(gy)
                parent.append(len(xs) - 2)
                goal_idx = len(xs) - 1
                break

    if goal_idx < 0:
        return None, {'reason': 'goal not reached', 'iterations': it,
                      'nodes': len(xs)}

    path = _extract(xs, ys, parent, goal_idx)
    raw_len = path_length(path)
    path = _shortcut(path, grid, smooth_passes, rng)

    return path, {
        'reason': 'ok',
        'iterations': it,
        'nodes': len(xs),
        'raw_length': raw_len,
        'length': path_length(path),
    }


# ======================================================================
def plan_rrt_star(grid, start_world, goal_world,
                  max_iterations=6000,
                  step_m=0.30,
                  goal_bias=0.08,
                  goal_tolerance_m=0.25,
                  gamma=2.5,
                  seed=None,
                  smooth_passes=120,
                  keep_improving=True,
                  **_ignored):
    """
    RRT*.  Same interface as plan_rrt, asymptotically optimal.

    `keep_improving=True` runs the full iteration budget even after the goal
    is first reached, which is what makes it converge — stopping at the first
    solution gives you plain RRT with extra steps.
    """
    rng = random.Random(seed)
    sx, sy = start_world
    gx, gy = goal_world

    if not grid.is_free_world(sx, sy):
        return None, {'reason': 'start occupied', 'iterations': 0}
    if not grid.is_free_world(gx, gy):
        return None, {'reason': 'goal occupied', 'iterations': 0}

    xs = [sx]
    ys = [sy]
    parent = [-1]
    cost = [0.0]

    best_goal = -1
    best_cost = float('inf')
    first_solution_at = None
    it = 0

    for it in range(1, int(max_iterations) + 1):
        tx, ty = _sample(grid, gx, gy, goal_bias, rng)

        ax = np.asarray(xs)
        ay = np.asarray(ys)
        d2 = (ax - tx) ** 2 + (ay - ty) ** 2
        ni = int(np.argmin(d2))

        nx, ny = _steer(xs[ni], ys[ni], tx, ty, step_m)
        if not grid.line_of_sight_world(xs[ni], ys[ni], nx, ny):
            continue

        n = len(xs)
        # Shrinking neighbourhood radius — the standard RRT* schedule.
        radius = min(gamma * math.sqrt(math.log(n + 1) / (n + 1)), step_m * 3.0)

        dn = np.sqrt((ax - nx) ** 2 + (ay - ny) ** 2)
        near = np.nonzero(dn <= radius)[0]

        # ---- choose parent: cheapest, not nearest ----------------------
        best_parent = ni
        best_parent_cost = cost[ni] + math.hypot(nx - xs[ni], ny - ys[ni])
        for k in near:
            k = int(k)
            c = cost[k] + float(dn[k])
            if c < best_parent_cost and grid.line_of_sight_world(
                    xs[k], ys[k], nx, ny):
                best_parent = k
                best_parent_cost = c

        xs.append(nx)
        ys.append(ny)
        parent.append(best_parent)
        cost.append(best_parent_cost)
        new_idx = len(xs) - 1

        # ---- rewire: is the new node a better parent for its neighbours?
        for k in near:
            k = int(k)
            if k == best_parent:
                continue
            through_new = best_parent_cost + float(dn[k])
            if through_new < cost[k] and grid.line_of_sight_world(
                    nx, ny, xs[k], ys[k]):
                parent[k] = new_idx
                improvement = cost[k] - through_new
                cost[k] = through_new
                # Propagate the saving to the subtree. Skipping this leaves
                # stale costs, and stale costs make later choose-parent
                # decisions wrong — the classic quiet RRT* bug, which shows
                # up as "RRT* is no better than RRT".
                _propagate(parent, cost, k, improvement)

        # ---- goal connection -------------------------------------------
        dg = math.hypot(nx - gx, ny - gy)
        if dg <= goal_tolerance_m and grid.line_of_sight_world(nx, ny, gx, gy):
            total = best_parent_cost + dg
            if total < best_cost:
                best_cost = total
                best_goal = new_idx
                if first_solution_at is None:
                    first_solution_at = it
            if not keep_improving:
                break

    if best_goal < 0:
        return None, {'reason': 'goal not reached', 'iterations': it,
                      'nodes': len(xs)}

    path = _extract(xs, ys, parent, best_goal)
    path.append((gx, gy))
    raw_len = path_length(path)
    path = _shortcut(path, grid, smooth_passes, rng)

    return path, {
        'reason': 'ok',
        'iterations': it,
        'nodes': len(xs),
        'first_solution_at': first_solution_at,
        'tree_cost': best_cost,
        'raw_length': raw_len,
        'length': path_length(path),
    }


# ======================================================================
def _propagate(parent, cost, root, improvement):
    """Subtract `improvement` from the cost of everything below `root`.

    Implemented iteratively with an explicit stack. A recursive version is
    prettier and blows the stack on a tree of a few thousand nodes, which is
    a normal size here.
    """
    children = {}
    for i, p in enumerate(parent):
        if p >= 0:
            children.setdefault(p, []).append(i)

    stack = list(children.get(root, ()))
    guard = 0
    while stack:
        node = stack.pop()
        cost[node] -= improvement
        stack.extend(children.get(node, ()))
        guard += 1
        if guard > 200_000:
            break


def _extract(xs, ys, parent, idx):
    out = []
    guard = 0
    while idx >= 0 and guard < 200_000:
        out.append((xs[idx], ys[idx]))
        idx = parent[idx]
        guard += 1
    out.reverse()
    return out


def _shortcut(path, grid, passes, rng):
    """Randomised shortcutting against the grid's line-of-sight test.

    Unlike the local planner's version, there is no turning-circle constraint
    here: a global path is a route, and the local planner is responsible for
    driving it with whatever geometry the car has. Imposing curvature limits
    at this level would make the global planner refuse routes the car can
    actually take with a single reversing manoeuvre.
    """
    if len(path) < 3:
        return list(path)
    pts = list(path)
    for _ in range(int(passes)):
        if len(pts) < 3:
            break
        i = rng.randint(0, len(pts) - 3)
        j = rng.randint(i + 2, len(pts) - 1)
        if grid.line_of_sight_world(pts[i][0], pts[i][1],
                                    pts[j][0], pts[j][1]):
            pts = pts[:i + 1] + pts[j:]
    return pts


__all__ = ['plan_rrt', 'plan_rrt_star']
