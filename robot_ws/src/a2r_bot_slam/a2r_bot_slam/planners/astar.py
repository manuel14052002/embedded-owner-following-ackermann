#!/usr/bin/env python3
"""
astar.py
========
A* and Dijkstra over the occupancy grid.

The two are the same algorithm with one difference: A* adds a heuristic
estimate of the remaining distance when ordering the frontier, and Dijkstra
does not.  They are implemented as one function with a `heuristic_weight`
argument rather than two copies, because two copies of a search loop is how
you end up fixing a bug in one of them.

    heuristic_weight = 0.0   ->  Dijkstra
    heuristic_weight = 1.0   ->  A*, optimal
    heuristic_weight > 1.0   ->  weighted A*, faster, up to w times longer

-----------------------------------------------------------------------------
Which should you use?
-----------------------------------------------------------------------------
A*, almost always.  With an admissible heuristic it returns exactly the same
path as Dijkstra while expanding far fewer nodes — on a typical room map,
about a fifth as many.

Dijkstra earns its place in two situations:

  * **No meaningful goal.** Dijkstra expands outwards in all directions, so
    one run gives the cost to *every* reachable cell. That is what you want
    for frontier exploration ("what is the nearest unexplored cell?") or for
    a coverage plan, where there is no single destination to aim at.

  * **Checking A*.** If A* and Dijkstra return different path lengths on the
    same map, the heuristic is inadmissible and A* is not returning optimal
    paths. The offline test does exactly this comparison.

The heuristic here is the **octile distance**, not Euclidean. On an
8-connected grid the true minimum cost between two cells is

    octile(dx, dy) = (dx + dy) + (sqrt(2) - 2) * min(dx, dy)

which is the cost of moving diagonally as far as possible and then straight.
Euclidean distance underestimates this, so it is still admissible — A* would
return correct paths — but it is a weaker bound, so the search expands more
nodes for no benefit. Octile is exact for an empty grid, which makes it the
tightest admissible heuristic available.
"""

import heapq
import math

from a2r_bot_slam.planners.grid import (NEIGHBOURS_8, NEIGHBOURS_4,
                                        reconstruct, soft_cost)

SQRT2_MINUS_2 = math.sqrt(2.0) - 2.0


def octile(c0, r0, c1, r1):
    """Exact minimum cost between two cells on an empty 8-connected grid."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    return (dc + dr) + SQRT2_MINUS_2 * min(dc, dr)


def euclidean(c0, r0, c1, r1):
    return math.hypot(c1 - c0, r1 - r0)


def manhattan(c0, r0, c1, r1):
    return abs(c1 - c0) + abs(r1 - r0)


HEURISTICS = {
    'octile': octile,
    'euclidean': euclidean,
    'manhattan': manhattan,
}


def search(grid, start_cell, goal_cell,
           heuristic_weight=1.0,
           heuristic='octile',
           allow_diagonal=True,
           obstacle_weight=0.6,
           max_expansions=400_000):
    """
    Grid search from start_cell to goal_cell, both (col, row).

    Returns (path_cells, stats) where path_cells is a list of (col, row) from
    start to goal inclusive, or None if unreachable.  stats is a dict with
    the expansion count and cost, which is what you look at when a planner
    feels slow.
    """
    h_fn = HEURISTICS.get(heuristic, octile)
    neighbours = NEIGHBOURS_8 if allow_diagonal else NEIGHBOURS_4

    sc, sr = start_cell
    gc, gr = goal_cell

    if not grid.in_bounds(sc, sr) or not grid.in_bounds(gc, gr):
        return None, {'reason': 'endpoint out of bounds', 'expansions': 0}
    if not grid.is_free(sc, sr):
        return None, {'reason': 'start occupied', 'expansions': 0}
    if not grid.is_free(gc, gr):
        return None, {'reason': 'goal occupied', 'expansions': 0}

    start = (sc, sr)
    goal = (gc, gr)

    if start == goal:
        return [start], {'expansions': 0, 'cost': 0.0, 'reason': 'trivial'}

    # g_score in CELLS, converted to metres only at the end. Working in cells
    # keeps the heuristic and the edge costs in the same units without a
    # multiply in the inner loop.
    g_score = {start: 0.0}
    came_from = {}
    closed = set()

    h0 = h_fn(sc, sr, gc, gr) * heuristic_weight
    open_heap = [(h0, 0.0, start)]
    expansions = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            # Stale heap entry. We push duplicates rather than implementing a
            # decrease-key operation, because Python's heapq has no such
            # operation and the lazy-deletion approach is both simpler and,
            # in practice, faster than maintaining an indexed heap.
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            path = reconstruct(came_from, current)
            return path, {
                'expansions': expansions,
                'cost': g * grid.resolution,
                'reason': 'ok',
            }

        if expansions > max_expansions:
            return None, {
                'reason': 'expansion limit (%d) hit — map too large or goal '
                          'unreachable' % max_expansions,
                'expansions': expansions,
            }

        cc, cr = current
        for dc, dr, step in neighbours:
            nc, nr = cc + dc, cr + dr
            if (nc, nr) in closed:
                continue
            if not grid.is_free(nc, nr):
                continue

            # Corner-cutting check. Without this a diagonal move can slip
            # between two obstacles that touch at their corners — the path is
            # legal cell by cell, but no physical robot fits through the gap,
            # because the gap has zero width.
            if dc != 0 and dr != 0:
                if not grid.is_free(cc + dc, cr) or not grid.is_free(cc, cr + dr):
                    continue

            tentative = g + step + soft_cost(grid, nc, nr, obstacle_weight)
            if tentative < g_score.get((nc, nr), float('inf')):
                g_score[(nc, nr)] = tentative
                came_from[(nc, nr)] = current
                f_new = tentative + heuristic_weight * h_fn(nc, nr, gc, gr)
                heapq.heappush(open_heap, (f_new, tentative, (nc, nr)))

    return None, {'reason': 'no path exists', 'expansions': expansions}


# ----------------------------------------------------------------------
#  Named entry points
# ----------------------------------------------------------------------
def plan_astar(grid, start_cell, goal_cell, **kw):
    """A* with an admissible octile heuristic. Optimal."""
    kw.setdefault('heuristic_weight', 1.0)
    return search(grid, start_cell, goal_cell, **kw)


def plan_dijkstra(grid, start_cell, goal_cell, **kw):
    """Dijkstra. Same result as A*, more nodes expanded."""
    kw['heuristic_weight'] = 0.0
    return search(grid, start_cell, goal_cell, **kw)


def plan_weighted_astar(grid, start_cell, goal_cell, weight=1.5, **kw):
    """Weighted A*. Up to `weight` times longer than optimal, much faster.

    Worth having for large maps where an optimal path is not worth a two
    second pause. The bound is a guarantee, not an estimate: the returned
    path is never worse than `weight` times the optimal one.
    """
    kw['heuristic_weight'] = float(weight)
    return search(grid, start_cell, goal_cell, **kw)


def cost_field(grid, start_cell, allow_diagonal=True,
               obstacle_weight=0.0, max_expansions=400_000):
    """Dijkstra with no goal: cost from start_cell to every reachable cell.

    This is the thing Dijkstra can do that A* cannot. Used by the exploration
    logic to answer "which frontier is actually nearest?" — nearest in
    *travel* distance through the map, not in straight-line distance, which
    on a map with walls in it is a very different question.
    """
    neighbours = NEIGHBOURS_8 if allow_diagonal else NEIGHBOURS_4
    sc, sr = start_cell
    if not grid.is_free(sc, sr):
        return {}

    dist = {(sc, sr): 0.0}
    heap = [(0.0, (sc, sr))]
    closed = set()
    expansions = 0

    while heap:
        d, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if expansions > max_expansions:
            break

        cc, cr = cur
        for dc, dr, step in neighbours:
            nc, nr = cc + dc, cr + dr
            if (nc, nr) in closed or not grid.is_free(nc, nr):
                continue
            if dc != 0 and dr != 0:
                if not grid.is_free(cc + dc, cr) or not grid.is_free(cc, cr + dr):
                    continue
            nd = d + step + soft_cost(grid, nc, nr, obstacle_weight)
            if nd < dist.get((nc, nr), float('inf')):
                dist[(nc, nr)] = nd
                heapq.heappush(heap, (nd, (nc, nr)))

    return dist


__all__ = ['search', 'plan_astar', 'plan_dijkstra', 'plan_weighted_astar',
           'cost_field', 'octile', 'euclidean', 'manhattan', 'HEURISTICS']
