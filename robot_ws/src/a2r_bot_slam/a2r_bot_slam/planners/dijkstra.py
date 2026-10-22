#!/usr/bin/env python3
"""
dijkstra.py
===========
Dijkstra's algorithm, as a named module.

The implementation is `astar.search` with the heuristic switched off, because
that is exactly what Dijkstra is: A* is Dijkstra plus a heuristic, and setting
the heuristic to zero recovers the original algorithm precisely.  Keeping one
priority-queue search rather than two near-identical copies means the tie
breaking, the corner-cutting rule, the obstacle weighting and the path
reconstruction cannot drift apart between them — which matters, because the
main reason to keep Dijkstra around is to check A*'s answers against it.

When to reach for it:

  * **Verifying the heuristic.**  If A* and Dijkstra return different costs on
    the same map, the heuristic is inadmissible and A* is no longer optimal.
    scripts/test_global_planners.py asserts they match on four maps; it is the
    first test in the file for a reason.

  * **Cost fields.**  `astar.cost_field` floods the whole map from one source,
    giving the distance to every reachable cell at once.  That is a Dijkstra
    expansion with no goal, and it is what you want for frontier selection or
    for "which of these twelve waypoints is nearest".

  * **No usable heuristic.**  If the cost function ever stops being a metric —
    one-way corridors, terrain that costs more uphill than down — the octile
    heuristic becomes inadmissible and Dijkstra is the correct fallback.

What it costs, measured on four maps at 5 cm resolution
(scripts/test_global_planners.py):

    scenario              A* expansions   Dijkstra expansions   speedup
    open field                      181                39,693      118x
    three-wall maze              20,004                32,464      1.4x
    two rooms, one door             980                38,961       30x
    random clutter                2,317                36,944       13x

Identical path costs in every case.  On an open map the heuristic does almost
all the work; in a maze it can do very little, because the walls mean straight
line distance stops predicting anything.  That spread is worth knowing before
you assume A* is always the answer.
"""

from a2r_bot_slam.planners.astar import search as _search
from a2r_bot_slam.planners.astar import cost_field  # noqa: F401  (re-export)


def plan(grid, start_cell, goal_cell, **kw):
    """Shortest path by Dijkstra.  Returns (path_cells, stats) or (None, stats).

    Same signature and same return shape as astar.plan_astar, so the two are
    drop-in interchangeable.  `heuristic_weight` is forced to zero: passing
    one would silently turn this back into A*, and a function called `plan` in
    a module called `dijkstra` should be Dijkstra.
    """
    kw.pop('heuristic_weight', None)
    kw.pop('heuristic', None)
    return _search(grid, start_cell, goal_cell,
                   heuristic_weight=0.0, **kw)


# Alias, so callers can use the same naming as the astar module.
plan_dijkstra = plan
