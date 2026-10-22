#!/usr/bin/env python3
"""
grid.py
=======
The shared substrate every global planner sits on: an occupancy grid, the
conversion between world metres and grid cells, line-of-sight checks, and
path post-processing.

Keeping this in one place is not tidiness for its own sake.  A* and Dijkstra
differ by about four lines of code; RRT and RRT* differ by about twenty.  If
each carried its own copy of the world model, they would drift, and then a
path that A* considers legal would be one RRT rejects, and the two would
disagree about the same map.  Debugging that is miserable.  One Grid class
means "is this cell free?" has exactly one answer.
"""

import heapq
import math

import numpy as np

# Occupancy conventions, matching nav_msgs/OccupancyGrid.
UNKNOWN = -1
FREE = 0
LETHAL = 100


class Grid:
    """An occupancy grid with world<->cell conversion and collision tests."""

    __slots__ = ('data', 'width', 'height', 'resolution',
                 'origin_x', 'origin_y', 'lethal_threshold',
                 'unknown_is_free')

    def __init__(self, data, width, height, resolution,
                 origin_x=0.0, origin_y=0.0,
                 lethal_threshold=65, unknown_is_free=False):
        self.data = np.asarray(data, dtype=np.int16).reshape(height, width)
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.lethal_threshold = int(lethal_threshold)
        # Whether to plan through cells the robot has never seen.
        #
        # This is a genuine policy choice, not a detail. `False` means the
        # robot will only commit to routes through territory it has actually
        # observed, which is the safe answer for navigation. `True` means it
        # will optimistically route through unexplored space, which is what
        # you want for frontier exploration — the whole point there is to go
        # somewhere you have not been.
        self.unknown_is_free = bool(unknown_is_free)

    # ------------------------------------------------------------------
    #  Conversion
    # ------------------------------------------------------------------
    def world_to_cell(self, x, y):
        """Metres in the map frame -> (col, row). May be out of bounds."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return col, row

    def cell_to_world(self, col, row):
        """(col, row) -> metres at the CENTRE of the cell.

        The half-cell offset matters. Without it every path is biased by
        2.5 cm towards the origin, which is small until you are threading a
        doorway with 4 cm of clearance either side.
        """
        return (self.origin_x + (col + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)

    def in_bounds(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    # ------------------------------------------------------------------
    #  Occupancy
    # ------------------------------------------------------------------
    def cost(self, col, row):
        if not self.in_bounds(col, row):
            return LETHAL
        return int(self.data[row, col])

    def is_free(self, col, row):
        if not self.in_bounds(col, row):
            return False
        v = int(self.data[row, col])
        if v == UNKNOWN:
            return self.unknown_is_free
        return v < self.lethal_threshold

    def is_free_world(self, x, y):
        return self.is_free(*self.world_to_cell(x, y))

    # ------------------------------------------------------------------
    #  Line of sight
    # ------------------------------------------------------------------
    def line_of_sight(self, c0, r0, c1, r1):
        """Bresenham visibility test between two cells.

        Used by Theta*-style path smoothing and by the RRT collision check.
        Integer-only, so it cannot miss a cell to floating-point error the
        way a naive "sample every 0.02 m" loop can.
        """
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        err = dc - dr
        c, r = c0, r0

        while True:
            if not self.is_free(c, r):
                return False
            if c == c1 and r == r1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def line_of_sight_world(self, x0, y0, x1, y1):
        c0, r0 = self.world_to_cell(x0, y0)
        c1, r1 = self.world_to_cell(x1, y1)
        return self.line_of_sight(c0, r0, c1, r1)

    # ------------------------------------------------------------------
    #  Nearest free cell
    # ------------------------------------------------------------------
    def nearest_free(self, col, row, max_radius_cells=40):
        """Spiral outwards for a free cell.

        Both endpoints of a plan routinely land in occupied space: the goal
        because the operator clicked on a wall in RViz, and the *start*
        because the robot's own inflation footprint marks the cell it is
        standing in as lethal. Refusing to plan in either case is technically
        correct and practically useless, so we snap to the nearest free cell
        and let the local planner handle the last few centimetres.
        """
        if self.is_free(col, row):
            return col, row
        for radius in range(1, int(max_radius_cells) + 1):
            for dc in range(-radius, radius + 1):
                for dr in (-radius, radius):
                    if self.is_free(col + dc, row + dr):
                        return col + dc, row + dr
            for dr in range(-radius + 1, radius):
                for dc in (-radius, radius):
                    if self.is_free(col + dc, row + dr):
                        return col + dc, row + dr
        return None


# ======================================================================
#  Path post-processing
# ======================================================================
def simplify(path, grid, max_passes=3):
    """Theta*-style line-of-sight shortcutting.

    Grid planners produce paths made of 45-degree staircases, because those
    are the only headings an 8-connected grid can express. Driving one is
    unpleasant and slow — the car saws left and right through every diagonal.

    Repeatedly removing any waypoint whose neighbours can see each other
    collapses the staircase into straight runs. This does NOT make the path
    shorter in the graph-search sense (A* already found the shortest
    8-connected path); it makes it shorter in the Euclidean sense, which is
    the one the robot actually drives.
    """
    if len(path) < 3:
        return list(path)

    pts = list(path)
    for _ in range(max_passes):
        out = [pts[0]]
        i = 0
        changed = False
        while i < len(pts) - 1:
            j = len(pts) - 1
            # Greedily reach as far ahead as visibility allows.
            while j > i + 1:
                if grid.line_of_sight_world(pts[i][0], pts[i][1],
                                            pts[j][0], pts[j][1]):
                    break
                j -= 1
            if j > i + 1:
                changed = True
            out.append(pts[j])
            i = j
        pts = out
        if not changed:
            break
    return pts


def densify(path, spacing):
    """Resample a path to roughly even spacing.

    A simplified path may have a 6 m straight run as two points. The path
    follower does lookahead by walking the point list, so it needs points to
    walk; and RViz draws straight lines between them, which hides the fact
    that a long segment might clip a corner after the map updates.
    """
    if len(path) < 2 or spacing <= 0.0:
        return list(path)

    out = [path[0]]
    for k in range(len(path) - 1):
        x0, y0 = path[k]
        x1, y1 = path[k + 1]
        d = math.hypot(x1 - x0, y1 - y0)
        n = int(d / spacing)
        for m in range(1, n + 1):
            t = (m * spacing) / d
            if t >= 1.0:
                break
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append((x1, y1))
    return out


def path_length(path):
    return sum(math.hypot(path[k + 1][0] - path[k][0],
                          path[k + 1][1] - path[k][1])
               for k in range(len(path) - 1))


# ======================================================================
#  Shared search scaffolding
# ======================================================================
# 8-connected neighbourhood with true Euclidean step costs. Using 1.0 for
# diagonals — which a surprising number of tutorials do — makes diagonal
# movement free and produces paths that zig-zag for no reason.
NEIGHBOURS_8 = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
)

NEIGHBOURS_4 = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
)


def reconstruct(came_from, current):
    """Walk the parent chain back to the start."""
    out = [current]
    guard = 0
    while current in came_from:
        current = came_from[current]
        out.append(current)
        guard += 1
        if guard > 2_000_000:
            break
    out.reverse()
    return out


def soft_cost(grid, col, row, weight):
    """Extra cost for passing near an obstacle.

    The costmap carries an inflation gradient: 100 at an obstacle, falling
    off with distance. Treating everything below the lethal threshold as
    equally free throws that gradient away and produces paths that scrape
    along walls — legal, and horrible to drive, because the local planner
    then spends the whole journey fighting to stay off them.

    Folding a fraction of the cell cost into the edge cost makes the planner
    prefer the middle of a corridor without forbidding the edges, so it will
    still squeeze through a tight gap when that is the only way.
    """
    if weight <= 0.0:
        return 0.0
    c = grid.cost(col, row)
    if c <= 0:
        return 0.0
    return weight * (c / 100.0) * grid.resolution


__all__ = [
    'Grid', 'UNKNOWN', 'FREE', 'LETHAL',
    'simplify', 'densify', 'path_length',
    'NEIGHBOURS_8', 'NEIGHBOURS_4', 'reconstruct', 'soft_cost',
    'heapq',
]
