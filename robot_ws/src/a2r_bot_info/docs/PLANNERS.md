# Planners

Two independent choices.  The **global** planner picks a route across the map;
the **local** planner decides what the wheels do in the next 100 ms.  Both can
be changed while the robot is driving.

```bash
ros2 param set /global_planner_node algorithm rrt_star
ros2 param set /local_planner_bridge planner fgm
```

---

## Local planners — `a2r_bot_follow`

Eight, all sharing one *driver-skill layer* (`_situation`, `_bias`,
`_commit_update`, `_finish`, `_score_arc`).  That sharing is the important
part: trap detection, the three-point escape, side commitment, heading memory,
the speed law and the steering slew limit apply to **all eight**, so switching
planner changes how the direction is chosen and nothing else about how the car
behaves.

| `planner:=` | What it does | Good at | Bad at |
|---|---|---|---|
| `dwa` | Samples a window of (v, steer) arcs, scores each | General purpose, predictable | Concave traps |
| `apf` | Attractive goal + repulsive obstacles | Smooth, cheap | Local minima between two obstacles |
| `teb` | Elastic band, time-optimal-ish | Smooth paths through gaps | Cost, tuning |
| `vfh` | Polar histogram of obstacle density | Cluttered rooms | Narrow gaps at speed |
| `pure_pursuit` | Geometric chase of a lookahead point | Following a path or a person — **the default** | Does no avoidance of its own |
| `fgm` | Follow the largest gap | Corridors, doorways | Open space (wanders) |
| `lattice` | Precomputed arc set, scored by rollout | Predictive, fast | Fixed resolution |
| `rrt` | Kinodynamic sampling tree | Anything requiring a multi-curvature path | Stochastic, needs a time budget |

### Why RRT was added, and what it actually buys

The other seven all commit to a **single curvature** per cycle.  DWA, lattice
and TEB sample a fan of arcs; APF descends a field; VFH and FGM pick a bearing
from a histogram.  Every one of them chooses one arc, and therefore none of
them can express a path that curves one way and then the other.

That matters in exactly one situation, and it is a common one: a gap on the
left followed by a gap on the right.  Measured in
`scripts/test_rrt_offline.py` on a two-wall S-curve, the best possible
single arc stalls **1.14 m short** of the goal — it is not a tuning problem,
it is a representational limit.  RRT solves the same scenario on **19 of 20**
random seeds.

Four things were needed to make it honest, and each was found by testing
rather than by inspection:

1. **The sampling cone must be wider than ±90°.** At ±90° the tree cannot curl
   around the tip of a wall, which removes the only capability RRT adds.
2. **The tree must be kinodynamic.** Edges are limited to the car's real
   minimum turning radius (0.445 m — 28° of turn per 22 cm step). Without this
   it returns paths an Ackermann car cannot drive.
3. **Shortcut smoothing must respect the same limit.** Individually legal
   edges can be shortcut into a corner tighter than the turning circle. The
   smoother now checks.
4. **Steps must follow the arc, not the new heading.** Advancing along the new
   heading is a first-order integration error; at 28° per step it is large
   enough that the polyline cuts inside the true arc. The measured symptom was
   pure pursuit demanding 35° of steer on a servo that stops at 30°, on 7 of
   12 seeds. After the fix: 0 of 20.

RRT correctly **refuses** a pocket that can only be left in reverse (10/10
seeds) and hands it to the existing VFH+T trap detector and three-point escape
manoeuvre, which can reverse.  A planner that admits defeat is worth more than
one that returns an undrivable path.

Cost: ~31 ms per plan on a development machine against a 100 ms budget at
10 Hz.  Budget several times that on a Pi 4 and lower `rrt_iterations` first if
it overruns.

---

## Global planners — `a2r_bot_slam`

| `algorithm:=` | Notes |
|---|---|
| `astar` | Octile heuristic. **The default, and it should stay the default.** |
| `dijkstra` | A* with the heuristic off. Same answer, more work. |
| `rrt` | Sampling with line-of-sight shortcutting. |
| `rrt_star` | Sampling with rewiring. |

### Measured, not asserted

From `scripts/test_global_planners.py`, four maps at 5 cm resolution:

**A\* against Dijkstra** — identical path cost on every map, which is the check
that the heuristic is admissible.  Expansions differ enormously:

| Map | A\* | Dijkstra | Speedup |
|---|---|---|---|
| open field | 181 | 39,693 | 118x |
| three-wall maze | 20,004 | 32,464 | 1.4x |
| two rooms, one door | 980 | 38,961 | 30x |
| random clutter | 2,317 | 36,944 | 13x |

The maze row is the interesting one.  In open space the straight-line
heuristic predicts almost perfectly and A* walks straight to the goal; in a
maze the walls mean straight-line distance predicts very little, and A*
degenerates towards Dijkstra.  Worth knowing before assuming A* is always the
answer.

**RRT against RRT\*** — rewiring genuinely works: RRT\* raw trees are 6–11%
shorter.  But line-of-sight shortcutting recovers that gain for plain RRT too,
so **after smoothing the two are within 1.1% of each other**, while RRT\* costs
about **38x** the time.

Both sampling planners land within 1–10% of A\*'s optimal cost.

**The conclusion is that neither sampling planner is the right default on a
2-D grid.**  A\* is optimal, deterministic and comparable in time.  RRT and
RRT\* are here because they were asked for, because RRT\* demonstrably does
what it claims, and because they become the right answer the moment you plan
in a space A\* cannot search — one that includes heading, or velocity, or a
trailer angle.

### Two settings that matter more than the algorithm

**`unknown_is_free`.** Whether to plan through cells never observed.
`false` while navigating a finished map: an unknown cell there is far more
likely to be somewhere the lidar could not see into (under a sofa, behind a
doorframe) than open floor.  `true` for frontier exploration, where the entire
point is to go somewhere you have not been.

**`obstacle_weight`.** Makes cells near obstacles expensive but passable, so
the route runs down the middle of a corridor rather than shaving the
doorframe.  The shortest path through a doorway touches the frame, and a path
that touches the frame is one control error from a collision.
