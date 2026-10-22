# Kinematics of the car

Measured by hand, September 2026. Before this, the numbers in the workspace
were nominal WR-01 figures and several were wrong by enough to matter.

## Measured

| Quantity | Value | Source |
|---|---|---|
| Length, tyre to tyre | 400 mm | measured |
| Tyre diameter | 120 mm | measured |
| Width, tyre to tyre | 310 mm | measured |
| Base plate I | 420 x 150 x 3 mm | measured |
| Plate II | 320 x 150 x 3 mm | measured |
| Plate I to plate II standoff | 80 mm | measured |
| Lidar above plate I | 130 mm | measured |
| Lidar above ground | 310 mm | measured |
| Lidar standoff | 48 mm | measured |
| Lidar (A2M12) footprint | 70 mm | measured |
| Lidar lateral standoff gaps | 55 mm right, 53 mm left | measured |
| IMU above plate I | 45 mm | measured |
| Camera centre above plate I | 110 mm | measured |
| Camera back from plate I front | 25 mm | measured |

## Derived

**Wheelbase = 280 mm.** "Tyre to tyre" spans both tyres, so the axles are
400 − 120 = 280 mm apart. The workspace said 257 mm, which made the planner
believe the car turns about 9 % tighter than it does.

**Wheel radius = 60 mm**, from the 120 mm tyre.

**Footprint**, in `base_link` (origin midway between the axles):

    front  +0.215 m   plate I half-length 0.210 + the sonar face
    rear   -0.210 m
    half width 0.155 m

**Turning geometry** at the 30° servo limit:

| | radius |
|---|---|
| rear axle | 0.485 m |
| `base_link` | 0.505 m |
| outer front corner | 0.732 m |

A full-lock U-turn therefore needs **about 1.46 m** wall to wall. The old
figures said 0.445 m and 1.34 m.

## The model

`a2r_bot_follow/ackermann.py` is the only implementation, used by the planner
and the collision monitor so they cannot disagree. It is a kinematic bicycle
referenced at `base_link`, which sits `lr = L/2` ahead of the rear axle:

    beta   = atan(lr * tan(delta) / L)
    dpsi   = cos(beta) * tan(delta) / L * ds
    dx, dy = ds * (cos(psi + beta), sin(psi + beta))

`ds` is signed arc length, so reverse arcs come out right instead of needing a
special case. The steering angle may ramp along the path at the servo's slew
rate rather than snapping to the target, because a tight arc genuinely starts
later than the instant-steer assumption predicts.

Collision checking sweeps the **rectangle above**, not a circle. The circle
(0.195 m) fitted neither the 0.215 m nose nor the 0.73 m corner sweep, so the
planner proposed arcs the collision monitor then vetoed, and the car stopped
in front of obstacles instead of driving round them.

## Still to measure

**`lidar_offset_x`** — how far forward of `base_link` the lidar spindle sits.
The sheet gives the lateral position but not this one, and it is currently
set to **-0.140 m** (lidar at the rear of the upper deck, as in the URDF). Every obstacle in the costmap shifts with it, so it
is worth a tape measure:

    lidar_offset_x = (distance from the front bumper to the lidar spindle) - 0.210

Set it in three places, which must agree:

    src/a2r_bot_follow/config/params.yaml        lidar_offset_x
    src/a2r_bot_slam/config/slam.yaml            lidar_offset_x_m
    src/a2r_bot_sensor_fusion/config/fusion.yaml scan_frame_offset_x_m
    src/a2r_bot_description/urdf/common.xacro    lidar_x   (the URDF is the
                                                 source of truth; TF wins over
                                                 the params where TF exists)

Two other assumptions worth confirming: the camera is taken to be on the
centreline (the sheet's "75 mm from centre" could not be placed), and the
front of the car is taken as 0.215 m on the assumption the sonar face sits
just proud of the plate.
