# A2R_bot

An Ackermann-steered robot that follows its owner, builds maps, and navigates
them.  Three computers: a Jetson Nano that sees, a Raspberry Pi 4 that thinks
about geometry, and an Arduino Nano ESP32 that counts microseconds.

This workspace is the Pi's half.

**The ROS workspace and the Arduino firmware are one project, and they build
with one command.**

```bash
cd ~/a2r_bot_ws

make first-run        # rosdep + colcon + arduino core + firmware, from scratch
make udev             # stable /dev/ttyARDUINO and /dev/ttyLIDAR
make flash            # firmware onto the board  (CAR ON BLOCKS)
make doctor           # what is installed, what is plugged in, what disagrees

make follow           # follow the owner
make slam             # build a map
make nav              # drive to a goal
make pet              # just the face
make base             # hardware only, no planners
```

`make help` lists everything. Variables override the defaults without editing
anything:

```bash
make follow PLANNER=rrt          # any of the eight local planners
make nav ALGO=dijkstra           # any of the four global planners
make flash PORT=/dev/ttyACM0     # if the udev rule is not installed yet
```

The plain ROS commands still work if you prefer them:

```bash
colcon build --symlink-install && source install/setup.bash
ros2 launch a2r_bot_bringup follow.launch.py
```

Start with **[`src/a2r_bot_info/docs/QUICKSTART.md`](src/a2r_bot_info/docs/QUICKSTART.md)**.

---

## Packages

| Package | What it is |
|---|---|
| `a2r_bot_description` | The URDF. 123 links, real steering joints, three RViz configs. |
| `a2r_bot_arduino` | Firmware, plus the serial bridge that is the sole owner of `/dev/ttyARDUINO`. |
| `a2r_bot_sensor_fusion` | Kalman filters: attitude, planar pose, and lidar+sonar scan fusion. |
| `a2r_bot_follow` | The owner-following local planner. Eight interchangeable algorithms. |
| `a2r_bot_slam` | Occupancy mapping, global costmap, four global planners, path following. |
| `a2r_bot_pet` | The animated face. |
| `a2r_bot_bringup` | One launch file per mode. |
| `a2r_bot_info` | All the documentation. |

---

## The one design decision worth knowing

**There is one path-following stack, with two front ends.**

In follow mode the Jetson publishes `/cmd_vel`.  In navigation mode
`navigation_node` publishes `/cmd_vel`.  `smart_planner_node` subscribes to it
either way and cannot tell the difference — so all eight local planners, the
VFH+T trap detector, the three-point escape manoeuvre and the speed law apply
unchanged in both modes, and either planner can be swapped mid-drive:

```bash
ros2 param set /global_planner_node algorithm rrt_star     # the route
ros2 param set /local_planner_bridge planner fgm           # the avoidance
```

That is what makes the two modes interchangeable in a sense worth the word:
not shared code between two stacks, but one stack.

---

## Two hardware corrections

Both were in the original brief as "everything on the shared A4/A5 I²C bus".
Neither is possible.

1. **The MAX7219 is not an I²C device.** It is a 3-wire shift register with no
   address and no ACK. Wiring its DIN to SDA would also break the IMU. It now
   has three dedicated GPIO pins and a bit-banged driver.

2. **"DYP-A21" is a family.** Only the **IIC** variant can share the bus —
   check the sticker on the back of yours. The firmware supports I²C, UART and
   analog behind one compile flag, and ROS sees identical topics either way.

Details in [`HARDWARE.md`](src/a2r_bot_info/docs/HARDWARE.md).

---

## What was verified, and how

No ROS installation was available while this was written, so everything
testable was tested standalone.  The suites run on any machine with numpy:

```bash
python3 scripts/test_rrt_offline.py        # local RRT, 10 checks
python3 scripts/test_global_planners.py    # A*, Dijkstra, RRT, RRT*
```

| Component | Result |
|---|---|
| URDF | 123 links, 122 joints, single root, valid tree — expanded and checked with a standalone xacro runner |
| Firmware | All five modules plus the sketch compile clean under `g++` against Arduino stubs |
| 7-segment font | All ten digit codes checked against the segment truth table |
| LiPo curve | Strictly monotonic; divider ratios verified for both 3.3 V and 5 V boards |
| Kalman library | Velocity converges, covariance stays positive-definite, gating fires, quaternions round-trip exactly, magnetometer yaw lands at 1.001 rad against a true 1.000, ±π seam handled |
| Scan fusion | Glass-door case (lidar blind) and table-leg case (sonar blind) both behave correctly; only-shortens invariant holds over 500 random cases |
| Local RRT | S-curve solved 19/20 seeds where the best single arc stalls 1.14 m short; must-reverse pocket correctly refused 10/10; every tree edge respects the 0.445 m turning circle; 31 ms per plan |
| Global planners | A* and Dijkstra agree exactly on four maps (so the heuristic is admissible); A* 12–118x fewer expansions; RRT* trees 6–11% shorter raw but within 1.1% of RRT after smoothing, at ~38x the cost |
| Distance transform | Exact against brute force |
| Map saver | PGM round-trip verified, including row order |
| Face | `draw_eye`, `draw_mouth` and `ease_in_out` byte-identical to the original |

The RRT numbers are worth a footnote: four real bugs surfaced only because the
planner was run against synthetic obstacle fields rather than inspected.  The
sampling cone was too narrow to do the one thing RRT is for; the tree had no
kinematic constraint and returned undrivable paths; the smoother created
corners tighter than the turning circle out of individually legal edges; and
the extension step integrated along the new heading instead of the chord,
which made pure pursuit demand 35° of steer on a servo that stops at 30°.
See [`PLANNERS.md`](src/a2r_bot_info/docs/PLANNERS.md).

---

## Layout

```
a2r_bot_ws/
├── README.md
├── scripts/            test suites and generators (no ROS needed)
├── udev/               stable /dev/ttyARDUINO and /dev/ttyLIDAR
└── src/
    ├── a2r_bot_description/
    ├── a2r_bot_arduino/        firmware/ + docs/WIRING.md
    ├── a2r_bot_sensor_fusion/
    ├── a2r_bot_follow/
    ├── a2r_bot_slam/           planners/{grid,astar,dijkstra,rrt}.py
    ├── a2r_bot_pet/
    ├── a2r_bot_bringup/
    └── a2r_bot_info/           docs/
```

Licence: Apache-2.0.
