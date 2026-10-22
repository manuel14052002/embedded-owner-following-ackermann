# Architecture

## The three computers

| | Runs | Talks to |
|---|---|---|
| **Jetson Nano** | Camera, person detection, owner tracking, the PET state machine | Publishes `/cmd_vel` and `/pet/state` to the Pi |
| **Raspberry Pi 4** | Everything in this workspace | ROS 2 to the Jetson; serial to the Arduino |
| **Arduino Nano ESP32** | Firmware: PWM, sensors, battery, display | JSON lines over USB serial to the Pi |

The division is deliberate.  The Jetson is the only machine that can see, so
it owns every decision that requires vision.  The Arduino is the only machine
with deterministic timing, so it owns every decision that requires exact
microseconds.  The Pi owns everything in between — which is to say, all the
geometry.

## Data flow, follow mode

```
  Jetson
    │ /cmd_vel  (Twist, m/s and rad/s)
    ▼
  smart_planner_node ◄── /scan/fused ◄── scan_fusion_node ◄── /scan
    │                                            ▲             (lidar)
    │                                            └── /ultrasonic/front,rear
    │ /local_planner/cmd_out  (normalised -1..1)
    ▼
  serial_bridge_node
    │ {"cmd":"DRIVE","v":..,"s":..}
    ▼
  Arduino ──► PCA9685 ch0 (servo)  +  ch1 (HobbyWing 880 ESC)
    │ {"ax":..,"gx":..}, {"f":..,"r":..}, {"bat_v":..}
    ▼
  serial_bridge_node ──► /imu/data_raw, /imu/mag, /ultrasonic/*, /battery
                              │
                              ▼
                       imu_filter_node ──► /imu/data
                              │
                              ▼
                       ekf_state_node ──► /odometry/filtered + odom→base TF
```

## Data flow, navigation mode

Only the top box changes:

```
  /goal_pose (RViz)
    │
    ▼
  global_planner_node ◄── /global_costmap ◄── global_costmap_node ◄── /map
    │ /plan                                                            ▲
    ▼                                                              slam_node
  navigation_node                                                      ▲
    │ /cmd_vel  ← the same topic, the same units, the same contract   /scan/fused
    ▼
  smart_planner_node   ← identical from here down
```

`smart_planner_node` receives a desired velocity and is responsible for
getting there without hitting anything.  It neither knows nor cares whether
that velocity came from a person-tracker or a path-follower.  That is what
makes the two modes interchangeable in any sense worth the word: not shared
code between two stacks, but **one stack with two front ends**.

## Why the serial port has exactly one owner

The original `smart_planner_node` opened `/dev/ttyARDUINO` itself.  That works
when the planner is the only thing that needs the Arduino.  It stops working
the moment you want the IMU, the sonars and the battery in ROS, because a
serial port has one reader and whoever gets there first owns it.

So the port moved to `serial_bridge_node`, and the planner gained an
`output_mode` parameter:

- **`topic`** (default) — the planner publishes a `Twist`; the bridge does the
  talking. This is the only arrangement in which sensor telemetry reaches ROS.
- **`serial`** — the planner opens the port itself, exactly as the original
  did. Useful for running the planner standalone, and for bisecting a problem
  down to "is it the bridge?".

`output_mode` is rejected at runtime by the parameter callback, because
changing who owns a file descriptor while both halves are running is not a
thing that can be made safe.

## Package map

| Package | Contains | Depends on |
|---|---|---|
| `a2r_bot_description` | URDF/xacro, RViz configs, `wheel_state_node` | — |
| `a2r_bot_arduino` | Firmware, `serial_bridge_node`, `firmware_monitor_node` | — |
| `a2r_bot_sensor_fusion` | `kalman.py`, IMU filter, EKF, scan fusion | — |
| `a2r_bot_follow` | `smart_planner_node` and the eight local planners | fusion, arduino |
| `a2r_bot_slam` | Mapping, costmap, global planners, path follower | fusion, follow |
| `a2r_bot_pet` | `robot_face.py`, `face_node`, `state_bridge_node` | — |
| `a2r_bot_bringup` | One launch file per mode | all |
| `a2r_bot_info` | This documentation | — |

The dependency arrows only ever point one way.  `a2r_bot_slam` depends on
`a2r_bot_follow` (it reuses the local planner); nothing depends on
`a2r_bot_slam`.  `a2r_bot_pet` depends on nothing, so the face can be
developed and run on a laptop with no robot attached.

## Layers, and what each is allowed to decide

| Layer | Decides | Explicitly does **not** decide |
|---|---|---|
| Jetson | Where to go, and why | How to get there |
| `global_planner_node` | The route across the map | Anything about obstacles it cannot see |
| `navigation_node` | Which way along the route | Obstacle avoidance |
| `smart_planner_node` | How to move right now without hitting anything | Where to go |
| `serial_bridge_node` | Nothing. It is a wire. | Everything |
| Firmware | Exact pulse widths, arming, failsafe | Anything geometric |

The one in the middle is worth stating explicitly, because getting it wrong is
the classic robotics bug: **`navigation_node` must not avoid obstacles.**  If
it did, there would be two controllers with an opinion about the steering, and
the visible symptom would be a robot that oscillates in doorways because one
layer pushes left while the other pushes right.  Its entire job is to look
ahead along the path and ask for that direction.  When the local planner
routes around something, the follower simply notices next cycle that it has
drifted, and aims back.
