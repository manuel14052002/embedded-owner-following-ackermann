# Topic reference

Every topic in the workspace, who publishes it and who listens.  Topic names
in **bold** are the ones you will actually want to look at when something is
wrong.

Names marked *(param)* are configurable — the value shown is the default.

---

## 1. The spine

These four carry the robot.  If you understand nothing else, understand these.

| Topic | Type | From | To |
|---|---|---|---|
| **`/cmd_vel`** *(param)* | `geometry_msgs/Twist` | Jetson tracker **or** `navigation_node` | `smart_planner_node` |
| **`/scan/fused`** *(param)* | `sensor_msgs/LaserScan` | `scan_fusion_node` | `smart_planner_node`, `slam_node` |
| **`/odometry/filtered`** | `nav_msgs/Odometry` | `ekf_state_node` | everything that needs a pose |
| **`/local_planner/cmd_out`** *(param)* | `geometry_msgs/Twist` | `smart_planner_node` | `serial_bridge_node` |

`/cmd_vel` having two possible publishers is the whole interchangeability
design, not an accident.  In follow mode the Jetson publishes it; in
navigation mode `navigation_node` does.  `smart_planner_node` cannot tell the
difference, so all eight local planners work in both modes unchanged.

Units on `/cmd_vel` are real: `linear.x` in m/s, `angular.z` in rad/s.
Units on `/local_planner/cmd_out` are **normalised** to -1..1, because that is
what the firmware wants.  `serial_bridge_node` has an `input_is_normalised`
parameter so it can take either.

---

## 2. Hardware — `a2r_bot_arduino`

### Published

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/imu/data_raw` | `sensor_msgs/Imu` | ~50 Hz | Gyro and accel, **no orientation** — the quaternion is all zeros on purpose. Attitude is solved on the Pi. |
| `/imu/mag` | `sensor_msgs/MagneticField` | ~25 Hz | AK8963, remapped into the body frame. |
| **`/ultrasonic/front`** | `sensor_msgs/Range` | ~10 Hz | DYP-A21. `+inf` means no echo (REP-117), which is information, not a fault. |
| **`/ultrasonic/rear`** | `sensor_msgs/Range` | ~10 Hz | Same. |
| **`/battery`** | `sensor_msgs/BatteryState` | 1 Hz | `percentage` is 0..1. Comes from the firmware's LiPo curve, not a linear map. |
| `/baro` | `sensor_msgs/FluidPressure` | 2 Hz | BMP280. |
| `/arduino/status` | `std_msgs/String` | 1 Hz | Free-text firmware state. |
| `/arduino/failsafe` | `std_msgs/Bool` | on change | True when the firmware has cut the throttle. |
| **`/arduino/healthy`** | `std_msgs/Bool` | 2 Hz | *latched.* One answer to "can I drive?". |
| `/arduino/health_text` | `std_msgs/String` | 2 Hz | *latched.* Why not, when not. |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1 Hz | Standard. |

### Subscribed

| Topic | Type | Notes |
|---|---|---|
| `/local_planner/cmd_out` *(param)* | `geometry_msgs/Twist` | What to drive. |
| `/arduino/command` | `std_msgs/String` | Raw passthrough: `CAL_IMU`, `RESET_ENC`, `PING`, `STOP`. |
| `/pet/display_text` | `std_msgs/String` | Up to 4 characters on the MAX7219. |

---

## 3. Fusion — `a2r_bot_sensor_fusion`

| Topic | Type | From | Notes |
|---|---|---|---|
| **`/imu/data`** | `sensor_msgs/Imu` | `imu_filter_node` | Now **with** orientation. Yaw variance is inflated 50x when the magnetometer is off, because yaw is then unobservable and saying otherwise would mislead the EKF. |
| `/imu/rpy` | `geometry_msgs/Vector3Stamped` | `imu_filter_node` | Roll/pitch/yaw in radians, for plotting. |
| `/imu/yaw` | `std_msgs/Float32` | `imu_filter_node` | Degrees. Convenience only. |
| **`/odometry/filtered`** | `nav_msgs/Odometry` | `ekf_state_node` | Also publishes `odom` → `base_footprint` TF. |
| **`/scan/fused`** | `sensor_msgs/LaserScan` | `scan_fusion_node` | Drop-in replacement for `/scan`. |
| `/scan/sonar_overlay` | `sensor_msgs/LaserScan` | `scan_fusion_node` | The sonar contribution alone. Add it to RViz in a different colour when you want to see what the sonars are actually doing. |

Commands in: `/imu/calibrate` (`Bool`), `/ekf/reset` (`Bool`).

---

## 4. Local planning — `a2r_bot_follow`

| Topic | Type | Notes |
|---|---|---|
| **`/local_planner/cmd_out`** | `geometry_msgs/Twist` | Normalised throttle/steer. |
| `/local_planner/active_planner` | `std_msgs/String` | Which of the eight is running right now. |
| **`/local_costmap`** | `nav_msgs/OccupancyGrid` | 4 m rolling window, body frame. |
| `/local_planner/path` | `nav_msgs/Path` | Chosen trajectory. |
| `/local_planner/trajectories` | `visualization_msgs/MarkerArray` | Every candidate arc scored this cycle. The single most useful display in RViz when a planner is behaving oddly. |
| `/local_planner/best_traj` | `visualization_msgs/MarkerArray` | The winner, in green. |
| `/local_planner/rrt_tree` | `visualization_msgs/MarkerArray` | RRT only — the tree and the chosen path. A sampling planner is nearly impossible to debug from logs. |
| `/local_planner/state` | `visualization_msgs/Marker` | Text: FOLLOW, ESCAPE-PRESTEER, ESCAPE-REVERSE. |
| `/local_planner/desired_cmd` | `visualization_msgs/Marker` | What was asked for, before avoidance. |
| `/local_planner/heading_dev` | `std_msgs/Float32` | Accumulated drift from the commanded bearing. |

Commands in: `/bridge/command` (`String`).

> **One inconsistency, flagged rather than hidden.** In `output_mode: serial`
> the planner publishes `/battery_state`, `/wheel_encoder`, `/e_stop` and
> `/imu/data_raw` itself, because in that mode it owns the serial port. In the
> default `output_mode: topic` it publishes none of them — `serial_bridge_node`
> does, on `/battery` (note: not `/battery_state`). The two names differ
> because the old topic came from the original workspace and renaming it would
> have broken anything you already have subscribed. If you run in `serial`
> mode, subscribe to `/battery_state`; otherwise `/battery`.

---

## 5. SLAM — `a2r_bot_slam`

| Topic | Type | Notes |
|---|---|---|
| **`/map`** | `nav_msgs/OccupancyGrid` | *latched*, 1 Hz. |
| `/map_metadata` | `nav_msgs/MapMetaData` | *latched*. |
| **`/global_costmap`** | `nav_msgs/OccupancyGrid` | `/map` with obstacles grown by the robot radius. **Plan against this, never `/map`** — planning a point robot across the raw map produces routes through 20 cm gaps. |
| **`/slam/pose`** | `geometry_msgs/PoseStamped` | Scan-matched pose in the map frame. |
| `/scan_match/odom` | `nav_msgs/Odometry` | Same pose, with a covariance that reflects the match score, fed back to the EKF. |
| `/slam/match_score` | `std_msgs/Float32` | 0..1, fraction of beams landing on occupied cells. Below ~0.3 the match is not to be trusted. |
| `/slam/status` | `std_msgs/String` | Scans, known cells, rejects. |
| **`/plan`** | `nav_msgs/Path` | *latched*. The global route. |
| `/global_planner/status` | `std_msgs/String` | Algorithm, point count, length, milliseconds. |
| `/global_planner/markers` | `visualization_msgs/MarkerArray` | The route, in cyan. |
| `/navigation/status` | `std_msgs/String` | Index, distance to goal, lookahead, steer, cross-track error. |
| `/navigation/cross_track_error` | `std_msgs/Float32` | Signed metres off the path. Large values usually mean the local planner is routing around something, which is correct behaviour. |
| `/navigation/goal_reached` | `std_msgs/Bool` | *latched*. |
| `/navigation/lookahead` | `visualization_msgs/Marker` | The pure-pursuit target. |
| `/global_costmap/status` | `std_msgs/String` | Size, lethal cell count, inflation time. |
| `/map_save/result` | `std_msgs/String` | Where it went, or why it did not. |

Commands in: `/goal_pose` (`PoseStamped`, RViz's *2D Goal Pose*),
`/global_planner/cancel` (`Bool`), `/navigation/enable` (`Bool`),
`/slam/pause` (`Bool`), `/slam/reset` (`Bool`),
`/map_save` (`String` — a path), `/map_save/trigger` (`Bool`).

---

## 6. Face — `a2r_bot_pet`

| Topic | Type | Notes |
|---|---|---|
| **`/pet/state`** | `std_msgs/String` | **From the Jetson.** The FSM state. This package does not run a state machine; it only translates. |
| `/pet/face` | `std_msgs/String` | *latched*. The expression the bridge decided on. |
| `/pet/expression` | `std_msgs/String` | Direct override for 10 seconds. Bypasses the mapping — this is how you test the screen. |
| `/pet/current_expression` | `std_msgs/String` | *latched*. What is actually on screen. |
| `/pet/bridge_status` | `std_msgs/String` | State, expression, health, battery. |
| `/pet/quit` | `std_msgs/Bool` | Close the window. |

Valid expressions: `idle` `curious` `happy` `angry` `sad` `bored` `sleep`
`talking`.

---

## 7. Transforms

```
map ──(slam_node)──> odom ──(ekf_state_node)──> base_footprint
                                                      │
                                              (robot_state_publisher)
                                                      │
                     base_link ── lidar_link, imu_link, camera_link,
                                  ultrasonic_front_link, ultrasonic_rear_link,
                                  wheels, steering knuckles, decks, ...
```

The split matters.  `odom` → `base_footprint` is published continuously and
**smoothly** by the EKF, so anything that differentiates it — the local
planner, chiefly — sees no discontinuities.  `map` → `odom` is published by
SLAM and **jumps** whenever the scan matcher corrects the pose.  That is the
standard ROS arrangement and it exists precisely so that corrections never
appear as impossible accelerations in the robot's own frame.

Running SLAM and the EKF both publishing `odom` → `base_footprint` would give
you two publishers on one transform and a pose that visibly vibrates.  The
launch files do not do this; if you assemble your own, do not either.
