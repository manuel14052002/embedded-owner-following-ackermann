# Testing the robot, stage by stage

Do not start with `make follow`. Each stage below proves one layer, so when
something breaks you know which layer it is. Skipping ahead means debugging the
whole stack at once, with the car on the floor.

**The whole way through: car on blocks, wheels off the ground.**

---

## Stage 0 — before any hardware

On your laptop, no robot needed:

```bash
make check      # firmware compiles, python syntax, yaml valid  (~2 s)
make test       # planner test suites                            (~60 s)
```

Both must pass before you flash anything.

---

## Stage 1 — firmware alone

No ROS. Just the board and a serial monitor.

```bash
make flash          # CAR ON BLOCKS
make monitor        # 921600
```

You should see the banner and the **I²C bus audit**, which names each device:

```
  0x0C  AK8963 magnetometer
  0x11  DYP-A21 front
  0x12  DYP-A21 rear
  0x40  PCA9685 #1 (servo ch0, ESC ch1)
  0x68  MPU9250 accel+gyro
  0x76  BMP280 barometer
  6 device(s), 0 problem(s)
```

**Fix the bus before going further.** Every later problem looks different and
is caused by this. See WIRING.md §9.8 for the symptom table.

Then type commands straight into the monitor:

```json
{"cmd":"PING"}
{"cmd":"CAL_IMU"}
{"cmd":"DRIVE","v":0,"s":0.3}
{"cmd":"DRIVE","v":0,"s":-0.3}
{"cmd":"STOP"}
```

Check as you go:

| Check | Expect |
|---|---|
| Display | `bAt`, then alternating percent and volts |
| Battery volts | within 0.1 V of a multimeter across the pack |
| `s: +0.3` | servo steers **left** (REP-103) |
| Telemetry lines | `{"ax":...}` streaming, `{"f":..,"r":..}` changing when you wave a hand at a sonar |

If the servo steers the wrong way, flip `SERVO_REVERSED` in `Config.h` — not
somewhere in the ROS stack.

Only when all of that is right:

```json
{"cmd":"DRIVE","v":0.15,"s":0}
```

The ESC should arm (a few seconds of neutral first) and the wheels spin.

---

## Stage 2 — the hardware layer in ROS

```bash
make build
make base          # serial bridge + fusion + lidar. No planners.
```

In a second terminal:

```bash
make preflight
```

This samples every topic for six seconds and prints a table. At this stage
expect everything green **except** `cmd_vel` and `cmd_out`, which have no
publisher yet — that is correct, nothing is planning.

If something is red, `preflight` names the **first broken link** rather than
listing symptoms.

Worth eyeballing by hand too:

```bash
ros2 topic echo /arduino/health_text     # "nominal"
ros2 topic echo /imu/rpy                 # tilt the robot, roll/pitch follow
ros2 topic hz /scan                      # ~10 Hz with the A2M12
ros2 topic echo /ultrasonic/front        # wave a hand in front of it
```

**If `/scan` is silent:** the A2M12 needs **256000** baud, not the A2M8's
115200. Check the lidar's motor is actually spinning.

---

## Stage 3 — look at it

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix a2r_bot_description)/share/a2r_bot_description/rviz/follow.rviz
```

Three displays earn their place immediately:

- **`/scan/fused`** — the merged scan the planner actually uses.
- **`/scan/sonar_overlay`** in a contrasting colour — the sonar contribution
  alone. Put a sheet of glass or a dark cloth in front of the robot: the lidar
  will miss it, the sonar will not, and you can watch the fusion catch it.
- **The robot model** — push the car by hand and the wheels should turn in
  RViz.

---

## Stage 4 — follow mode, on blocks, no Jetson

```bash
make follow
```

Nothing will move, because in follow mode the **Jetson** publishes `/cmd_vel`
and it is not running. That is expected. Stand in for it:

```bash
make wiggle        # sweeps the steering, no throttle
```

Watch the servo move and the wheels turn in RViz. Then:

```bash
make preflight
```

Now everything should be green, including `cmd_vel` and `cmd_out`.

Then drive it yourself:

```bash
make drive         # w/s/a/d, space to stop, q to quit, 1-8 switch planner
```

**Still on blocks.** Put your hand in front of the lidar while holding `w` —
`cmd_out` should drop toward zero even though `cmd_vel` still says forward.
That is the local planner doing its job, and it is the single most important
thing to confirm before the car touches the floor.

Press `1`–`8` to switch planners live and watch `/local_planner/active_planner`.

---

## Stage 5 — on the ground

Only after stage 4 passes.

```bash
make follow
make creep SPEED=0.2 SECONDS=5
```

Start at 0.2 m/s in an open space. Keep `make estop` ready in another
terminal — it publishes zeros immediately.

Walk a chair into its path and watch it steer around. Try the planners:

```bash
make follow PLANNER=fgm        # good in corridors and doorways
make follow PLANNER=rrt        # handles paths needing two opposite curves
```

---

## Stage 6 — with the Jetson

Stop `fake_jetson.py` first. Two things publishing `/cmd_vel` is exactly the
confusion this staging exists to avoid:

```bash
ros2 topic info /cmd_vel --verbose      # should show ONE publisher
```

Then start the Jetson's tracker. Everything below it is already proven.

---

## Quick reference

| Command | What it proves |
|---|---|
| `make check` | code compiles, no hardware |
| `make test` | planner algorithms are correct |
| `make doctor` | toolchain, devices, baud agreement |
| `make monitor` | firmware and the I²C bus |
| `make preflight` | every topic in the chain is alive |
| `make wiggle` | servo and steering, no throttle |
| `make drive` | full pipeline under your control |
| `make creep` | autonomous forward motion |
| `make estop` | stop now |

## If preflight shows a failure

It names the first broken link. The usual causes, in order of frequency:

| Broken link | Usual cause |
|---|---|
| `/arduino/healthy` | serial bridge not running, or `/dev/ttyARDUINO` missing |
| `/imu/data_raw` | baud mismatch — must be 921600 both ends |
| `/scan` | A2M12 at the wrong baud (needs 256000), or motor not spinning |
| `/imu/data` | `imu_filter_node` not running |
| `/odometry/filtered` | `ekf_state_node` not running |
| `/local_planner/cmd_out` | no `/cmd_vel` input, or `output_mode` is `serial` not `topic` |
