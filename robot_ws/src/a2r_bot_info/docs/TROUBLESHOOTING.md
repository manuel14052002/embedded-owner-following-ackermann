# Troubleshooting

Ordered roughly by how often each one happens.

---

## `FileNotFoundError: src/a2r_bot_description/launch/...` from a fix script

The script uses paths relative to the workspace, so it has to run **from the
workspace root** — not from wherever you saved it.

```bash
cd ~/A2R_BOT/a2r_bot_ws          # the workspace, not ~/Downloads
python3 ~/Downloads/fix.py
```

The version shipped in `scripts/fix_humble_urdf.py` locates the workspace
itself, so it works from anywhere:

```bash
python3 ~/A2R_BOT/a2r_bot_ws/scripts/fix_humble_urdf.py
make fix-humble                  # same thing
```

---

## Board freezes entirely - serial stops, only a USB replug brings it back

Preceded by `"status":"WARNING slow loop"`, sometimes with `"stage":"imu"` (or
`ultrasonic`, or `baro`) naming which telemetry call was slow.

**Cause:** the ESP32's I2C library will, by default, block indefinitely if a
device holds the bus in a bad state mid-transaction (clock stretching with no
timeout). A marginal SDA/SCL connection - one that "mostly" works - can be
tipped from "answers slowly" to "never releases the bus" by something as small
as the vibration from the servo or ESC moving. Because the main loop calls
into Wire directly, when it blocks, EVERYTHING stops at once: serial output,
the drive-command failsafe, all of it. That is why unplugging the USB "fixes"
it - it is a genuine hang, not a slow response.

**Fixed in firmware:** `Wire.setTimeOut(25)` bounds every I2C call to 25 ms. A
stuck bus now costs one bad reading instead of a full freeze, and the loop
watchdog names which stage was slow so you know which device to reseat -
`imu` almost always means the GY-91's connection specifically, not the bus in
general.

**What to actually check**, in order of how often this turns out to be it:

1. **The physical connection to the device named in `stage`.** Reseat the
   header, check for a strained or partially-inserted jumper. Wiggle it BY
   HAND (power on, nothing moving) while watching `make hwtest-raw` - if data
   cuts in and out as you touch it, that is the wire.
2. **Vibration coupling.** If the freeze only happens during or right after
   the self-test (`t`), the servo/ESC vibration is disturbing an already
   marginal connection. Tape or a properly seated header fixes this even
   before you find the root wiring fault.
3. **Voltage sag under load.** Put a meter on the sensor's own VCC pin and
   watch it while `t` runs. A rail that dips when the ESC draws current can
   brown out a sensor just enough to corrupt one transaction.

---

## `Unable to parse the value of parameter robot_description as yaml`

A launch file exits immediately with this, and because it prints its `[INFO]`
lines first it can look like it started. It did not — nothing is running, so
`make preflight` then shows every topic silent and `make wiggle` moves nothing.

**Cause.** A `Command()` substitution hands launch a plain string, and launch
infers the parameter type by parsing it as YAML. A URDF is XML and begins
`<?xml version="1.0"?>`, which YAML rejects. ROS 2 Humble is stricter about
this than Foxy was, which is why older tutorials omit the fix.

**Fix** — already applied in `description.launch.py`. Declare the type instead
of letting launch guess:

```python
from launch_ros.descriptions import ParameterValue

robot_description = ParameterValue(
    Command(['xacro ', xacro_file, ' styling:=', styling]),
    value_type=str,
)
```

The URDF itself is fine. Ours expands to 123 links and 122 joints; the error
is purely about how launch typed the parameter.

---

## Everything is silent / nothing moves

`make preflight` shows every topic silent, or `make wiggle` prints its
messages and moves nothing.

**Check a launch file is actually running.** Both tools now say so
explicitly — `preflight` reports "NOTHING IS RUNNING AT ALL" rather than
blaming the serial bridge, and `fake_jetson.py` refuses to publish into an
empty graph.

A launch file that fails still prints `[INFO]` lines before dying. Look for
`[ERROR]` just above your prompt.

```bash
make base          # terminal 1 — leave it running
make preflight     # terminal 2
```

---

## `./scripts/....sh: Permission denied` (exit 126)

Zip archives do not carry the Unix execute bit, so a workspace extracted from
a `.zip` arrives with none of the scripts marked executable.

```bash
make perms
```

That restores them. Nothing else is needed — every `make` target now invokes
its script through `bash` or `python3` explicitly, so the execute bit is not
required at all. `make perms` exists only for running the scripts directly:

```bash
./scripts/preflight.py        # needs +x
python3 scripts/preflight.py  # works either way
```

If `make perms` is not available, the same thing by hand:

```bash
chmod +x scripts/*.sh scripts/*.py
```

Cloning with `git` instead of unzipping preserves the bits and avoids this
entirely.

---

## `fatal error: Drive.h: No such file or directory`

The Arduino IDE has moved the `.ino` into a subfolder of its own and left the
other nine files behind. The tell-tale is a doubled folder in the path:

```
...\firmware\a2r_bot_firmware\a2r_bot_firmware\a2r_bot_firmware.ino
                              ^^^^^^^^^^^^^^^^^^ one level too deep
```

The IDE compiles whatever sits *beside* the `.ino`, so from down there
`Drive.h` genuinely is not there.

**Fix** — move it back up and delete the empty folder:

```cmd
cd ...\firmware\a2r_bot_firmware
move a2r_bot_firmware\a2r_bot_firmware.ino .
rmdir a2r_bot_firmware
```

```bash
cd .../firmware/a2r_bot_firmware
mv a2r_bot_firmware/a2r_bot_firmware.ino .
rmdir a2r_bot_firmware
```

**How to tell it is right:** the IDE shows **ten tabs** across the top —
`a2r_bot_firmware`, `Battery.cpp`, `Battery.h`, `Config.h`, `Drive.cpp`,
`Drive.h`, `Protocol.cpp`, `Protocol.h`, `Sensors.cpp`, `Sensors.h`. One tab
means it is still misplaced.

`make firmware` and `make check` now detect this before compiling and print
the fix, and the `.ino` itself carries a `#error` with the same explanation.

Avoid it entirely by not using the IDE:

```bash
make flash
```

---

## Front and rear sonar report the same distance, or swap

Both sonars are wired to the same RX pin. The DYP-A21 frame carries no sensor
ID, so the firmware cannot tell them apart — it receives valid readings from
both and has no way to attribute them.

The firmware now refuses to compile in this configuration. If you hit the
`#error`, give the rear sonar its own pin: D2–D9 and D12 are free on header
J18, then set `PIN_US_REAR_RX` in `Config.h`.

Auto-report needs only **one** pin per sonar (its TX into our RX), so two
sonars need two pins, not four.

---

## The car does not move

**Is anything reaching the Arduino?**

```bash
ros2 topic hz /local_planner/cmd_out     # should be ~20 Hz
ros2 topic echo /arduino/health_text     # should say "nominal"
ros2 topic echo /arduino/failsafe
```

| Symptom | Cause |
|---|---|
| `cmd_out` silent | Nothing is publishing `/cmd_vel`. Check the Jetson, or `navigation_node`. |
| `cmd_out` present, car still | The ESC has not armed. See below. |
| `failsafe: true` | Firmware cut the throttle — command timeout, or the e-stop pin. |
| `health_text` names a silent sensor | That sensor is not reporting; the bridge is fine. |

**ESC will not arm.** It needs continuous neutral at power-up.  Either the
Arduino was already sending throttle when it powered on, or the ESC's own
neutral is not 1500 µs.  Power-cycle the ESC with the Arduino already running
and idle.  Continuous beeping with no arm is the signature.

**`Device or resource busy` on the serial port.**  Two processes tried to open
it.  Almost always `output_mode: serial` left set in `params.yaml` while
`serial_bridge_node` is also running.

```bash
sudo lsof /dev/ttyARDUINO       # find the other owner
ros2 param get /local_planner_bridge output_mode    # should be "topic"
```

---

## Steering goes the wrong way

Positive `s` must steer **left** (REP-103).  Fix it in `Config.h` with
`SERVO_REVERSED`, not by negating it somewhere in the ROS stack — the sign
convention should be true at the lowest level, or every layer above inherits
the confusion.

---

## Battery percentage is wrong

| Symptom | Cause |
|---|---|
| Consistently off by a few percent | Divider tolerance. Set `BATTERY_CAL_GAIN` = measured / reported, **at half charge** — the error is multiplicative and calibrating at full range exaggerates it. |
| Drops hard when you accelerate, recovers when you stop | Sag compensation too low. Raise `BATTERY_SAG_V_AT_FULL`. |
| Drifts with throttle even when parked | Floating ground on the divider. Check the common ground. |
| Reads 50% for ages then falls off a cliff | You are on a linear map, not the LUT. Check you flashed this firmware. |
| Wanders randomly | Missing the 100 nF cap from A0 to GND. |

---

## The display shows nothing, or garbage

**Nothing at all, ever, on a new board:** first suspect the display polarity.
The MAX7219 drives **common cathode only**, and a 5641**BS** part is common
anode. Run the diode test in WIRING.md §9.1 — it takes a minute and no amount
of firmware tuning works around a wrong part.

The MAX7219 is **not** on the I²C bus — it has three dedicated pins. If you
wired DIN to SDA, that is the problem, and it will also have broken the IMU.

**Dim but working:** expected with RSET = 33 kΩ. Raise `MAX7219_INTENSITY`
toward 15. If it is still too dim, check the chip is on 5 V and not 3.3 V —
the drive headroom is VDD − Vf, so 3.3 V roughly halves it.

**Never lower RSET below 9.53 kΩ** to brighten it. That is the datasheet floor
and going under it destroys the part.

Flickering or browning out: it switches eight digits at 800 Hz.  100 nF **and**
10 µF across VCC/GND, close to the chip.

---

## A sensor is missing from the I²C scan

The firmware scans at boot and prints every address.  Expect `0x40`, `0x68`,
`0x0C`, `0x76`, and `0x11`/`0x12` if you have the I²C sonars.

- **Nothing at all** — SDA/SCL swapped, or no common ground.
- **Works with two devices, fails with three** — too many pull-ups in
  parallel. Each breakout carries its own; enough of them drags the total
  below the 967 Ω floor and devices can no longer pull the line low. Measure
  SDA-to-3.3 V with the power off and compare against the table in
  WIRING.md §9.5. Do **not** reduce the board's R12/R13 — that makes it worse.
- **Addresses come and go intermittently** — the bus is too fast for the
  wiring. Confirm `I2C_CLOCK_HZ` is 100000 in `Config.h`. At 400 kHz with
  six devices on flying leads there is no pull-up value that satisfies both
  the sink-current floor and the rise-time ceiling.
- **`0x0C` missing** — the AK8963 is behind the MPU9250's pass-through, which
  the firmware enables. If the MPU9250 itself is missing, that is the real
  fault.
- **Only one of `0x11`/`0x12`** — the second sonar was never reprogrammed.
  Both ship at the same address. WIRING.md §9.6.
- **Neither `0x11` nor `0x12`** — you may have the UART or analog DYP-A21
  variant. Check the sticker and set `ULTRASONIC_MODE` in `Config.h`.

---

## Heading drifts, or follows the throttle

If yaw swings when the motor spins, the magnetometer is picking up the motor
or the ESC leads.  Move the GY-91, or:

```bash
ros2 launch a2r_bot_bringup robot_base.launch.py use_mag:=false
```

With the magnetometer off, yaw is relative and will drift slowly — that is
expected and fine, because the EKF's yaw reference then comes from the gyro
and scan matching.  The node inflates the published yaw variance 50x so the
EKF weights it correctly.

Slow drift with the magnetometer **on** usually means it is uncalibrated.
Spin the robot slowly through all orientations, log `/imu/mag`, fit an
ellipsoid, and set `mag_hard_iron` and `mag_soft_iron_scale`.  Uncalibrated,
expect a bearing-dependent error of 10–30°.

---

## Odometry drifts while stationary

Should not happen — the zero-velocity update exists to prevent exactly this.
Check `zero_velocity_snap: true` and that `/imu/data` is actually arriving:

```bash
ros2 topic hz /imu/data
ros2 topic echo /odometry/filtered --field twist.twist.linear.x
```

---

## The robot drives into glass doors

That is the failure `/scan/fused` exists to fix.  Confirm you are using it:

```bash
ros2 param get /local_planner_bridge scan_topic    # want /scan/fused
ros2 topic hz /scan/fused
ros2 topic echo /ultrasonic/front
```

If the sonars publish `inf` constantly, they are not detecting anything —
check the variant and the wiring.  Put `/scan/sonar_overlay` in RViz in a
contrasting colour to see the sonar contribution on its own.

---

## The robot oscillates in doorways

Two controllers are fighting over the steering.  In this workspace that should
be impossible — `navigation_node` deliberately does no obstacle avoidance —
so check nothing else is publishing `/cmd_vel`:

```bash
ros2 topic info /cmd_vel --verbose
```

If it is a single publisher, try a planner better suited to gaps:

```bash
ros2 param set /local_planner_bridge planner fgm
```

---

## The map is mirrored or rotated

An `OccupancyGrid`'s first row is the **lowest y**; a PGM's first row is the
**top**.  `map_saver_node` flips on the way out.  If you are loading with
something else, check it flips on the way in.  This is surprisingly hard to
notice on a symmetric building and then impossible to debug.

---

## SLAM pose jumps

```bash
ros2 topic echo /slam/match_score
```

Below ~0.3 the match is not trustworthy.  Causes: too few beams (raise
`match_max_beams`), a featureless corridor (no 2-D scan matcher solves this —
the EKF's gyro carries you through), or the initial guess being too far out
(check odometry first; SLAM cannot fix broken odometry).

Corrections larger than `match_max_correction_m` are rejected and logged.  If
you see that warning constantly, odometry is the problem, not SLAM.

---

## Global planner says "no path"

| Message | Meaning |
|---|---|
| `start is walled in` | Inflation has covered the robot's own position. It is too close to something. |
| `goal is unreachable` | No free cell within `snap_radius_cells` of where you clicked. |
| `endpoint out of bounds` | The goal is off the map. |

If the route exists but is refused, the usual cause is `unknown_is_free:
false` with the route passing through unobserved space.  That is the correct
default for navigation; set it `true` only for exploration.

Planning against `/map` instead of `/global_costmap` gives routes through gaps
narrower than the car.  The node warns when it falls back.

---

## The face does not appear

It is cosmetic and never takes the launch file down — it logs and continues.

```bash
ros2 topic echo /pet/current_expression
ros2 topic pub --once /pet/expression std_msgs/msg/String "{data: happy}"
```

Over SSH with no desktop, set in `pet.yaml`:

```yaml
sdl_videodriver: "kmsdrm"     # or "fbcon" on older kernels
sdl_fbdev: "/dev/fb0"
```

An unmapped state shows `curious` and warns once.  Add it to `state_names` /
`state_expressions` in `pet.yaml`.

---

## Everything is slow

```bash
ros2 topic hz /local_planner/cmd_out   # want ~20 Hz
ros2 topic hz /scan/fused              # want ~7 Hz
top -H -p $(pgrep -f smart_planner)
```

RRT is the most expensive local planner.  Lower `rrt_iterations` first — it
has the gentlest quality gradient.  On the global side, A* is 12–118x cheaper
than Dijkstra for an identical answer, and RRT* costs ~38x RRT for a result
that is within 1.1% after smoothing.
