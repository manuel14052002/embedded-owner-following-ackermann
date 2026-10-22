# Quickstart

## Build everything

The ROS workspace and the Arduino firmware are one project and build together:

```bash
cd ~/a2r_bot_ws
make first-run
```

> **Extracted from a .zip?** Run `make perms` first. Zip does not store the
> Unix execute bit. The `make` targets do not need it — they call bash and
> python3 explicitly — but running a script directly does.

That runs `rosdep`, `colcon build --symlink-install`, installs the Arduino
ESP32 core via `arduino-cli`, and compiles the firmware. On a fresh Pi it takes
a while, mostly downloading the core. Afterwards:

```bash
make            # rebuild the workspace (and re-check the firmware compiles)
make firmware   # firmware only
make check      # syntax + firmware, no hardware, about two seconds
make doctor     # toolchain, devices, and whether the baud rates agree
```

`--symlink-install` is used throughout, so edits to Python files take effect
without rebuilding — which on a Pi saves a great deal of waiting.

**A successful `make` means the firmware compiles too.** `build.sh` runs the
stub-header compile at the end, so you find out about a broken `Config.h` at
your desk rather than standing next to the car.

## Flash the firmware

```bash
make flash                       # uses /dev/ttyARDUINO
make flash PORT=/dev/ttyACM0     # before the udev rule is installed
make monitor                     # serial monitor at 921600
```

No `arduino-cli` yet? `make firmware-setup` installs it and the core. There are
**no libraries to install** — the firmware includes only `Arduino.h` and
`Wire.h`, and every driver is written out in the sketch.

If you would rather use the IDE: open
`src/a2r_bot_arduino/firmware/a2r_bot_firmware/a2r_bot_firmware.ino`,
select **Arduino Nano ESP32**, Upload.

> ⚠️ **If the IDE offers to "create a folder for this sketch and move it",
> say NO.** It moves only the `.ino` and leaves the other nine files behind,
> and you get `fatal error: Drive.h: No such file or directory`. The folder is
> already named correctly.
>
> You should see **ten tabs** across the top of the IDE window. One tab means
> the `.ino` is in the wrong place — see TROUBLESHOOTING.md.

**Do this with the car on blocks.**  An ESC that arms into a throttle command
will launch it off the bench.

**On a newly built carrier board, work through
[`WIRING.md` §9](../../a2r_bot_arduino/docs/WIRING.md) first.** Eight bench
tests, mostly a multimeter and two minutes each. Test 9.1 — display polarity —
is the one that can stop you dead, so start there.

Open the serial monitor at 115200.  You should see the banner and an I²C scan
listing `0x40` (PCA9685), `0x68` (MPU9250), `0x0C` (AK8963), `0x76` (BMP280)
and `0x11` / `0x12` (the sonars) — plus `0x41` if you populated the second
PCA9685.  Fix the bus before going further; every later problem looks different
and is caused by this.

## Stable device names

Serial devices swap between `/dev/ttyUSB0` and `/dev/ttyUSB1` on reboot, which
will eventually have you sending drive commands to the lidar.

```bash
sudo cp udev/99-a2r-bot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/ttyARDUINO /dev/ttyLIDAR
```

Edit the rule file first — the `idVendor`/`idProduct` values must match your
hardware.  Find them with `udevadm info -a -n /dev/ttyUSB0 | grep -m2 id`.

## Run something

```bash
# Hardware only — no planners. The right first step.
ros2 launch a2r_bot_bringup robot_base.launch.py

# Owner following
ros2 launch a2r_bot_bringup follow.launch.py
ros2 launch a2r_bot_bringup follow.launch.py planner:=rrt pet:=true

# Build a map
ros2 launch a2r_bot_bringup slam.launch.py
ros2 topic pub --once /map_save std_msgs/msg/String "{data: ~/maps/lab}"

# Navigate on it — then click "2D Goal Pose" in RViz
ros2 launch a2r_bot_bringup navigation.launch.py

# Just the face
ros2 launch a2r_bot_bringup pet.launch.py fullscreen:=false
```

## First checks, in order

```bash
ros2 topic echo /arduino/health_text   # should say "nominal"
ros2 topic echo /battery               # percentage should match a multimeter
ros2 topic echo /imu/rpy               # tilt the robot; roll/pitch should follow
ros2 topic hz /scan/fused              # ~10 Hz with the RPLIDAR A2M12
ros2 topic echo /odometry/filtered     # push the robot 1 m; check it says 1 m
```

If `/battery` disagrees with a multimeter, set `BATTERY_CAL_GAIN` in
`Config.h` to `measured / reported`.  Do it at roughly half charge, not full —
the divider's error is multiplicative and calibrating at the top of the range
exaggerates it.

## Looking at things

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix a2r_bot_description)/share/a2r_bot_description/rviz/follow.rviz
```

Three displays earn their place immediately:

- **`/local_planner/trajectories`** — every arc considered this cycle. When a
  planner behaves oddly, this shows you what it was choosing between.
- **`/scan/sonar_overlay`** — the sonar contribution alone. Put it in a
  different colour and you can see the glass door the lidar is missing.
- **`/local_costmap`** — what the planner thinks is solid, including inflation.

## Changing planners while driving

```bash
ros2 param set /local_planner_bridge planner fgm
ros2 param set /global_planner_node algorithm dijkstra
ros2 topic echo /local_planner/active_planner
```

## Verifying without hardware

All three test suites run on a laptop with nothing attached:

```bash
python3 scripts/test_rrt_offline.py        # the local RRT, 10 checks
python3 scripts/test_global_planners.py    # A*, Dijkstra, RRT, RRT*
python3 scripts/gen_rviz_configs.py        # regenerate the RViz configs
```
