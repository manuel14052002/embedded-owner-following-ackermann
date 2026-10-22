# a2rbot_pet_car — Setup & Test Guide

Jetson-to-ESP32 person-following robot. The Jetson runs camera + AI + tracking
and sends movement commands to the ESP32 over USB serial as newline-delimited
JSON. The ESP32 drives a BTS7960 throttle and a PCA9685 steering servo.

```
D435 → camera_node → /camera/color+depth → tracker_node → /cmd_vel
                                                              ↓
                                          serial_bridge_node (Ackermann + watchdog)
                                                              ↓ USB JSON
                                                            ESP32 → BTS7960 + servo
                                          ESP32 telemetry ← /imu /battery /encoder
```

## 1. Prerequisites (Jetson, ROS2 Humble)

```bash
# ROS2 deps
sudo apt install ros-humble-sensor-msgs ros-humble-geometry-msgs

# Python deps
pip install pyserial ultralytics
pip install pyrealsense2          # or build librealsense from source on Jetson

# Verify camera is seen
rs-enumerate-devices | grep D435
```

## 2. Flash the ESP32

1. Open `firmware/a2r_bot_drive.ino` in Arduino IDE.
2. Board: **Arduino Nano ESP32** (or your ESP32 board).
3. Install libraries via Library Manager:
   - `Adafruit PWM Servo Driver Library`
   - `ArduinoJson` (v6.x)
4. Upload. Open Serial Monitor @115200 — you should see:
   ```
   {"status":"MPU6050 OK"}
   {"status":"IMU calibration done"}
   {"status":"a2r_bot_drive ready"}
   ```
   Keep the robot **still** during boot (IMU auto-calibrates).

## 3. Build the ROS2 package

```bash
cd ~/ros2_ws/src
cp -r /path/to/a2rbot_pet_car .
cd ~/ros2_ws
colcon build --packages-select a2rbot_pet_car
source install/setup.bash
```

## 4. Find your wheelbase & speed (needed for correct steering)

- **Wheelbase**: measure front axle centre → rear axle centre, in metres.
  Set it: `wheelbase_m:=0.30` (example).
- **Max speed**: with the robot on blocks, send full throttle briefly and
  estimate wheel surface speed, or just start with the 1.2 m/s default and
  tune. This only affects steering scaling, not safety.

## 5. FIRST TEST — dry run (no motors move)

Always start here. The bridge logs the JSON it *would* send:

```bash
ros2 launch a2rbot_pet_car pet_car.launch.py dry_run:=true
```

Stand in front of the camera. You should see in the bridge terminal:
```
[DRY] → {"cmd":"DRIVE","v":0.15,"s":-0.22}
```
Walk left/right → `s` changes sign. Walk closer than 1.2 m → `v` goes to 0 or
negative-clamped-to-0. Step out of frame → `{"cmd":"STOP"}` within 1 s.

Verify topics:
```bash
ros2 topic echo /cmd_vel
ros2 topic echo /robot_state       # IDLE → FOLLOWING → LOST
ros2 topic echo /owner_distance
```

## 6. SECOND TEST — wheels up

Put the robot **on blocks** so wheels spin free. Real motors, no driving away:

```bash
ros2 launch a2rbot_pet_car pet_car.launch.py max_linear:=0.3
```

Confirm: wheels spin forward when you back away, steering turns toward you when
you move sideways, everything stops when you leave frame or Ctrl+C.

## 7. THIRD TEST — on the ground, low speed

Clear a safe space. Keep the e-stop (power cut) within reach:

```bash
ros2 launch a2rbot_pet_car pet_car.launch.py max_linear:=0.4 wheelbase_m:=0.30
```

Increase `max_linear` gradually (0.4 → 0.6 → 0.8) once behaviour is stable.

## 8. Manual commands

```bash
# Reset encoder / recalibrate IMU (hold robot still)
ros2 topic pub --once /bridge/command std_msgs/String "data: RESET_ENC"
ros2 topic pub --once /bridge/command std_msgs/String "data: CAL_IMU"

# Force tracker on/off
ros2 topic pub --once /robot_state_cmd std_msgs/String "data: STOP"
ros2 topic pub --once /robot_state_cmd std_msgs/String "data: FOLLOW"
```

## Safety summary

| Layer | Protection |
|-------|-----------|
| Tracker | Output clamped to `max_linear`/`max_angular`; forward-only; zero on lost target |
| Bridge | Zero Twist → STOP; no cmd in 0.3 s → STOP; STOP on shutdown (×2); dry-run mode |
| ESP32 | 500 ms watchdog auto-stop; over-current cutoff; e-stop flag; steering clamped |

Every layer fails to STOP, never to "keep moving". Always test dry-run → blocks
→ ground, and start with low `max_linear`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No ESP32 port found` | Set `port:=/dev/ttyACM0` explicitly; check `ls /dev/ttyACM* /dev/ttyUSB*` |
| Servo works, motor doesn't | LEDC PWM already fixed in firmware; check BTS7960 R_EN/L_EN wired HIGH, VBUS power |
| Steering reversed | Negate `s` in bridge `_cmd_cb`, or flip servo horn |
| Turns too sharp/shallow | Tune `wheelbase_m` and `max_steer_deg` to match your geometry |
| Robot creeps when stopped | Widen `deadband_dist`/`deadband_bear` in tracker |
| Camera not found | Build librealsense with `-DBUILD_PYTHON_BINDINGS=true`; check USB3 port |
