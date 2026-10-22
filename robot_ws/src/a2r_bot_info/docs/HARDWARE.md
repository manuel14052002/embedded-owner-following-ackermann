# Hardware

Full pin tables, resistor values, the **bench-test checklist** and the
bring-up order are in
[`a2r_bot_arduino/docs/WIRING.md`](../../a2r_bot_arduino/docs/WIRING.md).
This page is the summary.

> **Before you apply power to a newly built board, work through WIRING.md §9.**
> Eight tests, most of them a multimeter and two minutes. Test 9.1 (display
> polarity) is the one that can stop the project dead, so do it first.

## Bill of materials

| | Part | Notes |
|---|---|---|
| Chassis | Ackermann RC car, WR-01 class | wheelbase 0.257 m, track 0.250 m, width 0.31 m |
| ESC | HobbyWing 880 | PCA9685 **channel 1** |
| Servo | Standard steering servo | PCA9685 **channel 0** |
| PWM driver | PCA9685 | I²C `0x40`, 50 Hz |
| IMU | GY-91 (MPU9250 + AK8963 + BMP280) | I²C `0x68` / `0x0C` / `0x76` |
| Sonar ×2 | DYP-A21 **UART**, IP67, 5 m | on their own serial lines, not I²C — see WIRING.md §3.2 |
| Display | MAX7219 + FYQ-5641**?**S-11 4-digit | ⚠️ polarity — test WIRING.md §9.1 |
| Battery | Tattu TAA67504S25X9 | 4S LiPo, 14.8 V, 6750 mAh, 25C |
| Lidar | RPLIDAR A2M12 (sllidar_ros2, 256000 baud) | `/dev/ttyLIDAR` |
| MCU | Arduino Nano ESP32 | `/dev/ttyARDUINO` |
| SBC | Raspberry Pi 4 | runs this workspace |
| Vision | Jetson Nano | publishes `/cmd_vel` and `/pet/state` |
| Screen | 800×480 LCD | the face |

## Two corrections to the original plan

Both were in the brief as "everything on the shared A4/A5 I²C bus".  Neither
is possible, and no amount of firmware fixes either.

### The MAX7219 is not an I²C device

It is a **3-wire SPI-style shift register**.  No address, no ACK, no concept of
a bus — it clocks 16 bits in on every CS pulse and that is the entire protocol.
Wiring its DIN to SDA would inject garbage into the bus and break the IMU too.

**Resolved, and the built board does this correctly:** three dedicated GPIO
pins (DIN D11, CS D10, CLK D13), protocol bit-banged in `Battery.cpp`.

The board runs the MAX7219 at **5 V**, which is the right choice. ISET
(schematic R1) is **33 kΩ** — safely above the 9.53 kΩ datasheet floor, and the
right direction for a 4-digit scan, where each driver is on ¼ of the time and a
smaller RSET would run hot. `MAX7219_INTENSITY` is set to 12 to compensate for
the dimmer setting.

⚠️ **The display is labelled FYQ-5641BS-11.** In this family AS = common
cathode, BS = common **anode** — and the MAX7219 can only drive common cathode.
Verify with the diode-mode test in WIRING.md §9.1 before soldering the socket.

### "DYP-A21" is a family, not a part

| Variant | Interface | Can share A4/A5? |
|---|---|---|
| DYP-A21**IIC** | I²C, `0x11` / `0x12` | **Yes** |
| DYP-A21**UART** | 9600 8N1 | No |
| DYP-A21**V** | Analog voltage | No |
| DYP-A21**PWM** | Pulse width | No |

**The sonars now run on UART, not I²C.** The schematic wires J3/J19 to
SCL/SDA, so the board needs rewiring: each sonar's TX goes to its own Arduino
RX pin (front → D0/J23, rear → D2). Full details and the logic-level caveat in
WIRING.md §3.2.

This keeps two ~70 ms devices off the bus the servo and ESC depend on, and
removes the address collision entirely — two identical sonars on I²C ship at
the same address and one would have to be reprogrammed.

The firmware supports I²C, UART and analog behind one compile flag
(`ULTRASONIC_MODE` in `Config.h`) if you ever change sensors, and ROS sees
identical `sensor_msgs/Range` topics either way.

## Battery

4S LiPo: 16.80 V full, 14.00 V nominal-empty. The carrier board reads it
through a **47 kΩ / 10 kΩ** divider on A0 (ratio 5.700; 16.8 V presents 2.947 V).
`Config.h` matches. Use 1 % resistors — with 5 % parts the ratio error alone is
±7 %, larger than the entire 3.7–3.5 V region at the bottom of the curve.

Note 2.947 V is 89 % of the ESP32's 3.3 V full scale, and that ADC is
non-linear in its top ~10 %. If the reading tracks at half charge but drifts
near full, fit R3 = 8.2 kΩ rather than fighting it with `BATTERY_CAL_GAIN`,
which is a single multiplier and cannot correct a curve.

The percentage is **not** a linear voltage map.  A LiPo falls quickly from
4.20 to about 3.95 V per cell, sits on a long plateau, then collapses below
3.60 V.  A linear map reads 50% for most of the pack's life and then drops to
zero in two minutes.  The firmware uses a 21-point lookup of the real
discharge curve, plus throttle-proportional sag compensation — without which
the display drops 15% every time you accelerate and recovers when you stop,
which is accurate to the voltage and useless as information.

0% is set at **3.50 V/cell**, not 3.00 V.  Below 3.5 V under load a LiPo
permanently loses capacity; below 3.0 V it is scrap.  Zero on the display
means *land it now*.

## Power

Do **not** run the Pi or the Jetson from the ESC's BEC.  It is sized for a
receiver and one servo, not a 15 W SBC.  Give them their own regulator off the
pack.

Common ground is mandatory — Arduino, PCA9685, ESC, MAX7219 and the divider's
low side must all be the same node.  A floating divider ground is the most
common cause of a battery reading that drifts with throttle.

## I²C bus

Seven devices share A4/A5: two PCA9685s, the GY-91's three chips, and two
sonars. The carrier board fits 4.7 kΩ pull-ups (R12/R13) to 3V3_ardu.

**The bus runs at 100 kHz, not 400 kHz**, and that is arithmetic rather than
caution. I²C is open-drain, so a pull-up faces two opposing limits: it must be
above 967 Ω for a device to sink 3 mA and still hold below 0.4 V, and below a
ceiling set by rise time (`t_r ≈ 0.847·R·C`). With six devices on flying leads
(~400 pF) the 400 kHz ceiling is 885 Ω — *below* the floor. No pull-up value
works. At 100 kHz the window opens to 967 Ω – 2951 Ω, and the bus still only
runs at ~15 % utilisation.

**Do not reduce R12/R13 pre-emptively.** 4.7 kΩ is correct. The variable is how
many modules bring their own pull-ups; enough of them in parallel drags the
total under 1 kΩ and devices stop being able to pull the line low. That is the
classic "works with two modules, fails with the third". Measure it —
WIRING.md §9.5 has the decision table.
