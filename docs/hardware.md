# Hardware

## Parts

| Part | What I used |
|---|---|
| Chassis | 1:10 RC car with Ackermann steering (WR-01 class), brushed motor |
| Camera | Intel RealSense D435 on a 3D-printed pillar at the front |
| Lidar | Slamtec RPLIDAR A2M12 at the back of the top deck |
| Perception computer | NVIDIA Jetson (Orin Nano), 15 W mode |
| Planning computer | Raspberry Pi 4 |
| Microcontroller | Arduino Nano ESP32 on my carrier board |
| PWM driver | PCA9685 (servo on channel 0, ESC on channel 1) |
| ESC | HobbyWing WP880 |
| IMU | GY-91 (MPU-9250 + BMP280) |
| Battery | 4S LiPo, 6750 mAh |
| Optional | SIM7600 modem for the GPS fallback, small LCD for the face |

The two front sonars (DYP-A21) are mounted but disabled in the firmware at the moment.

## Carrier board

The KiCad project is in [`hardware/carrier_pcb_kicad`](../hardware/carrier_pcb_kicad). It's a simple two-layer, all through-hole board (about 91 × 110 mm) that the Nano ESP32 and a few modules plug into:

- battery input with a switch and a 1N5822 diode against reverse polarity;
- an LM2596 buck converter (off-board) for the 6 V servo rail, plus 5 V and 3.3 V regulator modules;
- the 5 V rail also feeds two USB-A sockets that power the Raspberry Pi;
- a 47 k / 10 k divider so the ESP32 can measure the battery voltage;
- screw terminals for the PCA9685, IMU and sonars, and a MAX7219 driving a 4-digit display.

<p align="center">
  <img src="../hardware/images/PCB_1%20(1).png" width="300" alt="PCB top">
  <img src="../hardware/images/PCB_1%20(2).png" width="300" alt="PCB bottom">
</p>

Schematic screenshots are in [`hardware/images`](../hardware/images). The more detailed pin-by-pin wiring is in [WIRING.md](../robot_ws/src/a2r_bot_arduino/docs/WIRING.md).

### Things I'd change in the next revision

- Use one pull-up voltage on the I²C bus. Right now the ESP32 side pulls up to 3.3 V and the PCA9685 breakout pulls up to 5 V, which puts the bus a bit above what the ESP32 likes.
- Power the ESP32 from the board instead of only from USB, and give the Pi its own 5 V supply.
- Add a fuse, a TVS diode and some local decoupling at the connectors, and stitch the two ground planes together with vias.

## Wiring overview

```
4S LiPo ─ switch ─ diode ─┬─ LM2596 ─ 6 V ─► PCA9685 V+ (servo)
                          ├─ 5 V module ──► Pi (USB-A), PCA9685 logic, display
                          ├─ 3.3 V module ► IMU, sonars
                          └─ divider ─────► ESP32 A0 (battery voltage)

ESP32 A4/A5 (I²C) ─► PCA9685, GY-91, sonars
ESP32 D0/D1 (UART) ─► host (JSON commands, 921600 baud)
ESP32 D10/D11/D13 ─► MAX7219 display
PCA9685 ch0 ─► steering servo     ch1 ─► ESC ─► motor
```
