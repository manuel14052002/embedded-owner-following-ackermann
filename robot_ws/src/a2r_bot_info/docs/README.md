# a2r_bot_info

Documentation for the A2R_bot workspace.  This package contains no code.

| Document | What is in it |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Build, flash, run, and the first checks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the three computers divide the work, and which layer decides what |
| [TOPICS.md](TOPICS.md) | Every topic, its type, and who publishes it |
| [HARDWARE.md](HARDWARE.md) | Bill of materials, and two corrections to the original wiring plan |
| [SERIAL_PROTOCOL.md](SERIAL_PROTOCOL.md) | The JSON link between the Pi and the Arduino |
| [SENSOR_FUSION.md](SENSOR_FUSION.md) | The Kalman filters, and why each measurement is weighted as it is |
| [PLANNERS.md](PLANNERS.md) | All twelve planners, with measurements rather than claims |
| [TESTING.md](TESTING.md) | Stage-by-stage bring-up: what to prove, in what order |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Symptoms, in rough order of frequency |

Full pin tables live in
[`a2r_bot_arduino/docs/WIRING.md`](../../a2r_bot_arduino/docs/WIRING.md).

## If you read one thing

[ARCHITECTURE.md](ARCHITECTURE.md), and specifically the table at the end of
it — which layer is allowed to decide what.  Most robotics bugs that are hard
to find are two layers deciding the same thing.
