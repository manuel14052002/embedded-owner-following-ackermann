# A2R_bot — wiring and bring-up reference

Validated against the carrier-board schematic (`simplied_circuit_design.kicad_sch`)
and the firmware in `../firmware/a2r_bot_firmware/`.

If a device is not at the address or pin listed here, change `Config.h` — not
the code.

> **Read §9 before you apply power.** Four of those tests take a multimeter and
> two minutes each, and each one catches a fault that is expensive or
> impossible to diagnose once the board is fully populated and running.

---

## 1. Two corrections to the original plan

Both were specified as "everything on the shared A4/A5 I²C bus". Neither is
possible, and no amount of firmware fixes either.

### 1.1 The MAX7219 is not an I²C device

The MAX7219 is a **3-wire SPI-style shift register**: no address, no ACK, no
concept of a bus. It clocks 16 bits in on every CS pulse and that is the whole
protocol. Wiring its DIN to SDA would inject garbage into the I²C bus and break
the IMU as well.

**Resolved, and the built board does this correctly.** Three dedicated GPIO
pins, protocol bit-banged in `Battery.cpp`, so it does not compete with the
hardware SPI peripheral either.

| MAX7219 pin | Arduino pin | `Config.h` symbol | Schematic |
|---|---|---|---|
| DIN  | D11 | `PIN_MAX7219_DIN` | ✅ D11 |
| LOAD/CS | D10 | `PIN_MAX7219_CS` | ✅ D10 |
| CLK  | D13 | `PIN_MAX7219_CLK` | ✅ D13 |
| VCC  | **5 V** | — | confirmed |
| GND  | GND | — | |

The board runs the MAX7219 at **5 V**, which is the right choice — it drives
the LEDs to full brightness and puts the logic-high threshold question to bed.

> **If you ever move it to 3.3 V:** the MAX7219's logic-high threshold at
> VCC = 5 V is 3.5 V, so 3.3 V logic into a 5 V part is *marginal* — it usually
> works and then fails on a hot day. The clean fix is a 74AHCT125 level shifter
> on DIN, CS and CLK. On a classic 5 V Nano none of this applies.

### 1.2 "DYP-A21" is a family, not one part

| Variant | Interface | Can share A4/A5? |
|---|---|---|
| DYP-A21**IIC** | I²C, addresses 0x11 / 0x12 | **Yes** |
| DYP-A21**UART** / **TTL** | 9600 8N1 | No |
| DYP-A21**V** | Analog voltage | No |
| DYP-A21**PWM** | Pulse width | No |

### ⚠️ The sonars have moved OFF the I²C bus

The schematic wires J3 and J19 to SCL/SDA, but the firmware now runs them on
**UART** (`ULTRASONIC_MODE US_MODE_UART`). **The board must be rewired for
this** — see §3.2.

It is the better arrangement regardless of what the schematic says:

- The I²C bus no longer carries two devices that each need ~70 ms of
  time-of-flight to answer. The PCA9685 and the IMU get it to themselves, and
  the IMU is the one device here with a real deadline.
- A sonar that wedges its line can no longer take down the bus the **servo and
  ESC** depend on. Steering keeps working.
- No address collision to manage: two identical sonars on I²C ship at the same
  address and one has to be reprogrammed. On separate UARTs they simply cannot
  collide.
- Each sensor free-runs, so there is no trigger-then-wait round trip at all.

The firmware supports all three useful variants behind one compile-time switch
if you ever change sensors:

```c
#define ULTRASONIC_MODE   US_MODE_I2C     // default: shared A4/A5 bus
// #define ULTRASONIC_MODE   US_MODE_UART  // two SoftwareSerial ports
// #define ULTRASONIC_MODE   US_MODE_ANALOG // two ADC pins
```

Only the bottom 40 lines of `Sensors.cpp` change; ROS sees identical
`sensor_msgs/Range` topics either way.

---

## 2. Power

```
                 Tattu TAA67504S25X9
                 4S1P LiPo, 14.8 V nom, 6750 mAh, 25C
                 (16.80 V full  →  14.00 V empty)
                          │
              ┌───────────┴───────────┐
              │                       │
        J16 CENTRAL_IN          Voltage divider → A0
        → SW4 → D1 (1N5822)           │
        → J12 LM2596_IN               └── see §5
              │
        J7 LM2596_OUT → D2 → C3 470µF → 6 V rail
              │
              ├── J2 / J4      6V_I  / 6V_II   (servo rails)
              ├── J8 / J9      5V_I  / 5V_II
              └── J11 / J13    3V3_I / 3V3_II
```

**Do not** power the Raspberry Pi or the Jetson from the ESC's BEC — it is
sized for a receiver and one servo, not a 15 W SBC.

⚠️ **If J24/J25 (the USB-A outputs) feed the Pi or Jetson**, check the LM2596's
current headroom first. An LM2596 module is typically good for 2–3 A with
adequate heatsinking; a Pi 4 alone wants up to 3 A, and a Jetson Nano more. Two
SBCs off one LM2596 will brown out under load, and a Pi that browns out mid-SLAM
corrupts its SD card.

**Common ground is mandatory.** Arduino GND, PCA9685 GND, ESC GND, MAX7219 GND
and the divider's low side must all be the same node. A floating ground on the
divider is the single most common cause of a battery reading that drifts with
throttle.

---

## 3. I²C bus (A4 = SDA, A5 = SCL)

| Device | Address | Header | Notes |
|---|---|---|---|
| PCA9685 #1 | `0x40` | J5 (PCA9685_I) | servo ch0, ESC ch1 |
| PCA9685 #2 | `0x41` | J14 (PCA9685_II) | **A0 jumper must be bridged** |
| MPU9250 (GY-91) | `0x68` | J10 | `0x69` if AD0 pulled high |
| AK8963 (GY-91) | `0x0C` | J10 | behind MPU9250 pass-through |
| BMP280 (GY-91) | `0x76` | J10 | `0x77` on some clones |


**The two DYP-A21 sonars are no longer on this bus** — see §3.2.

The board fits **R12 / R13 = 4.7 kΩ** pull-ups to 3V3_ardu. That is a good
value — see §9.5 before changing anything.

### 3.2 Ultrasonics on UART — what to rewire

Each DYP-A21 in auto-report mode **talks and never listens**, so it needs
exactly **one** Arduino pin: its TX into our RX. Power and ground are
unchanged.

| Sonar | Sonar pin | Goes to | Arduino pin | `Config.h` |
|---|---|---|---|---|
| **Front** | TX | board header **J23 / RX** | D0 | `PIN_US_FRONT_RX` |
| | VCC | 5 V | | |
| | GND | GND | | |
| **Rear** | TX (via 1k/2k divider) | **J18** | D4 | `PIN_US_REAR_RX` |
| | RX (query mode only) | **J18** | D5 | `PIN_US_REAR_TX` |
| | VCC | 5 V | | |
| | GND | GND | | |

**Disconnect their SDA and SCL lines from J3 and J19.** Leaving an I²C-variant
sonar on the bus while the firmware is in UART mode gives you a device that
answers on I²C and is never read — the boot audit now warns about exactly this.

### ⛔ The two sonars cannot share one RX pin

This looks like a reasonable saving and it is not, so it is worth being
explicit about why. The DYP-A21 auto-report frame is:

```
0xFF, dist_HI, dist_LO, checksum
```

Note what is **not** in it: any sensor identity. Every frame from every sonar
is indistinguishable. Wire both transmitters to one RX pin and you receive a
stream of perfectly valid, correctly-checksummed distances with **no way to
know which sensor each came from**. No parsing recovers it — the information
was never transmitted.

Assigning them alternately (front, rear, front, rear…) is right only while two
free-running crystals stay in step, which they do not. The moment one sensor
drops a frame, front and rear **swap and stay swapped**. The robot then brakes
for things behind it and drives into things in front of it.

They also physically collide. Two sensors reporting at ~100 ms on independent
clocks drift through each other every few seconds; during a collision both
drive the line at once, and some corrupted frames still pass an 8-bit
checksum.

**The firmware refuses to build in this configuration** — `Config.h` has a
`#error` if `PIN_US_FRONT_RX == PIN_US_REAR_RX`.

**The fix is one jumper wire.** Auto-report needs only ONE pin per sonar, so
two sonars need **two pins, not four**. D2–D9 and D12 are all free and all
broken out on header **J18**. Move the second sonar's TX to any of them and set
`PIN_US_REAR_RX` to match.

If you genuinely cannot spare a second pin:

| Option | Cost |
|---|---|
| Fit one sonar, set `US_REAR_ENABLED 0` | lose rear detection; the topic reports honestly rather than guessing |
| 74HC4052 mux or two 74HC125 buffers on the TX lines, selected from a GPIO | a part **and a pin** — the pin you were trying to save |

On the ESP32 any GPIO can carry a UART, so D2 is not a special choice — change
`PIN_US_REAR_RX` to whatever suits your loom. Avoid D10, D11 and D13
(MAX7219), A0 (battery) and A4/A5 (I²C).

**Logic levels.** The DYP-A21's TX idles at its own VCC. Powered from 5 V it
presents 5 V to a 3.3 V ESP32 input, which is out of spec. Either power the
sonar from 3.3 V if your variant allows it (many do — check the label), or put
a divider on its TX line: **1 kΩ in series, 2 kΩ to ground**. A sonar is far
too slow for the divider to cost anything.

**Frame format**, for when you are staring at a logic analyser: 9600 8N1, four
bytes about every 100 ms — `0xFF, dist_HI, dist_LO, checksum`, where
`checksum = (0xFF + HI + LO) & 0xFF` and the distance is in **millimetres**.

**Crosstalk.** In auto-report mode both sensors fire on their own internal
timers and the firmware cannot interleave them the way the I²C path did. Two
transducers at similar rates will occasionally hear each other. That is what
`US_MAX_JUMP_M` is for: a large jump has to repeat before it is believed. If
it becomes a real problem, set `US_UART_AUTO_REPORT 0` — the firmware then
queries each sensor alternately and they are never in flight together, at the
cost of needing the TX pins wired too.

### 3.1 Bus speed: 100 kHz, deliberately

`I2C_CLOCK_HZ` in `Config.h` is **100000**, reduced from 400 kHz. This is not
conservatism, it is arithmetic.

I²C is open-drain: nothing drives the line high, a pull-up resistor does. That
creates two limits that pull in **opposite** directions:

**Floor — sink current.** A device must sink 3 mA and still hold below 0.4 V:

```
R > (3.3 − 0.4) / 3 mA = 967 Ω
```

Below this, a device physically cannot pull the line low enough to register as
a logic 0. Adding pull-ups in parallel only ever lowers the total, which is why
"works with two modules, fails with the third" is such a common fault.

**Ceiling — rise time.** `t_r ≈ 0.847 × R × C`, and it must beat 300 ns at
400 kHz or 1000 ns at 100 kHz.

With six devices on flying leads the bus sits around 400 pF. At 400 kHz that
gives a ceiling of 885 Ω — **below** the 967 Ω floor. The window is empty; no
pull-up value works. At 100 kHz the window opens to 967 Ω – 2951 Ω.

**And 100 kHz costs nothing here.** Against the real traffic — MPU9250 and
AK8963 at 50 Hz, BMP280 at 2 Hz, both sonars at 10 Hz, PCA9685 at 20 Hz — the
bus runs at about **15 % utilisation**.

Only raise it to 400 kHz if the sensors are on short leads (under ~250 pF)
*and* you have measured total pull-up resistance into the 1.0k–1.4k window.
Otherwise you get intermittent NACKs that look exactly like flaky wiring.

---

## 4. PCA9685 outputs

Channel assignment is fixed in `Config.h`:

| Channel | Device | Neutral | Range |
|---|---|---|---|
| **0** | Steering servo | 1500 µs | 1000–2000 µs, limited to ±85 % travel |
| **1** | HobbyWing 880 ESC | 1500 µs | 1000–2000 µs, forward capped at 70 % |

PWM frequency is 50 Hz, with the prescaler computed from the 25 MHz internal
oscillator rather than trusting a library default.

`SERVO_TRAVEL_LIMIT = 0.85` exists because a servo commanded to its mechanical
stop against Ackermann linkage will stall, draw 2 A and cook itself. Raise it
only after confirming the linkage reaches lock without binding.

**ESC arming.** The HobbyWing 880 will not arm unless it sees neutral for a
continuous period at power-up. The firmware holds channel 1 at exactly 1500 µs
for `ESC_ARM_MS` (3000 ms) and refuses all drive commands until then. If the
ESC beeps continuously and never arms: either the Arduino was already sending
throttle at power-up, or the ESC's neutral is not 1500 µs.

**Reverse.** Car ESCs use a brake-then-reverse handshake, not a simple negative
throttle. The firmware implements it: brake pulse → neutral gap → reverse.
Commanding reverse directly just brakes.

### 4.1 ⚠️ The 6 V question

J5 and J14 provide only **GND / SCL / SDA / 6V**. Before connecting either
board, establish where that 6 V lands — see test 9.4. The PCA9685's VDD is
rated 2.3–5.5 V recommended, 6.0 V absolute maximum, so:

- 6 V → **V+** (the servo rail): correct, and VCC must be fed separately.
- 6 V → **VCC** (logic): right at absolute maximum, and if the module's own
  pull-ups tie to VCC, your 3.3 V Nano ESP32 I²C pins will see 6 V.

---

## 5. Battery sensing (A0)

### 5.1 The divider as built

```
  BATT+ ──[ R2 47 kΩ ]──┬──[ R3 10 kΩ ]── GND
                        │
                        └──► A0        ratio = 57/10 = 5.700
                                       16.8 V → 2.947 V   ✓
```

`Config.h` matches this: `DIVIDER_R1_OHMS 47000`, `DIVIDER_R2_OHMS 10000`.

> **Naming clash, worth knowing.** The schematic calls these R2 and R3. The
> firmware calls them R1 (upper leg) and R2 (lower leg), which is the
> convention the divider formula uses. They do not line up. Do not "correct"
> one to match the other.

Add a 100 nF capacitor from A0 to GND. It costs nothing and removes most of the
switching noise the ESC injects into everything nearby.

Use 1 % resistors. With 5 % parts the ratio error alone is ±7 %, which is
larger than the entire 3.7–3.5 V region at the bottom of the discharge curve.

### 5.2 ADC headroom

2.947 V is **89 % of the ESP32's 3.3 V full scale**. That is fine on a classic
AVR, but the ESP32-S3's ADC is noticeably non-linear in its top ~10 %, and
`BATTERY_CAL_GAIN` is a single multiplier that cannot correct a curve.

If the reading tracks well at half charge but drifts near full, that is this
effect. Fit **R3 = 8.2 kΩ** (giving 2.53 V at 16.8 V) rather than fighting it
in software.

### 5.3 Calibration

Measure the pack with a multimeter, compare to what the firmware prints, then
set:

```c
#define BATTERY_CAL_GAIN   1.000f   // measured / reported
```

Do this **at roughly half charge**, not full — the divider's error is
multiplicative and calibrating at the top of the range exaggerates it.

### 5.4 Why the percentage is not a linear map

A LiPo's voltage-vs-charge curve is emphatically not a straight line. It falls
quickly from 4.20 V to about 3.95 V per cell, sits on a long flat plateau, then
collapses below 3.60 V. A linear map reads "50 %" for most of the pack's life
and then drops to zero in two minutes.

The firmware uses a 21-point lookup of the real discharge curve with linear
interpolation, plus throttle-proportional sag compensation
(`BATTERY_SAG_V_AT_FULL`) — without which the display drops 15 % every time you
accelerate and recovers when you stop, which is accurate to the voltage and
useless as information.

The reading is median-filtered over 15 samples then EMA-smoothed, and the
displayed percentage is rate-limited upward (a battery does not recharge while
you drive) but allowed to fall freely.

### 5.5 Thresholds

| Symbol | Value | Meaning |
|---|---|---|
| `CELL_FULL_V` | 4.20 V | 100 % |
| `CELL_EMPTY_V` | 3.50 V | 0 % on the display |
| `CELL_CRITICAL_V` | 3.60 V | display flashes `LO` |

0 % is set at 3.50 V/cell, not 3.00 V. Taking a LiPo below 3.5 V under load
permanently reduces its capacity; below 3.0 V it is scrap. Zero on the display
means *land it now*, not *keep going*.

---

## 6. MAX7219 → FYQ-5641 display

### 6.1 ISET resistor (schematic R1 = 33 kΩ)

**33 kΩ is safe and appropriate.** The datasheet's absolute minimum RSET is
9.53 kΩ, and going *below* that is what destroys the part. 33 kΩ is comfortably
above it, so the only effect is brightness — not damage.

It is also the right direction for this design. `SCANLIMIT` is set to 3 (four
digits), so each digit driver is on ¼ of the time instead of ⅛. The datasheet
warns that scanning few digits raises per-driver dissipation and that RSET
should be increased to compensate. A 10 kΩ RSET driving only four digits is the
case that runs hot; 33 kΩ is not.

The cost is a dimmer display, so `MAX7219_INTENSITY` is set to **12** to claw
some back. Raise to 15 if it is still hard to read in daylight — the duty-cycle
control costs nothing and cannot exceed the peak current RSET has already set.

> Exact segment current depends on RSET *and* your display's forward voltage —
> read it off Table 11 of the MAX7219 datasheet for your specific part rather
> than trusting a remembered figure. The number that matters is the 9.53 kΩ
> floor, and you are well clear of it.

### 6.2 Pin mapping

The FYQ-5641 is a 4-digit, 12-pin seven-segment display. **The MAX7219 can only
drive common-cathode displays** — see test 9.1.

| Display pin | Function | MAX7219 pin | MAX7219 name |
|---|---|---|---|
| 1  | Segment E       | 20 | SEG E |
| 2  | Segment D       | 23 | SEG D |
| 3  | Decimal point   | 22 | SEG DP |
| 4  | Segment C       | 21 | SEG C |
| 5  | Segment G       | 15 | SEG G |
| 6  | **Digit 4** (cathode) | 10 | DIG 3 |
| 7  | Segment B       | 16 | SEG B |
| 8  | **Digit 3** (cathode) | 11 | DIG 2 |
| 9  | **Digit 2** (cathode) | 7  | DIG 1 |
| 10 | Segment F       | 14 | SEG F |
| 11 | Segment A       | 17 | SEG A |
| 12 | **Digit 1** (cathode) | 2  | DIG 0 |

Digit ordering: display pin 12 is the leftmost digit and maps to DIG 0, so the
display reads left-to-right as DIG 0 → DIG 3. The firmware's `showText` writes
in that order, so what you pass is what you see. If it comes out mirrored, set
`DISPLAY_DIGITS_REVERSED 1`.

**Decoupling.** The board fits C1 100 µF and C2 100 nF — correct. The MAX7219
switches eight digits at 800 Hz and will brown itself out on long leads without
them.

### 6.3 What you will see

| Display | Meaning |
|---|---|
| `bAt` | boot self-test, briefly |
| `  87P` → `16.8` | alternating: percent, then pack volts |
| `LO` flashing | below `CELL_CRITICAL_V` |
| `----` | no valid reading yet |
| `E 1` | firmware failsafe active |

---

## 7. GY-91 orientation

The firmware remaps the MPU9250's axes into the ROS body frame (REP-103: **X
forward, Y left, Z up**) and separately remaps the AK8963, which ships in a
*different* frame from the accelerometer on the same die — a genuine hardware
quirk of the MPU9250, not a mistake.

Mount the GY-91 with its silkscreen X arrow pointing **forward** and the
component side **up**. If you mount it differently, fix it in `Config.h`:

```c
#define IMU_AXIS_REMAP_X   ( ax)
#define IMU_AXIS_REMAP_Y   (-ay)
#define IMU_AXIS_REMAP_Z   ( az)
```

Mount it **away from the ESC and the motor leads**. The magnetometer is the
sensitive one — a motor 3 cm away will swamp the Earth's field entirely and
your heading will follow the throttle instead of north. If you cannot get it
far enough away, run with `use_mag:=false` and let the gyro and scan matching
carry the yaw.

**Calibration.** Send `{"cmd":"CAL_IMU"}` with the robot stationary and level.
The firmware averages the gyro for two seconds and discards the average if it
detects motion partway through, rather than storing a bad one.

---

## 8. Optional e-stop

If `ENABLE_ESTOP_PIN` is set to 1 in `Config.h`, `PIN_ESTOP` (D2) becomes an
active-low emergency stop: pulling it to ground forces the ESC to neutral and
latches failsafe until released. A normally-closed mushroom switch to GND is
the right part. The state is echoed to ROS on `/arduino/failsafe`.

D2 is broken out on J6 and J18 and is otherwise unused.

---

## 9. Bench tests — do these BEFORE applying power

Each of these catches something that is cheap to find now and painful to find
later. Tests 9.1 to 9.5 need only a multimeter and an unpowered board.

### 9.1 Display polarity — **fit the -AS, not the -BS**

Your schematic labels U2 as **FYQ-5641BS-11**. In this family the suffix *is*
the polarity:

| Part | Common | MAX7219 can drive it? |
|---|---|---|
| FYQ-5641**AS**-11 | cathode | **Yes** — this is what you want |
| FYQ-5641**BS**-11 | anode | No, and no firmware setting changes that |

The MAX7219's digit drivers **sink** current to ground. A common-anode display
needs them to **source** it. That is a property of the silicon, not a
configuration.

**The good news: this is a part swap, not a board respin.** -AS and -BS share
the same 12-pin footprint and the same pin functions — only the internal
common connection differs. Drop an FYQ-5641**AS**-11 into the existing U2
footprint and the wiring in §6.2 is already correct.

**Test** (display out of circuit, multimeter in diode mode):

1. Black probe on **pin 12** (digit 1 common), red probe on **pin 11** (seg A).
2. Then reverse.

| Result | Part |
|---|---|
| Lights with **black on 12, red on 11** | **common cathode → -AS** ✅ |
| Lights with **red on 12, black on 11** | common anode → -BS ❌ |
| Neither | wrong pins or dead display |

Repeat on pin 9 (digit 2) against pin 11 to confirm. "5641" is a shape, and
mislabelled stock exists — check the part you actually received rather than
the part the BOM says.

### 9.2 Battery divider ratio

**Test** (board unpowered, battery disconnected, multimeter in resistance mode):

1. Measure **R2**: expect 47 kΩ ±1 %.
2. Measure **R3**: expect 10 kΩ ±1 %.
3. Compute `ratio = (R2 + R3) / R3`. Expect **5.700**.

Type the *measured* values into `Config.h`:

```c
#define DIVIDER_R1_OHMS   47000.0f   // your measured R2
#define DIVIDER_R2_OHMS   10000.0f   // your measured R3
```

Then with the battery connected and the firmware running, compare the reported
voltage to a multimeter across the pack **at about half charge** and set
`BATTERY_CAL_GAIN = measured / reported`.

Sanity check: at 16.8 V the A0 node should read **2.95 V** to ground.

### 9.3 MAX7219 supply and ISET

1. Confirm MAX7219 VCC measures **5 V**, not 3.3 V.
2. Measure R1 (ISET). The schematic says **27 kΩ**; you may have fitted
   **33 kΩ**. Either is fine — 27 kΩ is slightly brighter. The only value that
   matters is the **9.53 kΩ datasheet floor**: anything below it destroys the
   part. Do not power up if it reads low.
3. Confirm C1 (100 µF) and C2 (100 nF) are fitted across VCC/GND close to U3.

### 9.4 PCA9685 6 V routing — **do this before plugging a PCA9685 in**

1. With the board powered but **no PCA9685 connected**, measure the 6 V pin of
   J5 to GND. Confirm it is 6 V and not something unexpected.
2. On the **PCA9685 module itself** (disconnected), check continuity between
   its VCC pin and its V+ terminal. On most breakouts they are *separate*.
3. Work out which of the module's pins your 4-wire cable lands on.

| If 6 V goes to | Verdict | Action |
|---|---|---|
| **V+** (servo rail) | Correct | Feed VCC separately from 3.3 V or 5 V |
| **VCC** (logic) | At absolute max | Reduce to 5 V, or the chip is out of spec |

Then, with the PCA9685 connected and powered but **the Arduino unplugged**,
measure SDA and SCL to ground. If either reads above 3.6 V, the module's
pull-ups are tied to a rail that will damage the Nano ESP32's pins. Remove that
module's pull-ups before connecting the Arduino.

### 9.5 I²C pull-up total

**Do not reduce R12/R13 pre-emptively.** 4.7 kΩ is a good value. The variable
is how many *modules* bring their own.

**Test** (power off, everything connected):

Measure resistance from **SDA to the 3.3 V rail**, then **SCL to 3.3 V**.

| Measured | Verdict | Action |
|---|---|---|
| under 900 Ω | too low | remove pull-ups from modules until above 1 kΩ |
| 1.0k – 2.5k | ideal | nothing |
| 2.5k – 4.7k | fine | nothing |
| ~4.7k (board only) | fine if leads under ~1 m | nothing |
| above 4.7k | impossible | R12/R13 unfitted or open — check |

Four of five rows are "do nothing". At 100 kHz a 4.7 kΩ pull-up drives up to
251 pF, which covers six devices plus about a metre of dupont wire.

If you ever *do* need to lower it, solder a resistor in parallel across R12/R13
— no new PCB required:

| Added in parallel | Total | Sink at 3.3 V |
|---|---|---|
| 10 kΩ | 3.20 kΩ | 0.91 mA |
| 4.7 kΩ | 2.35 kΩ | 1.23 mA |
| 3.3 kΩ | 1.94 kΩ | 1.50 mA |

### 9.6 Ultrasonics on UART

No addresses to manage any more — each sonar has its own line. What to check:

1. **They are NOT on SDA/SCL.** The boot I²C audit will warn if anything
   answers at 0x11 or 0x12 while the firmware is in UART mode.
2. **Each sonar's TX goes to its own Arduino RX pin.** Front → D0 (J23),
   rear → D4 (J18). D2 is reserved for the optional e-stop. They must not share a line.
3. **Check the logic level.** With the sonar powered and the Arduino
   unplugged, measure its TX pin to ground. Above 3.6 V on a Nano ESP32 needs
   the 1 k / 2 k divider.
4. **Watch the raw frames** before trusting anything:

```bash
make monitor
```

You should see `{"us_f":1.234,"us_r":0.850}` at about 10 Hz, and the numbers
should change when you wave a hand at each sensor. `-1` means no valid frame —
check wiring, then baud (9600), then the variant.

If only one sonar reports, you have probably wired both to the same RX pin.

### 9.7 PCA9685 #2 address

If you are populating J14 as well, bridge its **A0 solder jumper** so it
answers at `0x41`. Then set in `Config.h`:

```c
#define PCA9685_II_ENABLED   1
```

Leave it at 0 if only J5 is used. The steering servo (ch0) and the HobbyWing
880 (ch1) both live on the first board at 0x40.

### 9.8 Boot I²C scan — the fastest overall check

The firmware scans the bus at boot and prints every address it finds. This is
the single best five-second test of the whole I²C side.

```bash
screen /dev/ttyARDUINO 115200     # or: picocom -b 115200 /dev/ttyARDUINO
```

Expected, with everything fitted:

```
0x0C  AK8963 magnetometer
0x11  DYP-A21 front
0x12  DYP-A21 rear
0x40  PCA9685 #1
0x41  PCA9685 #2   (only if J14 populated and A0 bridged)
0x68  MPU9250
0x76  BMP280
```

| Symptom | Likely cause |
|---|---|
| Nothing at all | SDA/SCL swapped, or no common ground |
| Works with 2 devices, fails with the third | pull-ups too low in parallel — test 9.5 |
| `0x0C` missing but `0x68` present | pass-through not enabled; usually harmless to retry |
| `0x68` missing | GY-91 unpowered, or AD0 pulled high (try `0x69`) |
| Only one of `0x11`/`0x12` | second sonar not reprogrammed — test 9.6 |
| Intermittent, addresses come and go | bus too fast for the wiring — confirm `I2C_CLOCK_HZ` is 100000 |

---

## 9.9 Lidar: RPLidar A2M12 baud

Not a multimeter test, but it belongs in the same list because the symptom is
identical to a dead device.

The **A2M12 runs at 256000 baud.** The older A2M8 runs at 115200, and that
number is everywhere online. At the wrong baud the driver opens the port,
reports no error, and never publishes a single scan.

```bash
ros2 launch a2r_bot_follow follow.launch.py     # rplidar is now the default
ros2 topic hz /scan                             # expect ~10 Hz
ros2 topic hz /scan/fused                       # same rate, after sonar fusion
```

If `/scan` is silent: check `serial_baudrate` is 256000, check `/dev/ttyLIDAR`
resolves (§ udev rule), and confirm the lidar's motor is actually spinning —
the A2M12 needs its motor power, not just the data line.

---

## 10. Bring-up order

1. Flash the firmware with the **propshaft disconnected or the car on blocks**.
2. Open the serial monitor at 115200. You should see the banner and the I²C
   scan (test 9.8).
3. Confirm every expected address appears. **Fix the bus before going
   further** — every later problem looks different and is caused by this.
4. Watch the display run its self-test and settle on a percentage. Compare to a
   multimeter; set `BATTERY_CAL_GAIN` (test 9.2).
5. Send `{"cmd":"PING"}` — you should get a status line back.
6. Send `{"cmd":"CAL_IMU"}` with the car level and still.
7. Send `{"cmd":"DRIVE","v":0,"s":0.3}` and watch the servo. Then `-0.3`.
   **Confirm positive `s` steers left** (REP-103). If it steers right, flip
   `SERVO_REVERSED` in `Config.h` rather than negating it somewhere in the ROS
   stack — the sign convention should be true at the lowest level, or every
   layer above inherits the confusion.
8. Only now, still on blocks, try `{"cmd":"DRIVE","v":0.15,"s":0}`.
9. Put it on the ground.

Step 1 is not optional. An ESC that arms into a throttle command will launch
the car off the bench.
