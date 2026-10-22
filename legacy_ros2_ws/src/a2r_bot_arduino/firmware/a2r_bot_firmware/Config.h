/* =============================================================================
 *  Config.h — A2R_bot firmware, single place for every pin and constant
 * =============================================================================
 *  Board:  Arduino Nano ESP32 (ESP32-S3)   — primary target
 *          Arduino Nano / Nano Every       — also supported, see notes below
 *
 *  ROLE OF THIS MCU
 *  ----------------
 *  The Jetson decides WHAT to do, the Pi decides HOW to get there, and this
 *  board is the only thing that touches a motor. It has exactly four jobs:
 *
 *      1. Parse DRIVE frames from the Pi over USB serial and convert them
 *         into an ESC pulse (PCA9685 ch1) and a servo pulse (PCA9685 ch0).
 *      2. Read the GY-91 IMU and the two DYP-A21 ultrasonics and stream them
 *         back to the Pi as telemetry.
 *      3. Measure pack voltage on A0 and show the battery percentage on the
 *         MAX7219 / FYQ-5641 7-segment display.
 *      4. Cut the throttle if the Pi goes quiet. This is the ONLY safety
 *         layer that survives a Pi crash, so it is not optional.
 *
 *  BUS MAP
 *  -------
 *      I2C  (A4 = SDA, A5 = SCL)   PCA9685 0x40
 *                                  GY-91: MPU-9250 0x68, AK8963 0x0C, BMP280 0x76
 *                                  DYP-A21 front 0x11, rear 0x12   (see note!)
 *      SPI-ish (3 GPIO, bit-banged) MAX7219
 *      ADC   A0                    battery divider
 *      USB serial                  link to the Raspberry Pi
 *
 *  >>> IMPORTANT — TWO HARDWARE FACTS WORTH CHECKING BEFORE YOU WIRE <<<
 *
 *  (a) THE MAX7219 IS NOT AN I2C DEVICE. It is a 3-wire SPI-style shift
 *      register (DIN / CS / CLK). It cannot share A4/A5 with the PCA9685 and
 *      the IMU no matter how it is wired — there is no I2C address decoder in
 *      the chip at all. It gets its own three GPIO pins below. Everything
 *      else on your list genuinely does share the I2C bus.
 *
 *  (b) DYP-A21 SHIPS IN SEVERAL OUTPUT VARIANTS: I2C, UART (9600 8N1),
 *      analogue voltage, and switched output. Only the I2C variant can sit on
 *      A4/A5. If your modules have a 4-wire pigtail and output 0xFF-headed
 *      frames, they are the UART variant — set ULTRASONIC_MODE to
 *      US_MODE_UART below and wire them to the pins listed there. The rest of
 *      the firmware, and every ROS topic, is unchanged either way.
 * ============================================================================= */

#ifndef A2R_CONFIG_H
#define A2R_CONFIG_H

#include <Arduino.h>

/* ---------------------------------------------------------------------------
 *  0. BOARD DETECTION
 *     ADC width and reference differ between the ESP32-S3 Nano (12-bit, 3.3 V)
 *     and a classic AVR Nano (10-bit, 5 V). Getting this wrong makes the
 *     battery percentage silently useless, so it is derived, never typed.
 * ------------------------------------------------------------------------- */
#if defined(ARDUINO_ARCH_ESP32)
  #define BOARD_IS_ESP32      1
  #define ADC_MAX_COUNTS      4095.0f
  #define ADC_VREF            3.30f
  #define BOARD_NAME          "nano_esp32"
#else
  #define BOARD_IS_ESP32      0
  #define ADC_MAX_COUNTS      1023.0f
  #define ADC_VREF            5.00f
  #define BOARD_NAME          "nano_avr"
#endif

/* ---------------------------------------------------------------------------
 *  1. SERIAL LINK TO THE RASPBERRY PI
 * ------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------
 *  SERIAL LINK SPEED
 *
 *  Raised from 115200 to 921600, and the reason is measured, not stylistic.
 *
 *  Telemetry out of this board is roughly:
 *      IMU     78 B @ 50 Hz = 3900 B/s
 *      MAG     38 B @ 25 Hz =  950 B/s
 *      RANGES  24 B @ 10 Hz =  240 B/s
 *      BARO/BATTERY/STATUS          ~200 B/s
 *  and DRIVE frames inbound are 38 B @ 50 Hz = 1900 B/s.
 *
 *  Total ~7200 B/s. At 115200 baud (11520 B/s) that is 62 % utilisation, and
 *  utilisation is not the real problem - HEAD-OF-LINE BLOCKING is. A 78-byte
 *  IMU line already clocking out of the UART delays the next DRIVE frame by
 *  6.8 ms. The drive loop runs at 100 Hz, so that is most of a control period
 *  of pure latency, arriving at random, on a robot whose whole job is to
 *  react quickly to a person moving.
 *
 *  At 921600 the same line costs 0.85 ms and utilisation drops to 8 %.
 *
 *  On the Arduino Nano ESP32 this number is nominal anyway - the port is
 *  native USB-CDC and the real throughput is USB-limited and far higher - but
 *  it must still MATCH the `baud` parameter in a2r_bot_arduino/config/
 *  arduino.yaml, and it matters for real if you ever drive the UART pins on
 *  J22/J23 or move to a classic Nano.
 * ------------------------------------------------------------------------- */
#define SERIAL_BAUD             921600UL
/* Link transport (fw 3.5.0): Serial0 = hardware UART on D0/D1 (PCB J22 TX / J23 RX)
 * wired to the Jetson. Native USB-CDC (Serial) can stall forever on the ESP32
 * core - see SafeSerial.h - which is what forced the USB-wedge reboots. */
#ifndef LINK_PORT
  #if BOARD_IS_ESP32
    #define LINK_PORT Serial0
  #else
    #define LINK_PORT Serial
  #endif
#endif
#define RX_BUFFER_LEN           192
#define SERIAL_PRINT_PRECISION  4

/* ---------------------------------------------------------------------------
 *  2. PCA9685 — 16-channel PWM driver
 *     ch0 = steering servo,  ch1 = Hobbywing 880 ESC.
 *     Both are ordinary RC pulses; the PCA9685 is just a very stable,
 *     interrupt-free way to generate them.
 * ------------------------------------------------------------------------- */
/* The carrier board breaks out TWO PCA9685 headers, J5 (PCA9685_I) and J14
 * (PCA9685_II), wired to the same SDA/SCL. Two boards on one bus need two
 * addresses: the second one must have its A0 solder jumper bridged, making it
 * 0x41. The firmware only drives the first - steering on channel 0 and the
 * HobbyWing 880 on channel 1 both live there. Set PCA9685_II_ENABLED to 1 only
 * once you actually have something on the second board. */
/* ---------------------------------------------------------------------------
 *  I2C BUS SPEED
 *
 *  Dropped from 400 kHz to 100 kHz, deliberately.
 *
 *  Six devices hang off SDA/SCL here - two PCA9685s, the GY-91, two DYP-A21s
 *  and whatever else - mostly on flying leads. Every centimetre of lead adds
 *  capacitance, and realistically that bus sits around 400 pF.
 *
 *  I2C is open drain: nothing drives the line high, a pull-up resistor does.
 *  That gives two limits pulling in OPPOSITE directions:
 *
 *    floor   - a device must sink 3 mA and still hold below 0.4 V, so
 *              R > (3.3 - 0.4) / 3 mA = 967 ohm. Go lower and a device
 *              physically cannot pull the line down far enough to register.
 *
 *    ceiling - rise time t_r = 0.8473 * R * C must beat the spec limit:
 *              300 ns at 400 kHz, 1000 ns at 100 kHz.
 *
 *  At 400 kHz with 400 pF the ceiling is 885 ohm, which is BELOW the 967 ohm
 *  floor. The window is empty: no pull-up value works. That is not a tuning
 *  problem, it is arithmetic.
 *
 *  At 100 kHz the window opens to 967..2951 ohm and everything is fine.
 *
 *  And 100 kHz costs nothing here. Measured against the actual traffic -
 *  MPU9250 and AK8963 at 50 Hz, BMP280 at 2 Hz, both sonars at 10 Hz and the
 *  PCA9685 at 20 Hz - the bus runs at about 15 % utilisation. The IMU has all
 *  the headroom it needs.
 *
 *  Only raise this to 400000 if the sensors are on SHORT leads (under ~250 pF)
 *  and you have measured the total pull-up resistance into the 1.0k..1.4k
 *  window. Otherwise you get intermittent NACKs that look like flaky wiring.
 * ------------------------------------------------------------------------- */
#define I2C_CLOCK_HZ            100000UL

#define PCA9685_ADDR            0x40
#define PCA9685_II_ADDR         0x41       /* J14, A0 jumper bridged           */
#define PCA9685_II_ENABLED      0
#define PCA9685_PWM_FREQ_HZ     50.0f      /* 20 ms frame — standard RC        */
#define PCA9685_OSC_HZ          25000000.0f
#define CH_SERVO                0          /* <-- steering servo               */
#define CH_ESC                  1          /* <-- Hobbywing 880 ESC            */

/* PCA9685 registers */
#define PCA_MODE1               0x00
#define PCA_MODE2               0x01
#define PCA_LED0_ON_L           0x06
#define PCA_ALL_LED_ON_L        0xFA
#define PCA_PRESCALE            0xFE
#define PCA_MODE1_SLEEP         0x10
#define PCA_MODE1_AI            0x20       /* auto-increment                   */
#define PCA_MODE1_RESTART       0x80
#define PCA_MODE2_OUTDRV        0x04       /* totem pole                       */

/* ---------------------------------------------------------------------------
 *  3. STEERING SERVO  (PCA9685 channel 0)
 *     Trim these three numbers on the bench with the wheels off the ground.
 *     SERVO_CENTER_US is the single most useful calibration on the car: if
 *     the robot pulls to one side when the planner commands straight, it is
 *     almost always this, not the planner.
 * ------------------------------------------------------------------------- */
#define SERVO_MIN_US            1285       /* full left  (s = -1.0)            */
#define SERVO_CENTER_US         1620       /* straight   (s =  0.0)            */
#define SERVO_MAX_US            2005       /* full right (s = +1.0)            */
#define SERVO_INVERT            0          /* 1 if the servo is mounted mirrored */
/* Mechanical travel limit as a fraction of the 1000..2000 range. The WR-01
 * steering rack binds before the servo does; 0.85 keeps it off the stops so
 * the servo does not buzz and cook itself at full lock. */
#define SERVO_TRAVEL_LIMIT      1.0f
/* Slew limit, µs per second. The planner already has its own steer_slew_dps,
 * but the firmware enforces its own ceiling so a corrupted frame can never
 * slam the rack from lock to lock in one step. */
#define SERVO_MAX_SLEW_US_PER_S 1000.0f

/* ---------------------------------------------------------------------------
 *  4. HOBBYWING 880 ESC  (PCA9685 channel 1)
 *     Bidirectional brushed ESC taking a standard RC pulse.
 *
 *     ARMING: it will ignore everything until it has seen a steady neutral
 *     pulse for a couple of seconds after power-up. Hence ESC_ARM_MS.
 *
 *     DEADBAND: the ESC has its own neutral deadband. Commanding inside it
 *     produces no motion but DOES produce the characteristic whine. We skip
 *     straight over it: any non-zero throttle request is mapped to start at
 *     ESC_DEADBAND_US past neutral. This is why the car moves cleanly from a
 *     standstill instead of buzzing.
 *
 *     REVERSE: most brushed RC ESCs need neutral → brake → neutral → reverse.
 *     ESC_REVERSE_HANDSHAKE implements that; set to 0 if yours is configured
 *     in "crawler / instant reverse" mode.
 * ------------------------------------------------------------------------- */
#define ESC_MIN_US              1000       /* full reverse                     */
#define ESC_NEUTRAL_US          1500
#define ESC_MAX_US              2000       /* full forward                     */
#define ESC_ARM_MS              3000       /* hold neutral this long at boot   */
#define ESC_DEADBAND_US         60         /* skip the ESC's own dead zone     */
#define ESC_MAX_FWD_FRAC        0.70f      /* throttle cap — matches params.yaml
                                              max_speed_cmd. Keep them equal.  */
#define ESC_MAX_REV_FRAC        0.50f      /* reverse is for escapes, not speed */
#define ESC_REVERSE_HANDSHAKE   1
/* Brake pulse AND the neutral gap that follows it. 60 ms was short enough
 * that the 880 often read the sequence as a single brake and never entered
 * reverse - the escape manoeuvre then just stopped the car dead, which is
 * exactly the reported "it does not reverse" symptom. 150 ms is comfortably
 * inside every published figure for this class of ESC. */
#define ESC_BRAKE_PULSE_MS      150
#define ESC_MAX_SLEW_US_PER_S   1800.0f    /* protects the driveline + the pack */

/* ---------------------------------------------------------------------------
 *  5. FAILSAFE
 *     If the Pi stops sending DRIVE frames the car coasts to neutral and
 *     centres the steering. Nothing upstream can disable this.
 * ------------------------------------------------------------------------- */
#define CMD_TIMEOUT_MS          500        /* no DRIVE for this long → neutral */

/* ---------------------------------------------------------------------------
 *  5.1 PCA9685 HARDWARE WATCHDOG
 *
 *  THE FAILURE THIS EXISTS FOR. The failsafe above is a software one: the
 *  drive task notices that no DRIVE frame arrived and writes a neutral pulse.
 *  That only works while the drive task still runs. If this board wedges or
 *  browns out - the freeze described in TROUBLESHOOTING.md, "serial stops,
 *  only a USB replug brings it back" - the PCA9685 is an independent chip
 *  with its own oscillator. It keeps emitting whatever pulse width was last
 *  loaded into its register, forever. If that was 60 % throttle, the car
 *  drives into the wall at 60 % throttle with nothing left to stop it.
 *
 *  So while the car is moving, the ESC channel is programmed as a train of
 *  pulses that EXPIRES: the drive task re-arms it every cycle. Stop re-arming
 *  - because the board hung - and the output decays to neutral on its own.
 *  Implemented in Drive.cpp with the PCA9685's ALL_LED_OFF latch.
 * ------------------------------------------------------------------------- */
#define ENABLE_PCA_WATCHDOG     1
#define PCA_WATCHDOG_MS         250        /* no re-arm for this long → outputs off */
#define ESTOP_ACTIVE_LOW        1
#define PIN_ESTOP               2          /* optional hardware e-stop button  */
#define ENABLE_ESTOP_PIN        0          /* 1 to enable the pin above        */

/* ---------------------------------------------------------------------------
 *  6. BATTERY — Tattu TA-AA-6750-4S1P  (TAA67504S25X9)
 *     14.8 V nominal, 4 cells in series, 6750 mAh, 25C.
 *
 *     DIVIDER: A0 sees  V_pack * R2 / (R1 + R2).
 *
 *     VALUES TAKEN FROM THE CARRIER-BOARD SCHEMATIC (simplied_circuit_design):
 *
 *         BAT_SENSE --[ R2 47k ]--+--[ R3 10k ]-- GND
 *                                 |
 *                                 +--> A0
 *
 *     so R1_OHMS = 47k (the top/series leg) and R2_OHMS = 10k (the bottom leg).
 *     Ratio 5.700; a full 16.8 V pack presents 2.947 V at A0.
 *
 *     NOTE ON THE NAMING: the schematic calls these R2 and R3. The names here
 *     are R1 = upper leg, R2 = lower leg, which is the convention the formula
 *     above uses. They do not line up, and that is worth knowing before you
 *     'correct' one to match the other.
 *
 *     Earlier revisions of this file assumed 100k/22k (ratio 5.545). Against
 *     the real 47k/10k board that under-reports by 2.7 %, which is 0.46 V at
 *     the top of the pack - enough to read 4.09 V/cell on a fully charged
 *     battery (about 85 % on the discharge LUT) and to trip the CELL_CRITICAL
 *     warning roughly 0.4 V early.
 *
 *     ADC HEADROOM: 2.947 V is 89 % of a 3.3 V full scale. That is fine for a
 *     classic AVR, but the ESP32-S3's ADC is noticeably non-linear in its top
 *     ~10 %, and BATTERY_CAL_GAIN is a single multiplier that cannot correct a
 *     curve. If the reading tracks well at half charge but drifts near full,
 *     that is this effect - lower the divider (e.g. 47k/8.2k, giving 2.53 V at
 *     16.8 V) rather than fighting it in software.
 *
 *     Put a 100 nF cap from A0 to GND. Measure both resistors with a meter and
 *     type the REAL values below - 1 % parts still land 2 % apart, which is
 *     0.3 V at the top of the pack, which is 8 % of displayed charge.
 *
 *     Then check with a multimeter and nudge BATTERY_CAL_GAIN until the
 *     reported voltage matches, AT ABOUT HALF CHARGE. That single number
 *     absorbs resistor tolerance, ADC gain error and wiring drop at once.
 * ------------------------------------------------------------------------- */
#define PIN_BATTERY_SENSE       A0

/* Carrier board fits the same divider regardless of which Nano is socketed. */
#define DIVIDER_R1_OHMS         47000.0f   /* schematic R2, upper leg          */
#define DIVIDER_R2_OHMS         10000.0f   /* schematic R3, lower leg          */

/* If you are NOT using the carrier board, the old standalone values were:
 *   ESP32-S3 Nano (3.3 V ADC):  R1 = 100k, R2 = 22k   -> 16.8 V reads 3.03 V
 *   Classic AVR   (5.0 V ADC):  R1 = 100k, R2 = 33k   -> 16.8 V reads 4.17 V
 */

#define BATTERY_CAL_GAIN        1.000f     /* multiply measured V by this      */
#define BATTERY_CAL_OFFSET      0.000f     /* then add this (volts)            */

#define BATTERY_CELLS           4
#define BATTERY_CAPACITY_MAH    6750.0f
#define CELL_FULL_V             4.20f
#define CELL_EMPTY_V            3.50f      /* LVC: land it, do not fly to 3.0  */
#define CELL_CRITICAL_V         3.60f      /* start warning here               */

/* Sampling and filtering. Raw ADC on a car with a motor on the same pack is
 * a noisy mess; the chain is median-of-N → slow EMA → sag compensation. */
#define BATTERY_SAMPLES         15         /* odd number, for the median       */
#define BATTERY_EMA_ALPHA       0.05f      /* ~4 s settling at 5 Hz            */
#define BATTERY_READ_PERIOD_MS  200

/* SAG COMPENSATION. Under throttle the pack voltage drops by I·R_internal,
 * so a naive gauge reads "40 %" every time you accelerate and "70 %" again
 * when you coast. We have no current sensor, but we know exactly how much
 * throttle we just asked for, which is a good enough proxy:
 *
 *     V_rest ≈ V_measured + BATTERY_SAG_V_AT_FULL * |throttle_fraction|
 *
 * 6750 mAh 4S packs sit near 4 mΩ/cell; at ~20 A that is about 0.32 V.
 * Verify yours: note the voltage at rest, then hold full throttle with the
 * wheels off the ground and note it again. The difference goes here. */
#define BATTERY_SAG_V_AT_FULL   0.32f
/* The gauge is not allowed to climb faster than this (percent per second).
 * A battery does recover when you let off, but not instantly, and a bouncing
 * percentage reads as a broken gauge. Falling is unrestricted. */
#define BATTERY_RISE_LIMIT_PPS  0.4f

/* ---------------------------------------------------------------------------
 *  7. MAX7219 + FYQ-5641AS-11 4-digit 7-segment display
 *     >>> SPI-STYLE, NOT I2C — see the note at the top of this file. <<<
 *
 *     >>> FIT THE -AS PART. THE SCHEMATIC SAYS -BS. <<<
 *
 *     The carrier-board schematic labels U2 as FYQ-5641BS-11. In this family
 *     the suffix IS the polarity:
 *
 *         5641AS  = COMMON CATHODE   <-- what this firmware drives
 *         5641BS  = COMMON ANODE     <-- the MAX7219 physically cannot drive
 *
 *     The MAX7219's digit drivers sink current to ground; a common-anode part
 *     needs them to source it. No firmware setting changes that, and neither
 *     does RSET or the intensity register. The part must be the AS.
 *
 *     THE GOOD NEWS: -AS and -BS share the SAME 12-pin footprint and the same
 *     pin functions. Only the internal common connection differs. So this is
 *     a part substitution, NOT a board respin - drop an FYQ-5641AS-11 into the
 *     existing U2 footprint and everything below is already correct.
 *
 *     Verify before soldering with a coin cell and a 1 kΩ resistor, or a
 *     multimeter in diode mode: see docs/WIRING.md §9.1. "5641" is a shape,
 *     and mislabelled stock exists.
 *
 *     MAX7219 wiring to the MCU (any three free digital pins):
 *         DIN  → PIN_MAX7219_DIN
 *         CS   → PIN_MAX7219_CS     (a.k.a. LOAD)
 *         CLK  → PIN_MAX7219_CLK
 *         VCC  → 5 V  (confirmed on the built board)
 *         GND  → GND
 *     Put 100 µF + 100 nF across the MAX7219's supply, right at the chip.
 *     The board fits C1/C2 for exactly this.
 *
 *     MAX7219 to FYQ-5641AS-11 (COMMON CATHODE, 12 pins):
 *         SEG A→pin11  B→pin7   C→pin4   D→pin2
 *         SEG E→pin1   F→pin10  G→pin5   DP→pin3
 *         DIG0→pin12 (leftmost)  DIG1→pin9  DIG2→pin8  DIG3→pin6
 *
 *     The digit order means the display reads left-to-right as DIG0..DIG3, so
 *     what showText() is handed is what appears. If it comes out mirrored, set
 *     DISPLAY_DIGITS_REVERSED to 1 rather than reworking the wiring.
 * ------------------------------------------------------------------------- */
#define ENABLE_DISPLAY          1
#define PIN_MAX7219_DIN         11
#define PIN_MAX7219_CS          10
#define PIN_MAX7219_CLK         13
#define MAX7219_NUM_DIGITS      4

/*  RSET = the ISET resistor, schematic R1.
 *
 *  NOTE A DISCREPANCY: the schematic labels R1 as 27k; you reported fitting
 *  33k. Both are fine - the only thing that matters is staying ABOVE the
 *  datasheet's 9.53k absolute minimum, and both are. 27k is slightly brighter
 *  than 33k. INTENSITY below is set for the dimmer of the two, so if you
 *  actually have 27k fitted the display will simply be a little brighter than
 *  planned, which is not a problem. Check which is on the board and stop
 *  worrying about it either way.
 *
 *  Going BELOW 9.53k is what destroys the part, and neither value does.
 *
 *  It is also the right direction for this design. SCANLIMIT is set to 3,
 *  i.e. four digits, so each digit driver is on 1/4 of the time instead of
 *  1/8. The datasheet warns that scanning few digits raises per-driver
 *  dissipation and that RSET should be increased to compensate. A 10k RSET
 *  driving only four digits is the case that runs hot; 33k is not.
 *
 *  The cost is a dimmer display, so INTENSITY starts high to claw some back.
 *  Raise to 15 if it is still hard to read in daylight -- the duty-cycle
 *  control costs nothing and cannot exceed the peak current RSET already set.
 *
 *  If the MAX7219 is running at 3.3V rather than 5V, 33k will probably be too
 *  dim to read at all: the drive headroom is only VDD - Vf, so dropping VDD
 *  from 5V to 3.3V roughly halves it. Either run the chip at 5V behind a
 *  level shifter (see docs/WIRING.md) or fit a smaller RSET -- but not below
 *  9.53k.
 */
#define MAX7219_INTENSITY       12         /* 0..15; was 6, raised for RSET=33k */
#define DISPLAY_DIGITS_REVERSED 0          /* 1 if the digits come out mirrored */
#define DISPLAY_UPDATE_MS       250
/* Alternate between "  87" (percent) and "16.8" (volts) so you get both
 * without a second display. Set to 0 to show percentage only. */
#define DISPLAY_SHOW_VOLTS      1
#define DISPLAY_PERCENT_MS      4000
#define DISPLAY_VOLTS_MS        2000
#define DISPLAY_LOW_BLINK_MS    500        /* flash "LO" below CELL_CRITICAL_V  */

/* MAX7219 registers */
#define MAX7219_REG_NOOP        0x00
#define MAX7219_REG_DIGIT0      0x01
#define MAX7219_REG_DECODEMODE  0x09
#define MAX7219_REG_INTENSITY   0x0A
#define MAX7219_REG_SCANLIMIT   0x0B
#define MAX7219_REG_SHUTDOWN    0x0C
#define MAX7219_REG_DISPLAYTEST 0x0F

/* ---------------------------------------------------------------------------
 *  8. GY-91  (MPU-9250 accel+gyro+mag, BMP280 baro)  — I2C on A4/A5
 * ------------------------------------------------------------------------- */
#define ENABLE_IMU              1
#define ENABLE_MAG              1
#define ENABLE_BARO             0

#define MPU9250_ADDR            0x68       /* 0x69 if AD0 is pulled high       */
#define AK8963_ADDR             0x0C
#define BMP280_ADDR             0x76       /* 0x77 on some GY-91 clones        */

#define MPU_WHO_AM_I            0x75
#define MPU_PWR_MGMT_1          0x6B
#define MPU_PWR_MGMT_2          0x6C
#define MPU_CONFIG              0x1A
#define MPU_GYRO_CONFIG         0x1B
#define MPU_ACCEL_CONFIG        0x1C
#define MPU_ACCEL_CONFIG2       0x1D
#define MPU_SMPLRT_DIV          0x19
#define MPU_INT_PIN_CFG         0x37
#define MPU_ACCEL_XOUT_H        0x3B

#define AK_WHO_AM_I             0x00
#define AK_ST1                  0x02
#define AK_HXL                  0x03
#define AK_CNTL1                0x0A
#define AK_ASAX                 0x10

/* Ranges. ±4 g and ±500 °/s comfortably cover an RC car that lives at
 * 1.2 m/s while keeping a good chunk of the 16-bit resolution. */
#define ACCEL_FS_SEL            0x08       /* ±4 g   → 8192 LSB/g              */
#define GYRO_FS_SEL             0x08       /* ±500 dps → 65.5 LSB/(°/s)        */
#define ACCEL_SCALE             (1.0f / 8192.0f * 9.80665f)   /* → m/s²        */
#define GYRO_SCALE              (1.0f / 65.5f * 0.0174532925f)/* → rad/s       */
#define MAG_SCALE               0.1499f    /* µT per LSB (16-bit, 4912 µT FS)  */

#define IMU_PUBLISH_HZ          50
#define BARO_PUBLISH_HZ         5
/* Gyro bias is re-estimated whenever the car has been commanded to neutral
 * and the accelerometer says it is not moving. The Pi does this too, but
 * doing it here means the raw stream is already close to honest. */
#define GYRO_BIAS_SAMPLES       256
#define GYRO_STILL_ACCEL_TOL    0.35f      /* m/s² deviation from 1 g          */

/* ---------------------------------------------------------------------------
 *  9. DYP-A21 ULTRASONICS  ×2  (IP67, 5 m)
 *     Read the variant note at the top of this file before wiring.
 * ------------------------------------------------------------------------- */
#define US_MODE_I2C             0
#define US_MODE_UART            1
#define US_MODE_TRIG_ECHO       2

#define ULTRASONIC_MODE         US_MODE_UART

#define ENABLE_ULTRASONIC       0   /* sonars omitted: no init, no polling, no telemetry */
#define US_PUBLISH_HZ           10
#define US_MIN_RANGE_M          0.03f
#define US_MAX_RANGE_M          5.00f
#define US_FOV_RAD              0.52f      /* ~30° cone, per the datasheet     */
/* Reject a reading that jumps more than this between consecutive samples
 * unless it repeats. Ultrasonics pick up spurious echoes off the floor and
 * off each other; one bad 0.1 m reading would make the planner slam on the
 * brakes. Two in a row is believed. */
#define US_MAX_JUMP_M           1.20f
/* Front and rear fire alternately, never together, so the front transducer
 * cannot hear the rear one's ping. This is why the effective per-sensor rate
 * is US_PUBLISH_HZ and not double it. */
#define US_PING_GAP_MS          60   /* alternating triggers, about 8 Hz per sonar */

#if ULTRASONIC_MODE == US_MODE_I2C
  #define US_FRONT_I2C_ADDR     0x11
  #define US_REAR_I2C_ADDR      0x12
  #define US_I2C_TRIGGER_CMD    0x01
  #define US_I2C_READ_DELAY_MS  70         /* time-of-flight for 5 m + margin  */
#elif ULTRASONIC_MODE == US_MODE_UART
  /* -- UART ultrasonics: the configuration this robot uses -----------------
   *
   *  The sonars are OFF the I2C bus and on their own serial lines. That is
   *  the better arrangement anyway, for three reasons:
   *
   *    - The I2C bus no longer carries two devices that each need ~70 ms of
   *      time-of-flight to answer. The PCA9685 and the IMU get the bus to
   *      themselves, and the IMU is the one device here with a real deadline.
   *    - A sonar that wedges its line can no longer take down the bus the
   *      servo and the ESC depend on. Steering keeps working.
   *    - Each sonar free-runs, so there is no trigger-then-wait round trip:
   *      readings simply arrive.
   *
   *  FRAME FORMAT
   *  DYP-A21 UART auto-output, 9600 8N1, four bytes about every 100 ms:
   *
   *      0xFF, dist_HI, dist_LO, checksum     checksum = (0xFF+HI+LO) & 0xFF
   *      distance in MILLIMETRES
   *
   *  WIRING
   *  In auto-report mode the sensor talks and never listens, so each sonar
   *  needs exactly ONE Arduino pin - its TX into our RX. The Arduino TX pins
   *  below are used only when US_UART_AUTO_REPORT is 0.
   *
   *      sonar            wire               Arduino pin
   *      ------------------------------------------------------------
   *      FRONT  TX   ->   PIN_US_FRONT_RX    D0   (board header J23/RX)
   *      FRONT  VCC  ->   5 V                     (see logic levels below)
   *      FRONT  GND  ->   GND
   *      REAR   TX   ->   PIN_US_REAR_RX     D2
   *      REAR   VCC  ->   5 V
   *      REAR   GND  ->   GND
   *
   *  >>> THE BOARD BREAKS OUT ONLY ONE TX/RX PAIR (J22/J23). <<<
   *  Two transmitters cannot share one RX line. They would collide, and
   *  because the checksum is only 8 bits you would get a plausible-looking
   *  but wrong distance a good fraction of the time - worse than no sensor.
   *  So the front sonar can use J23, and the SECOND needs a spare digital pin
   *  from J18 or J26. D2 is free and is the default here.
   *
   *  On the ESP32 any GPIO can be a UART pin, so D2 is not special - change
   *  it to whatever suits your loom.
   *
   *  LOGIC LEVELS
   *  The DYP-A21's TX idles at its own VCC. Powered from 5 V it presents 5 V
   *  to a 3.3 V ESP32 input, which is out of spec. Either power the sonar
   *  from 3.3 V if your variant allows it (many do - check the label), or put
   *  a divider on its TX line: 1k in series, 2k to ground. A sonar is far too
   *  slow for the divider to cost anything.
   * ---------------------------------------------------------------------- */
  #define US_UART_BAUD          9600

  /* 1 = sensor free-runs, we only listen (one wire per sonar, TX unused).
   * 0 = we send a 0x55 query and wait for the reply (needs the TX pins).
   * Auto is the default: one wire fewer and no round trip. */
  #define US_UART_AUTO_REPORT   0   /* 0 = query mode: trigger each sonar. Works for UW AND TW types */

  #define PIN_US_FRONT_RX       0   /* D0 <- front sonar TX */
  #define PIN_US_FRONT_TX       1   /* D1 -> front sonar RX */
  #define PIN_US_REAR_RX        5   /* D5 <- rear sonar TX */
  #define PIN_US_REAR_TX        4   /* D4 -> rear sonar RX */

  /* If no valid frame arrives inside this window the sonar is reported as
   * +inf (nothing in range) rather than holding its last value. A stale range
   * the planner still believes is the dangerous failure here: it would drive
   * at a wall the sensor stopped reporting ten seconds ago. */
  #define US_UART_TIMEOUT_MS    600

  /* Which hardware UART each sonar hangs off.
   *
   * The ESP32-S3 has three UARTs and none of them is spent on the Pi link -
   * that is native USB-CDC, a separate peripheral - so Serial1 and Serial2
   * are both genuinely free, and on this chip their pins can be remapped to
   * any GPIO. That is why the pin numbers above are a free choice rather than
   * fixed by the package.
   *
   * On a classic AVR these become SoftwareSerial instances instead, declared
   * in Sensors.cpp. */
  #define US_PORT_FRONT         Serial0   /* UART0 - its native pins are D0/D1 */
  #define US_PORT_REAR          Serial1   /* UART1 */

  /* Set to 0 if only ONE sonar is fitted. The other reports "no reading"
   * honestly rather than echoing whatever the fitted one said.
   *
   * ── CURRENT BUILD: FRONT SONAR ONLY ──────────────────────────────────
   *
   * The rear DYP-A21 is physically removed - VCC, GND and its signal wire
   * all disconnected. All three must come off, not just power: an unpowered
   * sensor still has ESD protection diodes from its TX pin to its own VDD
   * rail, so leaving the signal wire attached lets the FRONT sensor's drive
   * leak through them and partially power the dead one. A half-powered
   * microcontroller sometimes transmits, and you would be back to colliding
   * frames - but only intermittently, and only once the parasitic rail
   * creeps past the brown-out threshold. That is a horrible fault to chase.
   *
   * With US_REAR_ENABLED 0 the firmware never opens the rear UART, never
   * waits for frames that are not coming, and marks the rear range invalid
   * so the rest of the stack reports "no reading" instead of guessing.
   *
   * It also disables the shared-pin guard below, which is why PIN_US_REAR_RX
   * may stay equal to PIN_US_FRONT_RX here. Nothing is listening on it.
   *
   * WHAT YOU GIVE UP: the reverse-escape manoeuvre backs out along the
   * least-cost reverse arc, and the lidar is partly occluded to the rear by
   * the battery and the Jetson. Reversing therefore runs on degraded
   * information. Keep space behind the car, or lower reverse_speed. Forward
   * coverage - glass, gloss, low obstacles - is completely unaffected.
   *
   * To restore the rear sonar: reconnect all three wires to a free pin
   * (D2-D9 and D12 are on header J18), set PIN_US_REAR_RX to match, and set
   * US_REAR_ENABLED back to 1.
   * --------------------------------------------------------------------- */
  #define US_FRONT_ENABLED      1
  #define US_REAR_ENABLED       1   /* both sonars fitted */

/* ---------------------------------------------------------------------------
 *  >>> THE TWO SONARS CANNOT SHARE ONE RX PIN. <<<
 *
 *  This is worth stating loudly because the wiring looks reasonable and the
 *  failure is silent.
 *
 *  The DYP-A21 auto-report frame is:
 *
 *      0xFF, dist_HI, dist_LO, checksum
 *
 *  Note what is NOT in it: any kind of sensor identity. Every frame from
 *  every sonar looks exactly alike. So if both transmitters are wired to one
 *  RX pin, you receive a stream of perfectly valid, correctly-checksummed
 *  distances with NO WAY WHATSOEVER to know which sensor each came from.
 *  There is no clever parsing that recovers it, because the information was
 *  never transmitted.
 *
 *  Assigning them alternately - front, rear, front, rear - is right only
 *  while the two free-running clocks stay in step, which they do not. The
 *  moment one sensor drops a frame or drifts, front and rear swap and stay
 *  swapped. The robot then brakes for things behind it and drives into things
 *  in front of it.
 *
 *  On top of that they physically collide. Two sensors reporting at ~100 ms
 *  on independent crystals drift through each other every few seconds; during
 *  a collision both drive the line at once, corrupting bytes - and some
 *  corrupted frames still pass an 8-bit checksum.
 *
 *  THE FIX IS ONE JUMPER WIRE. Auto-report needs only ONE pin per sonar (its
 *  TX into our RX), so two sonars need two pins, not four. D2 through D9 and
 *  D12 are all free and all broken out on header J18. Move the second sonar's
 *  TX to any of them and set PIN_US_REAR_RX to match.
 *
 *  If you genuinely cannot spare a second pin, the options are:
 *    - Fit one sonar and set US_REAR_ENABLED to 0.
 *    - Put a 74HC4052 analog mux or a pair of 74HC125 tri-state buffers on
 *      the two TX lines, select one at a time from a GPIO. Works, but costs
 *      a part and a pin - the same pin you were trying to save.
 * ------------------------------------------------------------------------- */
#if US_FRONT_ENABLED && US_REAR_ENABLED
  #if PIN_US_FRONT_RX == PIN_US_REAR_RX
    #error "A2R_bot: both ultrasonics are on the same RX pin. The DYP-A21 frame carries no sensor ID, so front and rear readings are indistinguishable and the two transmitters also collide. Give the rear sonar its own pin (D2-D9 and D12 are free on header J18) and set PIN_US_REAR_RX, or fit one sonar and set US_REAR_ENABLED to 0. See the comment above this line."
  #endif
#endif
#else /* US_MODE_TRIG_ECHO */
  #define PIN_US_FRONT_TRIG     4
  #define PIN_US_FRONT_ECHO     5
  #define PIN_US_REAR_TRIG      6
  #define PIN_US_REAR_ECHO      7
  #define US_ECHO_TIMEOUT_US    30000UL
  #define US_SPEED_OF_SOUND     0.000343f  /* m per µs at 20 °C                */
#endif

/* ---------------------------------------------------------------------------
 *  10. SCHEDULER
 *      Cooperative, no RTOS, no delay() in the main loop. Every task states
 *      its own period; the loop runs them when they are due. Anything that
 *      blocks for more than a millisecond belongs in a state machine instead.
 * ------------------------------------------------------------------------- */
#define LOOP_DRIVE_HZ           100        /* servo/ESC refresh + failsafe     */
#define LOOP_SERIAL_HZ          200        /* RX drain — must outrun the Pi    */
#define STATUS_PERIOD_MS        5000       /* heartbeat line to the Pi         */

/* ---------------------------------------------------------------------------
 *  11. LINK ROBUSTNESS  (fw 3.4.0 - see SafeSerial.h for the full story)
 * ------------------------------------------------------------------------- */
/* Telemetry is assembled per line and sent only as fast as the 64-byte USB
 * FIFO has room. A line that cannot go out within this budget is dropped, so
 * a stalled host costs at most this much loop time per line - never forever. */
#define TX_LINE_MAX             512        /* longest single telemetry line   */
#define TX_LINE_BUDGET_US       3000       /* max time one line may take      */

/* Link beat: a tiny line at this rate, ALWAYS, whether or not any sensor is
 * fitted. It carries the sequence number of the last command received and
 * how long ago it arrived, so the Pi can measure the true command round-trip
 * and notice a dead link in ~1 s instead of waiting for the 5 s heartbeat. */
#define LINK_PUBLISH_HZ         20

/* Loop watchdog. If loop() ever stops running for this long - a stuck I2C
 * bus, a USB stall, anything - the chip reboots itself instead of hanging
 * until someone unplugs it. The PCA9685 keeps outputting its last pulse
 * while the ESP32 is frozen, so this bound IS the worst-case runaway time.
 * IDF 4.x (core 2.0.x) rounds this up to whole seconds. */
#define ENABLE_LOOP_WDT         1
#define LOOP_WDT_MS             1000

/* USB self-recovery (fw 3.4.2): commands still ARRIVING (last < RX ms ago)
 * but nothing has gone OUT for TX ms -> the USB send path is wedged
 * (typically after a supply dip) -> restart. Never in the first GRACE ms. */
#define USB_WEDGE_RX_MS         1000
#define USB_WEDGE_TX_MS         3000
#define USB_WEDGE_GRACE_MS      15000

#define FW_VERSION              "3.5.0-uart"

#if BOARD_IS_ESP32 && ENABLE_ULTRASONIC && (ULTRASONIC_MODE == US_MODE_UART) && US_FRONT_ENABLED
  #error "Serial0 (D0/D1) is the ROS link to the Jetson - the UART front sonar cannot share it"
#endif

#endif /* A2R_CONFIG_H */
