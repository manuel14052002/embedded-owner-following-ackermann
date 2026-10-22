/* =============================================================================
 *
 *      █████╗ ██████╗ ██████╗     ██████╗  ██████╗ ████████╗
 *     ██╔══██╗╚════██╗██╔══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
 *     ███████║ █████╔╝██████╔╝    ██████╔╝██║   ██║   ██║
 *     ██╔══██║██╔═══╝ ██╔══██╗    ██╔══██╗██║   ██║   ██║
 *     ██║  ██║███████╗██║  ██║    ██████╔╝╚██████╔╝   ██║
 *     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═════╝  ╚═════╝    ╚═╝
 *
 *      a2r_bot_firmware.ino — real-time layer, Arduino Nano ESP32
 *      firmware v3.0.0 · part of a2r_bot_ws
 *
 * =============================================================================
 *  WHERE THIS SITS
 *  ---------------
 *      Jetson Nano   vision, wake word, gestures, the PET state machine
 *            │  /pet/state  +  /cmd_vel   (ROS 2 over WiFi)
 *            ▼
 *      Raspberry Pi  sensor fusion, local planning, SLAM, the robot's face
 *            │  {"cmd":"DRIVE","v":..,"s":..}   (USB serial, 921600)
 *            ▼
 *      THIS BOARD    ESC + servo + IMU + sonar + battery gauge + failsafe
 *            │  I2C / PWM / ADC
 *            ▼
 *      PCA9685 → Hobbywing 880 ESC (ch1) and steering servo (ch0)
 *
 *  THE ONE RULE
 *  ------------
 *  Nothing in the main loop may block. There is no delay() below the setup
 *  banner, and every task is a period check plus a small amount of work. The
 *  drive task runs at 100 Hz whatever else happens, because that task is the
 *  only thing standing between a Pi crash and a car that keeps going.
 *
 *  FIRST BRING-UP, IN ORDER
 *  ------------------------
 *   1. Flash with the ESC UNPLUGGED from the battery and the car on a stand.
 *   2. Open the serial monitor at 115200. You should see a boot banner and
 *      then a heartbeat line every 5 s. Check "pca":1 — if it is 0, nothing
 *      will move and the problem is I2C wiring, not software.
 *   3. Check "imu":1, "us_f":1, "us_r":1. Anything at 0 was not found on the
 *      bus; `i2cdetect`-style scan output is printed once at boot to help.
 *   4. Confirm the 7-segment shows a plausible pack voltage. Compare with a
 *      multimeter and adjust BATTERY_CAL_GAIN in Config.h.
 *   5. Send  {"cmd":"DRIVE","v":0,"s":0.5}  and watch the servo. Then -0.5.
 *      Trim SERVO_CENTER_US until "s":0 points dead ahead.
 *   6. Only now connect the ESC. It arms 3 s after power-up.
 *      Send  {"cmd":"DRIVE","v":0.2,"s":0}  with the wheels off the ground.
 *
 *  BOARD SETTINGS (Arduino IDE)
 *  ----------------------------
 *      Board:  Arduino Nano ESP32          (or Arduino Nano for AVR)
 *      Port:   the one that appears when you plug the board in
 *      No external libraries are required. Wire.h ships with the core.
 * ============================================================================= */

#include <Wire.h>

/* ---------------------------------------------------------------------------
 *  SANITY GUARD: are the other nine files actually next to this one?
 *
 *  The Arduino IDE compiles every .cpp and .h that sits BESIDE the .ino, in
 *  the same folder. If the .ino has been moved into a subfolder of its own,
 *  its siblings are no longer visible and you get:
 *
 *      fatal error: Drive.h: No such file or directory
 *
 *  which says nothing at all about the real cause.
 *
 *  This happens because the IDE sometimes offers to "create a folder for this
 *  sketch and move the file", and saying yes moves ONLY the .ino - the nine
 *  support files stay where they were. The tell-tale sign is a doubled path:
 *
 *      .../firmware/a2r_bot_firmware/a2r_bot_firmware/a2r_bot_firmware.ino
 *                                    ^^^^^^^^^^^^^^^^ one folder too deep
 *
 *  FIX: move the .ino back up one level and delete the empty folder.
 *
 *      Windows:  cd ...\firmware\a2r_bot_firmware
 *                move a2r_bot_firmware\a2r_bot_firmware.ino .
 *                rmdir a2r_bot_firmware
 *
 *      Linux:    cd .../firmware/a2r_bot_firmware
 *                mv a2r_bot_firmware/a2r_bot_firmware.ino .
 *                rmdir a2r_bot_firmware
 *
 *  When it is right, the IDE shows TEN tabs across the top of the window:
 *  a2r_bot_firmware, Battery.cpp/.h, Config.h, Drive.cpp/.h, Protocol.cpp/.h,
 *  Sensors.cpp/.h. One tab means it is still in the wrong place.
 * ------------------------------------------------------------------------- */
#if defined(__has_include)
#  if !__has_include("Config.h") || !__has_include("Drive.h") || \
      !__has_include("Sensors.h") || !__has_include("Battery.h") || \
      !__has_include("Protocol.h")
#    error "A2R_bot: the .ino is not in the same folder as Config.h, Drive.h, Sensors.h, Battery.h and Protocol.h. The IDE has probably moved it into a subfolder of its own - move it back up one level. See the comment directly above this line."
#  endif
#endif

#include "Config.h"
#include "Drive.h"
#include "Sensors.h"
#include "Battery.h"
#include "Protocol.h"
#include "SafeSerial.h"

#if BOARD_IS_ESP32 && ENABLE_LOOP_WDT
  #include <esp_task_wdt.h>
  #include <esp_idf_version.h>
#endif

/* Feed the loop watchdog. Called once per loop(), and from inside anything
 * that legitimately runs long (the I2C audit) so it is not mistaken for a
 * hang. A no-op when the watchdog is disabled or on AVR. */
static inline void feedWatchdog() {
#if BOARD_IS_ESP32 && ENABLE_LOOP_WDT
  esp_task_wdt_reset();
#endif
}

/* Arm the loop watchdog (fw 3.4.0). See LOOP_WDT_MS in Config.h. Armed at the
 * END of setup(), because setup() legitimately blocks for well over a second
 * (USB wait, display self-test, IMU reset). */
static void startWatchdog() {
#if BOARD_IS_ESP32 && ENABLE_LOOP_WDT
  #if ESP_IDF_VERSION_MAJOR >= 5
    esp_task_wdt_config_t cfg = {};
    cfg.timeout_ms     = LOOP_WDT_MS;
    cfg.idle_core_mask = 0;
    cfg.trigger_panic  = true;
    if (esp_task_wdt_reconfigure(&cfg) != ESP_OK) esp_task_wdt_init(&cfg);
  #else
    /* IDF 4.x: re-initialising an already-running TWDT just updates it. */
    esp_task_wdt_init((LOOP_WDT_MS + 999) / 1000, true);
  #endif
  esp_task_wdt_add(NULL);           /* this task = setup()/loop() */
#endif
}

/* ── Scheduler bookkeeping ─────────────────────────────────────────────── */
static uint32_t s_next_drive  = 0;
static uint32_t s_loop_count  = 0;
static uint32_t s_worst_loop_us = 0;
static uint32_t s_worst_stage_us = 0;

/* Prints every responding address on the I2C bus, once, at boot. Two minutes
 * of staring at this line has saved more debugging time than any other
 * feature in this firmware: if the PCA9685 is not at 0x40 here, no amount of
 * reading the planner logs will help. */
/* ---------------------------------------------------------------------------
 *  I2C bus audit at boot.
 *
 *  Everything on this robot except the display shares SDA/SCL: two PCA9685s,
 *  the GY-91's three separate dies, and both DYP-A21 sonars. Seven addresses
 *  on one bus, and the failure modes are all silent:
 *
 *    - two DYP-A21s ship at the SAME address, and unless one is reprogrammed
 *      they corrupt each other's replies. The symptom is a front sonar that
 *      occasionally reports the rear distance, which is far harder to diagnose
 *      than a missing device.
 *    - a second PCA9685 without its A0 jumper bridged answers at 0x40 along
 *      with the first, so both boards drive both channels.
 *    - the AK8963 hides behind the MPU9250's pass-through and simply will not
 *      appear until that is enabled.
 *
 *  So the scan does not just list addresses - it names what it found, says
 *  what is missing, and flags the specific conflicts above.
 * ------------------------------------------------------------------------- */
const char *i2cName(uint8_t a) {
  switch (a) {
    case 0x0C: return "AK8963 magnetometer";
#if ULTRASONIC_MODE == US_MODE_I2C
    case 0x11: return "DYP-A21 front";
    case 0x12: return "DYP-A21 rear";
#else
    case 0x11: return "UNEXPECTED (sonars are on UART, not I2C)";
    case 0x12: return "UNEXPECTED (sonars are on UART, not I2C)";
#endif
    case 0x40: return "PCA9685 #1 (servo ch0, ESC ch1)";
    case 0x41: return "PCA9685 #2";
    case 0x68: return "MPU9250 accel+gyro";
    case 0x69: return "MPU9250 (AD0 high)";
    case 0x76: return "BMP280 barometer";
    case 0x77: return "BMP280 (alt address)";
    case 0x70: return "PCA9685 ALLCALL (harmless)";
    default:   return "UNEXPECTED";
  }
}

bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

/* Not static: Protocol.cpp calls this for the {"cmd":"SCAN"} command, so the
 * bus can be re-probed at runtime. That matters because the Arduino Nano
 * ESP32 enumerates over native USB-CDC, and toggling DTR does NOT reset it
 * the way it does on a classic Nano - so "just reboot and read the boot scan"
 * is not actually available without physically replugging the cable. */
void scanI2C() {
  bool seen[128];
  uint8_t found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    if ((addr & 0x0F) == 0) feedWatchdog();   /* a stuck bus: 25 ms/addr */
    seen[addr] = i2cPresent(addr);
    if (seen[addr]) found++;
  }

  LINK.println(F("{\"status\":\"--- i2c bus audit ---\"}"));
  for (uint8_t addr = 1; addr < 127; addr++) {
    if (!seen[addr]) continue;
    LINK.print(F("{\"status\":\"  0x"));
    if (addr < 16) LINK.print('0');
    LINK.print(addr, HEX);
    LINK.print(F("  "));
    LINK.print(i2cName(addr));
    LINK.println(F("\"}"));
  }

  if (found == 0) {
    LINK.println(F("{\"status\":\"  NOTHING FOUND - check SDA/SCL, common "
                     "ground, and pull-ups\"}"));
    return;
  }

  /* ---- conflicts and omissions ---------------------------------------- */
  uint8_t problems = 0;

#if ULTRASONIC_MODE == US_MODE_I2C
  if (seen[0x11] && !seen[0x12]) {
    LINK.println(F("{\"status\":\"  WARNING only one sonar at 0x11. If BOTH "
                     "are plugged in, the second was never reprogrammed to "
                     "0x12 and they are colliding. See WIRING.md 9.6\"}"));
    problems++;
  }
  if (!seen[0x11] && !seen[0x12]) {
    LINK.println(F("{\"status\":\"  WARNING no sonars on I2C. Wrong DYP-A21 "
                     "variant, or unpowered\"}"));
    problems++;
  }
#else
  /* The sonars are on their own UARTs, not this bus. Anything answering at
   * 0x11 or 0x12 is therefore a surprise worth mentioning - most likely an
   * I2C-variant sonar still wired to SDA/SCL while ULTRASONIC_MODE says UART,
   * which would leave both range topics permanently silent. */
  if (seen[0x11] || seen[0x12]) {
    LINK.println(F("{\"status\":\"  WARNING something is answering at 0x11 or "
                     "0x12, but ULTRASONIC_MODE is UART. Is an I2C sonar still "
                     "wired to SDA/SCL? It will never be read.\"}"));
    problems++;
  }
#endif
  if (!seen[0x40]) {
    LINK.println(F("{\"status\":\"  ERROR no PCA9685 at 0x40 - the car "
                     "cannot steer or drive\"}"));
    problems++;
  }
#if PCA9685_II_ENABLED
  if (!seen[0x41]) {
    LINK.println(F("{\"status\":\"  ERROR PCA9685_II_ENABLED but nothing at "
                     "0x41 - bridge its A0 solder jumper\"}"));
    problems++;
  }
#endif
  if (!seen[0x68] && !seen[0x69]) {
    LINK.println(F("{\"status\":\"  ERROR no MPU9250 - heading and the EKF "
                     "will not work\"}"));
    problems++;
  }
  if (!seen[0x0C]) {
    LINK.println(F("{\"status\":\"  note AK8963 not visible yet - it sits "
                     "behind MPU9250 pass-through and appears after init\"}"));
  }

  LINK.print(F("{\"status\":\"  "));
  LINK.print(found);
  LINK.print(F(" device(s), "));
  LINK.print(problems);
  LINK.println(F(" problem(s)\"}"));
}

/* ═════════════════════════════════════════════════════════════════════════
 *  SETUP
 * ═════════════════════════════════════════════════════════════════════════ */
void setup() {
  Protocol::begin();

  /* ── SAFETY FIRST (fw 3.4.0) ───────────────────────────────────────────
   * The PCA9685 is a separate chip that stays powered through an ESP32
   * reset and keeps outputting the LAST servo/ESC pulse it was given. After
   * a watchdog reboot that pulse may be "drive forward". So the very first
   * thing we do - before waiting for USB, before the display self-test - is
   * free the I2C bus and park both channels. Everything below can take its
   * time; the car is already stopped. */
  Sensors::i2cRecoverBus();              /* also Wire.begin/clock/timeout */
  const bool pca_ok = Drive::begin();    /* servo centre, ESC neutral     */

  /* Give USB-CDC on the ESP32-S3 a moment to enumerate, but never hang
   * waiting for a host that may not be there — the car must boot standalone. */
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 1500) { /* spin */ }

  LINK.println();
  LINK.println(F("{\"status\":\"A2R_bot firmware v" FW_VERSION " booting\"}"));

  /* fw 3.4.0: a slave left mid-byte by the last reset can hold SDA low and
   * make every I2C device look dead. Free the bus before first use. */
  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);    /* see Config.h - 100 kHz by default, and why */

#if BOARD_IS_ESP32
  /* ---------------------------------------------------------------------
   *  I2C TIMEOUT — without this, one flaky wire can freeze the board.
   *
   *  The symptom that led here: intermittent GY-91 reads, a "WARNING slow
   *  loop" status line, then total serial silence until the USB cable was
   *  unplugged and replugged. That sequence is the signature of I2C clock
   *  stretching with no timeout: a device with a marginal connection can
   *  hold SDA or SCL in a bad state mid-transaction, and Wire's default
   *  behaviour on the ESP32 is to wait. Not "wait a while" - wait. The main
   *  loop lives inside that Wire call, so serial, the drive-command
   *  failsafe, and everything else stop at the same instant, for as long
   *  as the bad connection persists. Reseating a wire, or the vibration
   *  from the servo/ESC test itself, can flip a marginal connection from
   *  "answers slowly" to "never releases the bus".
   *
   *  setTimeOut() bounds how long any single Wire call is allowed to block.
   *  On timeout it aborts the transaction and Wire.endTransmission() simply
   *  returns an error, which the code already checks - no logic elsewhere
   *  has to change. A stuck bus now costs one bad reading instead of the
   *  whole board.
   * --------------------------------------------------------------------- */
  Wire.setTimeOut(25);             /* ms; well under the 8 ms loop budget */
#endif
  delay(50);

  scanI2C();

  /* ── Battery and display first ──────────────────────────────────────────
   * The display is the only output device on the robot that works with no
   * Pi, no WiFi and no serial monitor attached, so bringing it up first
   * means every later failure has somewhere to be reported. */
  Battery::begin();

  /* ── Drive ─────────────────────────────────────────────────────────────
   * This parks the servo and starts the ESC arming timer. Nothing can be
   * commanded until arming finishes. */
  if (!pca_ok) {
    LINK.println(F("{\"status\":\"ERROR: PCA9685 not responding at 0x40 - "
                     "car will not move. Check SDA/SCL, 5V and the address "
                     "solder jumpers.\"}"));
    Battery::showText("Err1");
    delay(1500);
  }

  /* ── Sensors ───────────────────────────────────────────────────────── */
  if (!Sensors::begin()) {
    LINK.println(F("{\"status\":\"WARNING: IMU not found - odometry will "
                     "fall back to command integration only\"}"));
  }

  LINK.print(F("{\"status\":\"boot complete\",\"board\":\""));
  LINK.print(F(BOARD_NAME));
  LINK.print(F("\",\"esc_ch\":"));    LINK.print(CH_ESC);
  LINK.print(F(",\"servo_ch\":"));    LINK.print(CH_SERVO);
  LINK.print(F(",\"cells\":"));       LINK.print(BATTERY_CELLS);
  LINK.print(F(",\"arm_ms\":"));      LINK.print(ESC_ARM_MS);
  LINK.println(F("}"));

  uint32_t now = millis();
  s_next_drive  = now;

  startWatchdog();
  LINK.print(F("{\"status\":\"loop watchdog "));
  LINK.print(ENABLE_LOOP_WDT ? F("armed ") : F("OFF "));
  LINK.print(LOOP_WDT_MS);
  LINK.println(F(" ms\"}"));
}

/* ═════════════════════════════════════════════════════════════════════════
 *  LOOP
 * ═════════════════════════════════════════════════════════════════════════
 *  Ordered by how badly it matters if the task is late:
 *
 *    1. serial RX    a dropped DRIVE frame becomes a failsafe stop
 *    2. drive        pulses and the command timeout
 *    3. IMU          50 Hz, feeds the Pi's EKF
 *    4. ultrasonic   10 Hz, alternating
 *    5. battery      5 Hz sample, 4 Hz display
 *    6. baro/status  whenever there is time left over
 * ═════════════════════════════════════════════════════════════════════════ */
/* Which named stage was running when the loop's worst-case time was set.
 * Recorded so "WARNING slow loop" can name the culprit instead of just the
 * total - which matters a great deal when the cause is one flaky I2C device
 * among five, because "it was slow" and "it was the GY-91 specifically" are
 * a ten-second fix apart versus an afternoon of reseating every connector. */
static const char *s_worst_stage = "?";

void loop() {
  feedWatchdog();
#if BOARD_IS_ESP32
  { /* fw 3.4.2: USB send path wedged while the Pi is still talking -> restart */
    const uint32_t n = millis(), c = Protocol::lastCmdMs();
    if (c && n > USB_WEDGE_GRACE_MS && (uint32_t)(n - c) < USB_WEDGE_RX_MS &&
        (uint32_t)(n - LINK.lastOkMs()) > USB_WEDGE_TX_MS) {
      esp_restart();          /* setup() parks servo + ESC first */
    }
  }
#endif
  uint32_t loop_t0 = micros();
  uint32_t now = millis();
  const char *stage = "rx";
  (void)stage;   /* reserved: rx/drive stages are fast enough not to need
                  * individual timing, unlike imu/ultrasonic/baro above,
                  * which are the ones actually seen going slow in practice */

  /* 1. Serial RX — highest priority, and now UNCONDITIONAL.
   *
   *    This used to be rate-limited to LOOP_SERIAL_HZ, which put up to 5 ms of
   *    avoidable latency in front of every DRIVE frame for no benefit: poll()
   *    costs nothing when the buffer is empty (one Serial.available() call),
   *    and the whole point of this task is to be the thing that never waits.
   *
   *    It is also called AGAIN between the telemetry emitters further down.
   *    Telemetry is the only thing in this loop that can occupy the link for a
   *    meaningful time, so draining RX on both sides of it is what stops an
   *    outgoing IMU line from delaying an incoming steering command. */
  Protocol::poll(now);

  /* 2. Drive — pulses, slew limits, arming, reverse handshake, failsafe. */
  stage = "drive";
  if (now >= s_next_drive) {
    s_next_drive = now + (1000UL / LOOP_DRIVE_HZ);
    Drive::update(now);
  }

  /* 3. Sensors. Each of these rate-limits itself internally.
   * Timed individually with a running micros() checkpoint, rather than only
   * timing the loop as a whole - see s_worst_stage above. */
  uint32_t stage_t0 = micros();
  stage = "imu";      Sensors::updateImu(now);
  if (micros() - stage_t0 > s_worst_stage_us) {
    s_worst_stage_us = micros() - stage_t0; s_worst_stage = "imu";
  }
  stage_t0 = micros();
  /* ultrasonic stage omitted - no sonars fitted (a2r_fix_final.sh) */
  stage_t0 = micros();
  stage = "baro";     Sensors::updateBaro(now);
  if (micros() - stage_t0 > s_worst_stage_us) {
    s_worst_stage_us = micros() - stage_t0; s_worst_stage = "baro";
  }

  /* Re-estimate gyro bias whenever the car is commanded to a standstill.
   * Bias walks with temperature and the chassis warms up several degrees in
   * the first few minutes of running, so a single boot-time calibration is
   * stale by the time it matters. */
  Sensors::updateGyroCalibration(fabsf(Drive::throttleFraction()) < 0.01f);

  /* 4. Battery. The applied throttle drives the sag compensation, so pass
   *    what the motor actually got, not what the Pi asked for. */
  Battery::update(now, Drive::throttleFraction());
  Battery::updateDisplay(now);

  /* 5. Telemetry out, with an RX drain between each emitter.
   *
   *    Each publishX() is internally rate-limited, so on most iterations these
   *    are all no-ops. But when several fall due in the same millisecond -
   *    which happens regularly, because 50 Hz, 25 Hz, 10 Hz and 5 Hz all share
   *    factors - the board would otherwise push ~170 bytes back to back with
   *    no chance to notice a DRIVE frame that arrived meanwhile.
   *
   *    Interleaving poll() costs one Serial.available() per emitter and bounds
   *    the worst-case command latency to a single telemetry line rather than a
   *    whole burst of them. */
  Protocol::publishImu(now);
  Protocol::poll(now);
  Protocol::publishLink(now);
  /* range telemetry omitted with the sonars */
  Protocol::publishBattery(now);
  Protocol::publishBaro(now);
  Protocol::poll(now);
  Protocol::publishStatus(now);

  /* Loop-time watchdog. If the worst-case loop ever approaches the 10 ms
   * drive period, the drive task is at risk of jitter and something above
   * has started blocking. Reported so it is visible before it bites. */
  uint32_t dt_us = micros() - loop_t0;
  if (dt_us > s_worst_loop_us) s_worst_loop_us = dt_us;
  if (++s_loop_count >= 20000) {
    if (s_worst_loop_us > 8000) {
      /* Names the stage, not just the total. If this consistently points at
       * "imu", the GY-91's I2C connection is the thing to reseat/replace -
       * not the PCA9685, not the sonars, not the wiring in general. */
      LINK.print(F("{\"status\":\"WARNING slow loop\",\"worst_us\":"));
      LINK.print(s_worst_loop_us);
      LINK.print(F(",\"stage\":\""));
      LINK.print(s_worst_stage);
      LINK.println(F("\"}"));
    }
    s_loop_count = 0;
    s_worst_loop_us = 0;
    s_worst_stage_us = 0;
    s_worst_stage = "?";
  }
}
