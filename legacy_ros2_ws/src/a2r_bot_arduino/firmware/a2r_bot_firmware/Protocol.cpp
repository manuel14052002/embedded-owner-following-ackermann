/* =============================================================================
 *  Protocol.cpp — implementation
 * ============================================================================= */

#include "Protocol.h"
#include "SafeSerial.h"

/* Defined in the .ino - re-probes the I2C bus and prints a named report. */
extern void scanI2C();
#include "Drive.h"
#include "Sensors.h"
#include "Battery.h"

namespace Protocol {

static char     s_rx[RX_BUFFER_LEN];
static uint16_t s_rx_len   = 0;
static bool     s_overflow = false;
static uint32_t s_frames   = 0;
static uint32_t s_errors   = 0;

static uint32_t s_next_imu = 0, s_next_us = 0, s_next_bat = 0;
static uint32_t s_next_baro = 0, s_next_status = 0;
static bool     s_last_estop = false;

/* Command sequence tracking for the link beat (fw 3.4.0). The Pi stamps every
 * DRIVE/STOP with "q"; we echo the last one back with how long ago it came. */
static long     s_last_q     = -1;
static uint32_t s_last_q_ms  = 0;
static uint32_t s_next_link  = 0;

static bool getInt(const char *json, const char *key, long &out);

static void noteSeq(const char *line, uint32_t now_ms) {
  long q;
  if (getInt(line, "q", q)) { s_last_q = q; s_last_q_ms = now_ms; }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Minimal JSON scanning
 * ══════════════════════════════════════════════════════════════════════════
 *  Not a parser — a finder. Locates "key" then walks past the colon and
 *  reads the value. Good enough for a flat object with known keys, and it
 *  cannot allocate, recurse or throw.
 * ══════════════════════════════════════════════════════════════════════════ */

static const char* findKey(const char *json, const char *key) {
  size_t klen = strlen(key);
  for (const char *p = json; *p; p++) {
    if (*p != '"') continue;
    if (strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
      const char *q = p + 2 + klen;
      while (*q == ' ' || *q == ':') q++;
      return q;
    }
  }
  return nullptr;
}

static bool getFloat(const char *json, const char *key, float &out) {
  const char *v = findKey(json, key);
  if (!v) return false;
  char *end = nullptr;
  float f = strtod(v, &end);
  if (end == v) return false;
  out = f;
  return true;
}

static bool getInt(const char *json, const char *key, long &out) {
  const char *v = findKey(json, key);
  if (!v) return false;
  char *end = nullptr;
  long n = strtol(v, &end, 10);
  if (end == v) return false;
  out = n;
  return true;
}

/* Copies a quoted string value into `out`, always NUL-terminated. */
static bool getString(const char *json, const char *key, char *out, size_t cap) {
  const char *v = findKey(json, key);
  if (!v || *v != '"') return false;
  v++;
  size_t i = 0;
  while (*v && *v != '"' && i < cap - 1) out[i++] = *v++;
  out[i] = '\0';
  return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Command dispatch
 * ══════════════════════════════════════════════════════════════════════════ */

static void handleLine(const char *line, uint32_t now_ms) {
  if (line[0] != '{') { s_errors++; return; }

  char cmd[16];
  if (!getString(line, "cmd", cmd, sizeof(cmd))) { s_errors++; return; }

  if (strcmp(cmd, "DRIVE") == 0) {
    float v = 0.0f, s = 0.0f;
    bool okv = getFloat(line, "v", v);
    bool oks = getFloat(line, "s", s);
    if (!okv && !oks) { s_errors++; return; }
    Drive::setCommand(v, s, now_ms);
    noteSeq(line, now_ms);
    s_frames++;

  } else if (strcmp(cmd, "STOP") == 0) {
    Drive::stop();
    noteSeq(line, now_ms);
    s_frames++;

  } else if (strcmp(cmd, "CAL_IMU") == 0) {
    Sensors::beginGyroCalibration();
    LINK.println(F("{\"status\":\"gyro calibration started - hold still\"}"));

  } else if (strcmp(cmd, "PING") == 0) {
    noteSeq(line, now_ms);
    publishStatus(now_ms, true);

  } else if (strcmp(cmd, "SCAN") == 0) {
    /* Re-probe the I2C bus on demand. The boot scan is the best diagnostic
     * the firmware produces, and on a native-USB board you cannot get back
     * to it without unplugging the cable. This makes it available any time. */
    scanI2C();
    publishStatus(now_ms, true);

  } else if (strcmp(cmd, "DISP") == 0) {
    char text[8];
    if (getString(line, "text", text, sizeof(text))) Battery::showText(text);

  } else if (strcmp(cmd, "DIM") == 0) {
    long level = MAX7219_INTENSITY;
    if (getInt(line, "level", level)) Battery::setIntensity((uint8_t)level);

  } else if (strcmp(cmd, "RESET_ENC") == 0) {
    /* Accepted and acknowledged for backwards compatibility with the older
     * planner, which sends this on startup. There is no encoder on this
     * vehicle — odometry comes from the IMU plus the commanded velocity. */
    LINK.println(F("{\"status\":\"no encoder fitted - ignored\"}"));

  } else {
    s_errors++;
  }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ══════════════════════════════════════════════════════════════════════════ */

void begin() {
#if BOARD_IS_ESP32
  LINK_PORT.setRxBufferSize(1024);
  LINK_PORT.setTxBufferSize(2048);
#endif
  LINK_PORT.begin(SERIAL_BAUD);
  s_rx_len = 0;
  s_overflow = false;
}

void poll(uint32_t now_ms) {
  while (LINK_PORT.available() > 0) {
    char c = (char)LINK_PORT.read();

    if (c == '\n' || c == '\r') {
      if (s_overflow) {
        /* The line was too long; we already threw it away. Reset and wait
         * for the next one rather than acting on a fragment. */
        s_overflow = false;
        s_rx_len = 0;
        s_errors++;
        continue;
      }
      if (s_rx_len > 0) {
        s_rx[s_rx_len] = '\0';
        handleLine(s_rx, now_ms);
        s_rx_len = 0;
      }
      continue;
    }

    if (s_rx_len >= RX_BUFFER_LEN - 1) {
      s_overflow = true;
      s_rx_len = 0;
      continue;
    }
    s_rx[s_rx_len++] = c;
  }
}

void publishImu(uint32_t now_ms) {
  if (!Sensors::imuPresent()) return;
  const uint32_t period = 1000UL / IMU_PUBLISH_HZ;
  if (now_ms < s_next_imu) return;
  s_next_imu = now_ms + period;

  const Sensors::ImuSample &s = Sensors::imu();
  /* Hand-built rather than sprintf'd: dtostrf/sprintf with floats pulls in a
   * large formatter and, on AVR, is slow enough to matter at 50 Hz. */
  LINK.print(F("{\"t\":"));      LINK.print(now_ms);
  LINK.print(F(",\"ax\":"));     LINK.print(s.ax, 4);
  LINK.print(F(",\"ay\":"));     LINK.print(s.ay, 4);
  LINK.print(F(",\"az\":"));     LINK.print(s.az, 4);
  LINK.print(F(",\"gx\":"));     LINK.print(s.gx, 5);
  LINK.print(F(",\"gy\":"));     LINK.print(s.gy, 5);
  LINK.print(F(",\"gz\":"));     LINK.print(s.gz, 5);
  if (s.mag_valid) {
    LINK.print(F(",\"mx\":"));   LINK.print(s.mx, 2);
    LINK.print(F(",\"my\":"));   LINK.print(s.my, 2);
    LINK.print(F(",\"mz\":"));   LINK.print(s.mz, 2);
  }
  LINK.print(F(",\"tc\":"));     LINK.print(s.temp_c, 1);
  LINK.println('}');
}

void publishRanges(uint32_t now_ms) {
#if ENABLE_ULTRASONIC
  const uint32_t period = 1000UL / US_PUBLISH_HZ;
  if (now_ms < s_next_us) return;
  s_next_us = now_ms + period;

  const Sensors::RangeSample &f = Sensors::rangeFront();
  const Sensors::RangeSample &r = Sensors::rangeRear();
  /* An invalid reading is reported as -1, never as max range. "I did not see
   * anything" and "the path is clear for 5 m" are different claims and the
   * fusion node treats them differently. */
  LINK.print(F("{\"t\":"));      LINK.print(now_ms);
  LINK.print(F(",\"us_f\":"));   LINK.print(f.valid ? f.range_m : -1.0f, 3);
  LINK.print(F(",\"us_r\":"));   LINK.print(r.valid ? r.range_m : -1.0f, 3);
  LINK.println('}');
#else
  (void)now_ms;
#endif
}

void publishBattery(uint32_t now_ms) {
  if (now_ms < s_next_bat) return;
  s_next_bat = now_ms + 500;                   /* 2 Hz is plenty             */
  if (!Battery::isValid()) return;

  LINK.print(F("{\"t\":"));         LINK.print(now_ms);
  LINK.print(F(",\"bat_v\":"));     LINK.print(Battery::packVoltage(), 3);
  LINK.print(F(",\"bat_pct\":"));   LINK.print(Battery::percent(), 1);
  LINK.print(F(",\"bat_cell\":"));  LINK.print(Battery::cellVoltage(), 3);
  LINK.print(F(",\"bat_mah\":"));   LINK.print(Battery::remainingMah(), 0);
  LINK.print(F(",\"bat_low\":"));   LINK.print(Battery::isLow() ? 1 : 0);
  LINK.println('}');
}

void publishBaro(uint32_t now_ms) {
#if ENABLE_BARO
  if (!Sensors::baroPresent()) return;
  const uint32_t period = 1000UL / BARO_PUBLISH_HZ;
  if (now_ms < s_next_baro) return;
  s_next_baro = now_ms + period;

  LINK.print(F("{\"t\":"));       LINK.print(now_ms);
  LINK.print(F(",\"press\":"));   LINK.print(Sensors::pressurePa(), 1);
  LINK.print(F(",\"btc\":"));     LINK.print(Sensors::baroTempC(), 2);
  LINK.println('}');
#else
  (void)now_ms;
#endif
}

uint32_t lastCmdMs() { return s_last_q < 0 ? 0 : s_last_q_ms; }

void publishLink(uint32_t now_ms) {
  const uint32_t period = 1000UL / LINK_PUBLISH_HZ;
  if (now_ms < s_next_link) return;
  s_next_link = now_ms + period;

  LINK.print(F("{\"lk\":"));  LINK.print(s_last_q);
  LINK.print(F(",\"qa\":"));  LINK.print(s_last_q < 0 ? -1L : (long)(now_ms - s_last_q_ms));
  LINK.print(F(",\"t\":"));   LINK.print(now_ms);
  LINK.print(F(",\"fs\":"));  LINK.print(Drive::isFailsafe() ? 1 : 0);
  LINK.println('}');
}

void publishStatus(uint32_t now_ms, bool force) {
  bool estop = Drive::isFailsafe();
  if (estop != s_last_estop) {
    s_last_estop = estop;
    LINK.print(F("{\"e_stop\":"));
    LINK.print(estop ? 1 : 0);
    LINK.println('}');
  }

  if (!force && now_ms < s_next_status) return;
  s_next_status = now_ms + STATUS_PERIOD_MS;

  /* One line that answers "is the hardware actually there and working?" —
   * the first thing to read when the robot does nothing. */
  LINK.print(F("{\"status\":\"heartbeat\",\"fw\":\"" FW_VERSION "\",\"board\":\""));
  LINK.print(F(BOARD_NAME));
  LINK.print(F("\",\"up_ms\":"));   LINK.print(now_ms);
#if defined(ARDUINO_ARCH_ESP32)
  LINK.print(F(",\"rst\":"));  LINK.print((int)esp_reset_reason());  /* why it last booted */
#endif
  LINK.print(F(",\"pca\":"));       LINK.print(Drive::isPcaOk() ? 1 : 0);
  LINK.print(F(",\"armed\":"));     LINK.print(Drive::isArmed() ? 1 : 0);
  LINK.print(F(",\"failsafe\":"));  LINK.print(Drive::isFailsafe() ? 1 : 0);
  LINK.print(F(",\"wdog\":"));      LINK.print(Drive::watchdogTripped() ? 1 : 0);
  LINK.print(F(",\"imu\":"));       LINK.print(Sensors::imuPresent() ? 1 : 0);
  LINK.print(F(",\"mag\":"));       LINK.print(Sensors::magPresent() ? 1 : 0);
  LINK.print(F(",\"baro\":"));      LINK.print(Sensors::baroPresent() ? 1 : 0);
  /* NOTE the _ok suffix. These are PRESENCE FLAGS (0/1); the range telemetry
   * line uses "us_f"/"us_r" for the actual DISTANCE IN METRES. Using the same
   * key for both meant any consumer keeping a dict of the latest values saw
   * the range replaced by a 0 or 1 once a second, and the sonar display
   * flickered between "1.23 m" and "1 m". Different meaning, different key. */
#if ENABLE_ULTRASONIC  /* heartbeat sonar fields - off: sonars omitted */
  LINK.print(F(",\"us_f_ok\":"));   LINK.print(Sensors::ultrasonicFrontPresent() ? 1 : 0);
  LINK.print(F(",\"us_r_ok\":"));   LINK.print(Sensors::ultrasonicRearPresent() ? 1 : 0);
  {
    /* Link counters: bytes seen / valid frames / checksum failures.
     * us_f_rx stuck at 0 = no signal reaching the pin at all. */
    uint32_t rx, frm, bad;
    Sensors::ultrasonicStats(0, rx, frm, bad);
    LINK.print(F(",\"us_f_rx\":"));  LINK.print(rx);
    LINK.print(F(",\"us_f_frm\":")); LINK.print(frm);
    LINK.print(F(",\"us_f_bad\":")); LINK.print(bad);
    Sensors::ultrasonicStats(1, rx, frm, bad);
    LINK.print(F(",\"us_r_rx\":"));  LINK.print(rx);
    LINK.print(F(",\"us_r_frm\":")); LINK.print(frm);
#if ULTRASONIC_MODE == US_MODE_UART
    /* Boot wire probe: front rx, front tx, rear rx, rear tx (1 = live). */
    LINK.print(F(",\"us_lvl\":\""));
    for (uint8_t i = 0; i < 4; i++) LINK.print(Sensors::ultrasonicIdle(i));
    LINK.print(F("\",\"us_pins\":\""));
    LINK.print(PIN_US_FRONT_RX); LINK.print(','); LINK.print(PIN_US_FRONT_TX);
    LINK.print(',');
    LINK.print(PIN_US_REAR_RX);  LINK.print(','); LINK.print(PIN_US_REAR_TX);
    LINK.print('"');
#endif
  }
#endif  /* ENABLE_ULTRASONIC heartbeat */
  LINK.print(F(",\"gyro_cal\":"));  LINK.print(Sensors::gyroCalibrated() ? 1 : 0);
  LINK.print(F(",\"imu_rec\":")); LINK.print(Sensors::imuRecoveries());
  LINK.print(F(",\"esc_us\":"));    LINK.print(Drive::escPulseUs());
  LINK.print(F(",\"srv_us\":"));    LINK.print(Drive::servoPulseUs());
  LINK.print(F(",\"rx\":"));        LINK.print(s_frames);
  LINK.print(F(",\"rx_err\":"));    LINK.print(s_errors);
  /* fw 3.4.0 link health: telemetry lines dropped because the host was not
   * draining USB, and lines cut short mid-way. Rising numbers = the Pi side
   * is stalling; zero = the link is clean. */
  LINK.print(F(",\"txdrop\":"));    LINK.print(LINK.drops());
  LINK.print(F(",\"torn\":"));      LINK.print(LINK.torn());
  LINK.println('}');
}

uint32_t rxFrames() { return s_frames; }
uint32_t rxErrors() { return s_errors; }

}  /* namespace Protocol */
