/* =============================================================================
 *  Sensors.cpp — implementation
 * ============================================================================= */

#include <Wire.h>
#include "Sensors.h"
#include "SafeSerial.h"

#if ULTRASONIC_MODE == US_MODE_UART && !BOARD_IS_ESP32
  #include <SoftwareSerial.h>
  static SoftwareSerial usFront(PIN_US_FRONT_RX, PIN_US_FRONT_TX);
  static SoftwareSerial usRear (PIN_US_REAR_RX,  PIN_US_REAR_TX);
#endif

namespace Sensors {

/* ── Module state ───────────────────────────────────────────────────────── */
static ImuSample   s_imu   = {0,0,9.80665f, 0,0,0, 0,0,0, 25.0f, false};
static RangeSample s_front = {0.0f, false, 0};
static RangeSample s_rear  = {0.0f, false, 0};

static bool s_imu_ok = false, s_mag_ok = false, s_baro_ok = false;
/* IMU dropout recovery (fw 3.1.2). One failed read used to latch the IMU off
 * until reboot; now it takes 5 in a row, and re-init is retried every second. */
static uint8_t  s_imu_fail = 0;
static uint32_t s_imu_retry_ms = 0;
static uint32_t s_imu_recoveries = 0;
/* Sonar wire idle levels probed at boot, before the UARTs claim the pins:
 * [front rx, front tx, rear rx, rear tx]. 1 = held HIGH against the internal
 * pull-down (a powered device drives it), 0 = low (nothing powered on it),
 * 2 = not probed (not UART mode / not ESP32). */
static uint8_t  s_us_idle[4] = {2, 2, 2, 2};
static bool s_us_front_ok = false, s_us_rear_ok = false;

/* Sonar link statistics, [0] = front, [1] = rear. Reported in the heartbeat
 * so "-1 forever" can be split into its two very different causes:
 *   rx == 0            nothing is arriving on the pin at all
 *                      (wrong variant, TX on the wrong pin, no power,
 *                       UART-controlled variant never being asked)
 *   rx > 0, frm == 0   bytes arrive but never a valid frame
 *                      (wrong baud, bad level/divider, noise, inverted)  */
static uint32_t s_us_rx[2]  = {0, 0};
static uint32_t s_us_frm[2] = {0, 0};
static uint32_t s_us_bad[2] = {0, 0};

static float s_mag_adj[3] = {1.0f, 1.0f, 1.0f};   /* AK8963 fuse-ROM sensitivity */

static float s_gyro_bias[3] = {0.0f, 0.0f, 0.0f};
static bool  s_gyro_cal_done = false;
static bool  s_gyro_cal_run  = false;
static uint16_t s_gyro_cal_n = 0;
static float s_gyro_cal_sum[3] = {0.0f, 0.0f, 0.0f};

static uint32_t s_imu_next = 0, s_baro_next = 0;
/* Only the modes that actively trigger a ping use these. In UART auto-report
 * mode the sensors free-run, so there is nothing to schedule. */
#if ULTRASONIC_MODE != US_MODE_UART || US_UART_AUTO_REPORT == 0
static uint32_t s_us_next = 0;
#endif
#if ULTRASONIC_MODE != US_MODE_UART || US_UART_AUTO_REPORT == 0
static uint8_t  s_us_turn  = 0;     /* 0 = front, 1 = rear                   */
#endif

static float s_press_pa = 101325.0f;
static float s_baro_t   = 25.0f;

/* BMP280 factory calibration, read once at begin() */
static uint16_t dig_T1, dig_P1;
static int16_t  dig_T2, dig_T3, dig_P2, dig_P3, dig_P4, dig_P5,
                dig_P6, dig_P7, dig_P8, dig_P9;
static int32_t  t_fine = 0;

/* ══════════════════════════════════════════════════════════════════════════
 *  Minimal I2C helpers
 * ══════════════════════════════════════════════════════════════════════════ */

static bool i2cWrite8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n)   return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

static uint8_t i2cRead8(uint8_t addr, uint8_t reg) {
  uint8_t v = 0;
  return i2cRead(addr, reg, &v, 1) ? v : 0xFF;
}

#if BOARD_IS_ESP32 && ULTRASONIC_MODE == US_MODE_UART
/* 1 if the pin sits mostly HIGH for 20 ms with the internal pull-down on.
 * A UART line idles high, so a POWERED sonar's TX wire reads 1; a sonar with
 * no power, a broken wire or a wire on some other pin reads 0. */
static uint8_t usProbeIdle(uint8_t pin) {
  pinMode(pin, INPUT_PULLDOWN);
  delay(2);
  uint16_t hi = 0;
  for (uint16_t i = 0; i < 200; i++) {
    hi += digitalRead(pin) ? 1 : 0;
    delayMicroseconds(100);
  }
  pinMode(pin, INPUT);                  /* leave no pull for the UART */
  return hi > 150 ? 1 : 0;
}
#endif

void i2cRecoverBus() {
  Wire.end();
  pinMode(SDA, INPUT_PULLUP);
  pinMode(SCL, INPUT_PULLUP);
  delayMicroseconds(10);
  for (uint8_t i = 0; i < 9 && digitalRead(SDA) == LOW; i++) {
    pinMode(SCL, OUTPUT); digitalWrite(SCL, LOW);  delayMicroseconds(10);
    pinMode(SCL, INPUT_PULLUP);                     delayMicroseconds(10);
  }
  /* STOP condition: SDA low -> high while SCL is high */
  pinMode(SDA, OUTPUT); digitalWrite(SDA, LOW);     delayMicroseconds(10);
  pinMode(SCL, INPUT_PULLUP);                       delayMicroseconds(10);
  pinMode(SDA, INPUT_PULLUP);                       delayMicroseconds(10);

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
#if BOARD_IS_ESP32
  Wire.setTimeOut(25);
#endif
}

static bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MPU-9250 (accelerometer + gyroscope)
 * ══════════════════════════════════════════════════════════════════════════ */

static bool mpuBegin() {
  if (!i2cPresent(MPU9250_ADDR)) return false;

  uint8_t who = i2cRead8(MPU9250_ADDR, MPU_WHO_AM_I);
  /* 0x71 MPU-9250, 0x73 MPU-9255, 0x70 MPU-6500 (GY-91 clones ship all three;
   * the accel/gyro half is register-compatible, only the mag differs). */
  if (who != 0x71 && who != 0x73 && who != 0x70 && who != 0x68) return false;

  i2cWrite8(MPU9250_ADDR, MPU_PWR_MGMT_1, 0x80);   /* reset                  */
  delay(100);
  i2cWrite8(MPU9250_ADDR, MPU_PWR_MGMT_1, 0x01);   /* PLL from X gyro        */
  delay(20);
  i2cWrite8(MPU9250_ADDR, MPU_PWR_MGMT_2, 0x00);   /* all axes on            */

  /* DLPF 41 Hz on both. An RC car's useful signal is well under 20 Hz;
   * everything above that is chassis buzz and drivetrain vibration, and
   * filtering it in the chip is free. */
  i2cWrite8(MPU9250_ADDR, MPU_CONFIG,        0x03);
  i2cWrite8(MPU9250_ADDR, MPU_ACCEL_CONFIG2, 0x03);
  i2cWrite8(MPU9250_ADDR, MPU_SMPLRT_DIV,    0x04);  /* 1 kHz / 5 = 200 Hz   */
  i2cWrite8(MPU9250_ADDR, MPU_GYRO_CONFIG,   GYRO_FS_SEL);
  i2cWrite8(MPU9250_ADDR, MPU_ACCEL_CONFIG,  ACCEL_FS_SEL);

  /* Bypass mode: puts the AK8963 magnetometer directly on the main I2C bus
   * instead of behind the MPU's own master. Simpler, and it means a failed
   * magnetometer cannot wedge the accel/gyro path. */
  i2cWrite8(MPU9250_ADDR, MPU_INT_PIN_CFG, 0x02);
  delay(10);
  return true;
}

static bool mpuRead(float &ax, float &ay, float &az,
                    float &gx, float &gy, float &gz, float &temp) {
  uint8_t b[14];
  if (!i2cRead(MPU9250_ADDR, MPU_ACCEL_XOUT_H, b, 14)) return false;

  int16_t raw_ax = (int16_t)((b[0]  << 8) | b[1]);
  int16_t raw_ay = (int16_t)((b[2]  << 8) | b[3]);
  int16_t raw_az = (int16_t)((b[4]  << 8) | b[5]);
  int16_t raw_t  = (int16_t)((b[6]  << 8) | b[7]);
  int16_t raw_gx = (int16_t)((b[8]  << 8) | b[9]);
  int16_t raw_gy = (int16_t)((b[10] << 8) | b[11]);
  int16_t raw_gz = (int16_t)((b[12] << 8) | b[13]);

  ax = (float)raw_ax * ACCEL_SCALE;
  ay = (float)raw_ay * ACCEL_SCALE;
  az = (float)raw_az * ACCEL_SCALE;
  gx = (float)raw_gx * GYRO_SCALE;
  gy = (float)raw_gy * GYRO_SCALE;
  gz = (float)raw_gz * GYRO_SCALE;
  temp = (float)raw_t / 333.87f + 21.0f;     /* datasheet 4.19               */
  return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AK8963 magnetometer (inside the MPU-9250 package)
 * ══════════════════════════════════════════════════════════════════════════ */

#if ENABLE_MAG
static bool akBegin() {
  if (!i2cPresent(AK8963_ADDR)) return false;
  if (i2cRead8(AK8963_ADDR, AK_WHO_AM_I) != 0x48) return false;

  i2cWrite8(AK8963_ADDR, AK_CNTL1, 0x00);      /* power down                 */
  delay(10);
  i2cWrite8(AK8963_ADDR, AK_CNTL1, 0x0F);      /* fuse ROM access            */
  delay(10);
  uint8_t asa[3];
  if (i2cRead(AK8963_ADDR, AK_ASAX, asa, 3)) {
    /* Per-axis sensitivity adjustment, datasheet 8.3.11:
     *     Hadj = H * ((ASA - 128) * 0.5 / 128 + 1)                          */
    for (uint8_t i = 0; i < 3; i++)
      s_mag_adj[i] = ((float)asa[i] - 128.0f) * 0.5f / 128.0f + 1.0f;
  }
  i2cWrite8(AK8963_ADDR, AK_CNTL1, 0x00);
  delay(10);
  i2cWrite8(AK8963_ADDR, AK_CNTL1, 0x16);      /* 16-bit, continuous 100 Hz  */
  delay(10);
  return true;
}

/* NOTE ON AXES: the AK8963 die is rotated relative to the MPU's accel/gyro
 * axes. Per the MPU-9250 datasheet (9.1), mag X aligns with accel Y, mag Y
 * with accel X, and mag Z is inverted. We correct that HERE so that
 * everything leaving this board is in one consistent body frame and the Pi
 * never has to know the part number. */
static bool akRead(float &mx, float &my, float &mz) {
  uint8_t st1 = i2cRead8(AK8963_ADDR, AK_ST1);
  if (!(st1 & 0x01)) return false;                 /* no new data            */

  uint8_t b[7];                                    /* 6 data + ST2           */
  if (!i2cRead(AK8963_ADDR, AK_HXL, b, 7)) return false;
  if (b[6] & 0x08) return false;                   /* magnetic overflow      */

  int16_t rx = (int16_t)((b[1] << 8) | b[0]);      /* little-endian here!    */
  int16_t ry = (int16_t)((b[3] << 8) | b[2]);
  int16_t rz = (int16_t)((b[5] << 8) | b[4]);

  float hx = (float)rx * MAG_SCALE * s_mag_adj[0];
  float hy = (float)ry * MAG_SCALE * s_mag_adj[1];
  float hz = (float)rz * MAG_SCALE * s_mag_adj[2];

  mx =  hy;
  my =  hx;
  mz = -hz;
  return true;
}
#endif /* ENABLE_MAG */

/* ══════════════════════════════════════════════════════════════════════════
 *  BMP280 barometer
 * ══════════════════════════════════════════════════════════════════════════
 *  Useful on a ground robot for one thing: detecting that you are on a slope
 *  or a ramp, where the accelerometer alone cannot separate tilt from
 *  acceleration. The fusion node treats it as a slow, low-weight hint.
 * ══════════════════════════════════════════════════════════════════════════ */

#if ENABLE_BARO
static bool bmpBegin() {
  uint8_t addr = BMP280_ADDR;
  if (!i2cPresent(addr)) {
    addr = 0x77;                                   /* some clones use 0x77   */
    if (!i2cPresent(addr)) return false;
  }
  uint8_t id = i2cRead8(addr, 0xD0);
  if (id != 0x58 && id != 0x60) return false;      /* 0x60 = BME280          */

  uint8_t c[24];
  if (!i2cRead(addr, 0x88, c, 24)) return false;
  dig_T1 = (uint16_t)(c[1]  << 8 | c[0]);
  dig_T2 = (int16_t) (c[3]  << 8 | c[2]);
  dig_T3 = (int16_t) (c[5]  << 8 | c[4]);
  dig_P1 = (uint16_t)(c[7]  << 8 | c[6]);
  dig_P2 = (int16_t) (c[9]  << 8 | c[8]);
  dig_P3 = (int16_t) (c[11] << 8 | c[10]);
  dig_P4 = (int16_t) (c[13] << 8 | c[12]);
  dig_P5 = (int16_t) (c[15] << 8 | c[14]);
  dig_P6 = (int16_t) (c[17] << 8 | c[16]);
  dig_P7 = (int16_t) (c[19] << 8 | c[18]);
  dig_P8 = (int16_t) (c[21] << 8 | c[20]);
  dig_P9 = (int16_t) (c[23] << 8 | c[22]);

  i2cWrite8(addr, 0xF5, 0x10);      /* t_sb 0.5 ms, IIR filter ×16          */
  i2cWrite8(addr, 0xF4, 0x57);      /* temp ×2, press ×16, normal mode      */
  delay(10);
  return true;
}

/* Bosch's fixed-point compensation, datasheet 3.11.3. Kept verbatim because
 * a "simplified" float version drifts by tens of pascals. */
static void bmpRead(float &press_pa, float &temp_c) {
  uint8_t b[6];
  if (!i2cRead(BMP280_ADDR, 0xF7, b, 6)) return;

  int32_t adc_P = ((int32_t)b[0] << 12) | ((int32_t)b[1] << 4) | (b[2] >> 4);
  int32_t adc_T = ((int32_t)b[3] << 12) | ((int32_t)b[4] << 4) | (b[5] >> 4);
  if (adc_T == 0x80000 || adc_P == 0x80000) return;   /* measurement skipped */

  int32_t var1 = ((((adc_T >> 3) - ((int32_t)dig_T1 << 1))) * ((int32_t)dig_T2)) >> 11;
  int32_t var2 = (((((adc_T >> 4) - ((int32_t)dig_T1)) *
                    ((adc_T >> 4) - ((int32_t)dig_T1))) >> 12) *
                  ((int32_t)dig_T3)) >> 14;
  t_fine = var1 + var2;
  temp_c = (float)((t_fine * 5 + 128) >> 8) * 0.01f;

  int64_t p1 = ((int64_t)t_fine) - 128000;
  int64_t p2 = p1 * p1 * (int64_t)dig_P6;
  p2 = p2 + ((p1 * (int64_t)dig_P5) << 17);
  p2 = p2 + (((int64_t)dig_P4) << 35);
  p1 = ((p1 * p1 * (int64_t)dig_P3) >> 8) + ((p1 * (int64_t)dig_P2) << 12);
  p1 = (((((int64_t)1) << 47) + p1)) * ((int64_t)dig_P1) >> 33;
  if (p1 == 0) return;                                 /* divide-by-zero     */
  int64_t p = 1048576 - adc_P;
  p = (((p << 31) - p2) * 3125) / p1;
  p1 = (((int64_t)dig_P9) * (p >> 13) * (p >> 13)) >> 25;
  p2 = (((int64_t)dig_P8) * p) >> 19;
  p = ((p + p1 + p2) >> 8) + (((int64_t)dig_P7) << 4);
  press_pa = (float)p / 256.0f;
}
#endif /* ENABLE_BARO */

/* ══════════════════════════════════════════════════════════════════════════
 *  DYP-A21 ultrasonic
 * ══════════════════════════════════════════════════════════════════════════
 *  Two guards matter more than the driver itself:
 *
 *  CROSSTALK. Two ultrasonics firing together will hear each other. They are
 *  therefore strictly alternated — front, gap, rear, gap — which halves the
 *  per-sensor rate and completely removes a class of phantom obstacle that is
 *  otherwise almost impossible to debug from a log.
 *
 *  SPIKES. Ultrasonics throw occasional wild readings off the floor, off a
 *  soft surface, or off nothing at all. A jump larger than US_MAX_JUMP_M is
 *  held back until a second sample agrees with it. A single phantom 0.1 m
 *  reading at the front of a moving car means an emergency stop, so "believe
 *  it the second time" is worth the one extra cycle of latency.
 * ══════════════════════════════════════════════════════════════════════════ */

static float s_us_pending[2] = {-1.0f, -1.0f};   /* provisional, unconfirmed */

static void acceptRange(RangeSample &s, uint8_t idx, float raw_m, uint32_t now) {
  if (raw_m < US_MIN_RANGE_M || raw_m > US_MAX_RANGE_M) {
    s.valid = false;
    s_us_pending[idx] = -1.0f;
    return;
  }
  if (s.valid && fabsf(raw_m - s.range_m) > US_MAX_JUMP_M) {
    /* Big jump: demand confirmation before acting on it. */
    if (s_us_pending[idx] >= 0.0f &&
        fabsf(raw_m - s_us_pending[idx]) <= US_MAX_JUMP_M) {
      s.range_m = raw_m;                 /* two samples agree → believe them */
      s.stamp_ms = now;
      s_us_pending[idx] = -1.0f;
    } else {
      s_us_pending[idx] = raw_m;         /* hold, keep the previous value    */
    }
    return;
  }
  s.range_m = raw_m;
  s.valid = true;
  s.stamp_ms = now;
  s_us_pending[idx] = -1.0f;
}

#if ULTRASONIC_MODE == US_MODE_I2C
static bool usTrigger(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(US_I2C_TRIGGER_CMD);
  return Wire.endTransmission() == 0;
}

static bool usReadMm(uint8_t addr, uint16_t &mm) {
  if (Wire.requestFrom(addr, (uint8_t)2) != 2) return false;
  uint8_t h = Wire.read();
  uint8_t l = Wire.read();
  mm = (uint16_t)((h << 8) | l);
  return true;
}
#elif ULTRASONIC_MODE == US_MODE_UART
/* ---------------------------------------------------------------------------
 *  DYP-A21 UART framing.
 *
 *  Frame: 0xFF, dist_HI, dist_LO, checksum, checksum = (0xFF+HI+LO) & 0xFF,
 *  distance in millimetres, sent unprompted about every 100 ms.
 *
 *  The naive reader - "wait until 4 bytes are available, then read 4 bytes" -
 *  fails the first time it starts mid-frame, because from then on every read
 *  is offset by one or two bytes forever. It also fails when the sensor is
 *  hot-plugged. So this is a proper byte-at-a-time state machine that hunts
 *  for the 0xFF header and validates the checksum before believing anything.
 *
 *  It is also non-blocking and per-port: it consumes whatever has arrived and
 *  returns, keeping partial frames in the port's own state across calls. That
 *  matters because this runs inside the same 100 Hz loop as the drive task.
 *
 *  CROSSTALK. In auto-report mode both sensors fire on their own internal
 *  timers and we cannot interleave them the way the I2C path does. Two
 *  transducers pinging at similar rates will occasionally hear each other,
 *  producing one wrong reading. That is what US_MAX_JUMP_M is for: a large
 *  jump has to repeat before it is believed. It is handled in acceptRange(),
 *  not here.
 * ------------------------------------------------------------------------- */
struct UsUartState {
  uint8_t  buf[4];
  uint8_t  n;
  uint32_t last_ok_ms;
};

static bool usReadUart(Stream &port, UsUartState &st, uint16_t &mm,
                       uint32_t now_ms, uint8_t idx) {
  bool got = false;
  /* Bound the work per call. A sensor that has gone mad and is streaming
   * continuously must not be able to hold this loop indefinitely. */
  uint8_t budget = 32;
  while (port.available() > 0 && budget--) {
    uint8_t b = (uint8_t)port.read();
    s_us_rx[idx]++;

    if (st.n == 0) {
      if (b != 0xFF) continue;          /* hunt for the header */
      st.buf[st.n++] = b;
      continue;
    }
    st.buf[st.n++] = b;
    if (st.n < 4) continue;

    uint8_t sum = (uint8_t)(st.buf[0] + st.buf[1] + st.buf[2]);
    if (sum == st.buf[3]) {
      mm = (uint16_t)((st.buf[1] << 8) | st.buf[2]);
      st.last_ok_ms = now_ms;
      s_us_frm[idx]++;
      got = true;
      st.n = 0;
    } else {
      /* Bad checksum. The 0xFF we locked onto was probably a data byte, not a
       * header. Do not throw the whole frame away - one of the bytes we have
       * may be a real header, so slide down and re-hunt from there. */
      s_us_bad[idx]++;
      uint8_t k = 1;
      while (k < 4 && st.buf[k] != 0xFF) k++;
      st.n = 0;
      for (uint8_t i = k; i < 4; i++) st.buf[st.n++] = st.buf[i];
      /* Keeps the candidate header AND the bytes after it, so resync
       * costs at most the corrupted frame. */
    }
  }
  return got;
}
#else /* US_MODE_TRIG_ECHO */
static bool usPing(uint8_t trig, uint8_t echo, uint16_t &mm) {
  digitalWrite(trig, LOW);  delayMicroseconds(4);
  digitalWrite(trig, HIGH); delayMicroseconds(12);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, US_ECHO_TIMEOUT_US);
  if (us == 0) return false;
  mm = (uint16_t)((float)us * US_SPEED_OF_SOUND * 0.5f * 1000.0f);
  return true;
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ══════════════════════════════════════════════════════════════════════════ */

bool begin() {
#if ENABLE_IMU
  s_imu_ok = mpuBegin();
#endif
#if ENABLE_MAG
  s_mag_ok = s_imu_ok ? akBegin() : false;
#endif
#if ENABLE_BARO
  s_baro_ok = bmpBegin();
#endif

#if ENABLE_ULTRASONIC
  #if ULTRASONIC_MODE == US_MODE_I2C
    s_us_front_ok = i2cPresent(US_FRONT_I2C_ADDR);
    s_us_rear_ok  = i2cPresent(US_REAR_I2C_ADDR);
  #elif ULTRASONIC_MODE == US_MODE_UART
    #if BOARD_IS_ESP32
      /* Probe the four sonar wires BEFORE the UARTs take them (fw 3.1.2). */
      s_us_idle[0] = usProbeIdle(PIN_US_FRONT_RX);
      s_us_idle[1] = usProbeIdle(PIN_US_FRONT_TX);
      s_us_idle[2] = usProbeIdle(PIN_US_REAR_RX);
      s_us_idle[3] = usProbeIdle(PIN_US_REAR_TX);
      /* Any GPIO can carry a UART on the ESP32, so the pin numbers in
       * Config.h are a free choice. Both sonars get their own hardware UART
       * and therefore their own FIFO - no listen()-switching, no lost frames. */
      #if US_FRONT_ENABLED
        US_PORT_FRONT.begin(US_UART_BAUD);   /* Serial0 = header D0/D1 */
      #endif
      #if US_REAR_ENABLED
        US_PORT_REAR.begin(US_UART_BAUD, SERIAL_8N1,
                           PIN_US_REAR_RX, PIN_US_REAR_TX);
      #endif
      /* Say exactly which pins were claimed, so a pin-numbering mix-up
       * (Arduino D0 vs GPIO0) shows up in the boot log, not as -1 forever. */
      LINK.print(F("{\"status\":\"sonar uart 9600 mode="));
      LINK.print(US_UART_AUTO_REPORT ? F("auto") : F("query"));
      LINK.print(F(" front="));
      LINK.print(US_FRONT_ENABLED ? F("on") : F("off"));
      LINK.print(F(" rx=D")); LINK.print(PIN_US_FRONT_RX);
      LINK.print(F(" tx=D")); LINK.print(PIN_US_FRONT_TX);
      LINK.print(F(" rear="));
      LINK.print(US_REAR_ENABLED ? F("on") : F("off"));
      LINK.print(F(" rx=D")); LINK.print(PIN_US_REAR_RX);
      LINK.print(F(" tx=D")); LINK.print(PIN_US_REAR_TX);
      #if defined(BOARD_HAS_PIN_REMAP)
        LINK.print(F(" | gpio front rx=")); LINK.print(digitalPinToGPIONumber(PIN_US_FRONT_RX));
        LINK.print(F(" tx="));             LINK.print(digitalPinToGPIONumber(PIN_US_FRONT_TX));
        LINK.print(F(" rear rx="));        LINK.print(digitalPinToGPIONumber(PIN_US_REAR_RX));
        LINK.print(F(" tx="));             LINK.print(digitalPinToGPIONumber(PIN_US_REAR_TX));
      #else
        LINK.print(F(" | GPIO numbering: pins above are raw GPIOs"));
      #endif
      LINK.println(F("\"}"));
    #else
      usFront.begin(US_UART_BAUD);
      usRear.begin(US_UART_BAUD);
    #endif
    /* UART: presence is PROVEN by a valid frame (see updateUltrasonic), not
     * assumed. This used to be hard-coded to 1, so the heartbeat said
     * "us_f_ok":1 even with nothing connected - and "us_r_ok":1 for a sonar
     * that had been physically removed. */
    s_us_front_ok = false;
    s_us_rear_ok  = false;
  #else
    pinMode(PIN_US_FRONT_TRIG, OUTPUT); pinMode(PIN_US_FRONT_ECHO, INPUT);
    pinMode(PIN_US_REAR_TRIG,  OUTPUT); pinMode(PIN_US_REAR_ECHO,  INPUT);
    s_us_front_ok = true;
    s_us_rear_ok  = true;
  #endif
#endif

  beginGyroCalibration();
  return s_imu_ok;
}

void rescan() {
  s_imu_ok = false; s_mag_ok = false; s_baro_ok = false;
  begin();
}

void updateImu(uint32_t now_ms) {
  if (!s_imu_ok) {
#if ENABLE_IMU
    /* Down: try to bring it back once a second instead of staying silent
     * until the next reboot. mpuBegin() returns at once if nothing answers,
     * so a missing IMU costs almost nothing; a successful re-init costs its
     * ~130 ms reset once (the PCA9685 holds servo/ESC pulses meanwhile). */
    if (now_ms >= s_imu_retry_ms) {
      s_imu_retry_ms = now_ms + 1000;
      i2cRecoverBus();        /* fw 3.4.0: unstick SDA before re-init */
      if (mpuBegin()) {
        s_imu_ok = true; s_imu_fail = 0; s_imu_recoveries++;
  #if ENABLE_MAG
        s_mag_ok = akBegin();
  #endif
      }
    }
#endif
    return;
  }
  const uint32_t period = 1000UL / IMU_PUBLISH_HZ;
  if (now_ms < s_imu_next) return;
  s_imu_next = now_ms + period;

  float ax, ay, az, gx, gy, gz, t;
  if (!mpuRead(ax, ay, az, gx, gy, gz, t)) {
    /* One failed read is a glitch, not a dead IMU (this GY-91 has shown slow
     * reads for a while). Skip the sample; give up only after 5 in a row, and
     * even then the top of this function keeps trying to re-initialise. */
    if (++s_imu_fail >= 5) { s_imu_ok = false; s_imu_retry_ms = now_ms + 200; }
    return;
  }
  s_imu_fail = 0;
  s_imu.ax = ax; s_imu.ay = ay; s_imu.az = az;
  s_imu.gx = gx - s_gyro_bias[0];
  s_imu.gy = gy - s_gyro_bias[1];
  s_imu.gz = gz - s_gyro_bias[2];
  s_imu.temp_c = t;

#if ENABLE_MAG
  if (s_mag_ok) {
    float mx, my, mz;
    if (akRead(mx, my, mz)) {
      s_imu.mx = mx; s_imu.my = my; s_imu.mz = mz;
      s_imu.mag_valid = true;
    }
  }
#endif

  /* Feed the bias estimator with the UNCORRECTED sample. */
  if (s_gyro_cal_run) {
    float amag = sqrtf(ax * ax + ay * ay + az * az);
    if (fabsf(amag - 9.80665f) < GYRO_STILL_ACCEL_TOL) {
      s_gyro_cal_sum[0] += gx;
      s_gyro_cal_sum[1] += gy;
      s_gyro_cal_sum[2] += gz;
      if (++s_gyro_cal_n >= GYRO_BIAS_SAMPLES) {
        for (uint8_t i = 0; i < 3; i++)
          s_gyro_bias[i] = s_gyro_cal_sum[i] / (float)s_gyro_cal_n;
        s_gyro_cal_done = true;
        s_gyro_cal_run  = false;
      }
    } else {
      /* Moved mid-calibration: throw the partial average away rather than
       * baking a rotation into the bias. */
      s_gyro_cal_n = 0;
      s_gyro_cal_sum[0] = s_gyro_cal_sum[1] = s_gyro_cal_sum[2] = 0.0f;
    }
  }
}

void updateBaro(uint32_t now_ms) {
#if ENABLE_BARO
  if (!s_baro_ok) return;
  const uint32_t period = 1000UL / BARO_PUBLISH_HZ;
  if (now_ms < s_baro_next) return;
  s_baro_next = now_ms + period;
  bmpRead(s_press_pa, s_baro_t);
#else
  (void)now_ms;
#endif
}

void updateUltrasonic(uint32_t now_ms) {
#if !ENABLE_ULTRASONIC
  (void)now_ms;
  return;
#else
  /* The UART path drains its FIFOs on EVERY visit; only the modes where we
   * actively trigger a ping are rate-limited here. */
  #if ULTRASONIC_MODE != US_MODE_UART || US_UART_AUTO_REPORT == 0
  if (now_ms < s_us_next) return;
  #endif

  #if ULTRASONIC_MODE == US_MODE_I2C
    /* Two-phase, non-blocking: trigger on one visit, read on the next, so we
     * never sit in delay() for the 70 ms time-of-flight. */
    static bool triggered = false;
    uint8_t addr = (s_us_turn == 0) ? US_FRONT_I2C_ADDR : US_REAR_I2C_ADDR;
    bool present = (s_us_turn == 0) ? s_us_front_ok : s_us_rear_ok;
    if (!present) { s_us_turn ^= 1; s_us_next = now_ms + US_PING_GAP_MS; return; }

    if (!triggered) {
      if (usTrigger(addr)) triggered = true;
      s_us_next = now_ms + US_I2C_READ_DELAY_MS;
      return;
    }
    triggered = false;
    uint16_t mm = 0;
    if (usReadMm(addr, mm)) {
      acceptRange(s_us_turn == 0 ? s_front : s_rear, s_us_turn,
                  (float)mm * 0.001f, now_ms);
    }
    s_us_turn ^= 1;
    s_us_next = now_ms + US_PING_GAP_MS;

  #elif ULTRASONIC_MODE == US_MODE_UART
    /* No trigger, no waiting: the sensors free-run and we drain whatever has
     * arrived. Called every visit rather than on the publish timer, because
     * a 4-byte frame left sitting in a 64-byte hardware FIFO while we sleep
     * is how you lose readings under load. */
    {
      static UsUartState st_f = {{0, 0, 0, 0}, 0, 0};
      static UsUartState st_r = {{0, 0, 0, 0}, 0, 0};
      uint16_t mm = 0;

    #if US_UART_AUTO_REPORT == 0
      /* Query mode: poke each sensor in turn, alternating, so the two are
       * never in flight together and cannot hear each other. */
      if (now_ms >= s_us_next) {
        /* Never poke a port that was not begun (a removed sonar). */
        if (s_us_turn == 0) { if (US_FRONT_ENABLED) US_PORT_FRONT.write((uint8_t)0x55); }
        else                { if (US_REAR_ENABLED)  US_PORT_REAR.write((uint8_t)0x55); }
        s_us_turn = (US_FRONT_ENABLED && US_REAR_ENABLED) ? (s_us_turn ^ 1)
                                                          : (US_FRONT_ENABLED ? 0 : 1);
        s_us_next = now_ms + US_PING_GAP_MS;
      }
    #endif

    #if BOARD_IS_ESP32
      #if US_FRONT_ENABLED
        if (usReadUart(US_PORT_FRONT, st_f, mm, now_ms, 0))
          acceptRange(s_front, 0, mm * 0.001f, now_ms);
      #else
        s_front.valid = false;      /* not fitted: say so, do not guess */
      #endif
      #if US_REAR_ENABLED
        if (usReadUart(US_PORT_REAR, st_r, mm, now_ms, 1))
          acceptRange(s_rear, 1, mm * 0.001f, now_ms);
      #else
        s_rear.valid = false;
      #endif
    #else
      /* SoftwareSerial can only listen to one port at a time on AVR, so we
       * alternate. This halves the effective rate per sensor and is the main
       * reason the ESP32 is the supported target for UART sonars. */
      usFront.listen();
      if (usReadUart(usFront, st_f, mm, now_ms, 0))
        acceptRange(s_front, 0, mm * 0.001f, now_ms);
      usRear.listen();
      if (usReadUart(usRear, st_r, mm, now_ms, 1))
        acceptRange(s_rear, 1, mm * 0.001f, now_ms);
    #endif

      /* Staleness. A sonar that has stopped talking must stop being believed.
       * Holding its last value is the dangerous failure: a stale 0.4 m makes
       * the planner brake forever, and a stale 4 m lets it drive into
       * something the sensor stopped reporting ten seconds ago.
       *
       * Marking it invalid is what the rest of the stack already understands -
       * publishRanges() sends -1, the bridge turns that into +inf per REP-117,
       * and scan_fusion treats it as "no information" rather than "clear". */
      if (st_f.last_ok_ms != 0 &&
          (now_ms - st_f.last_ok_ms) > US_UART_TIMEOUT_MS) {
        s_front.valid = false;
      }
      if (st_r.last_ok_ms != 0 &&
          (now_ms - st_r.last_ok_ms) > US_UART_TIMEOUT_MS) {
        s_rear.valid = false;
      }
      /* Presence = a checksummed frame within the timeout window. */
      s_us_front_ok = US_FRONT_ENABLED && st_f.last_ok_ms != 0 &&
                      (now_ms - st_f.last_ok_ms) <= US_UART_TIMEOUT_MS;
      s_us_rear_ok  = US_REAR_ENABLED && st_r.last_ok_ms != 0 &&
                      (now_ms - st_r.last_ok_ms) <= US_UART_TIMEOUT_MS;
    }

  #else /* TRIG/ECHO */
    uint16_t mm = 0;
    if (s_us_turn == 0) {
      if (usPing(PIN_US_FRONT_TRIG, PIN_US_FRONT_ECHO, mm))
        acceptRange(s_front, 0, (float)mm * 0.001f, now_ms);
    } else {
      if (usPing(PIN_US_REAR_TRIG, PIN_US_REAR_ECHO, mm))
        acceptRange(s_rear, 1, (float)mm * 0.001f, now_ms);
    }
    s_us_turn ^= 1;
    s_us_next = now_ms + US_PING_GAP_MS;
  #endif
#endif
}

/* ── Gyro calibration ──────────────────────────────────────────────────── */

void beginGyroCalibration() {
  s_gyro_cal_run  = true;
  s_gyro_cal_done = false;
  s_gyro_cal_n    = 0;
  s_gyro_cal_sum[0] = s_gyro_cal_sum[1] = s_gyro_cal_sum[2] = 0.0f;
}

void updateGyroCalibration(bool is_commanded_still) {
  /* Re-arm a calibration if the car has been parked and the existing bias is
   * old. Gyro bias walks with temperature, and an RC car warms up ~15 °C in
   * its first few minutes of running. */
  if (!is_commanded_still) return;
  if (!s_gyro_cal_done && !s_gyro_cal_run) beginGyroCalibration();
}

bool gyroCalibrated() { return s_gyro_cal_done; }

void gyroBias(float &bx, float &by, float &bz) {
  bx = s_gyro_bias[0]; by = s_gyro_bias[1]; bz = s_gyro_bias[2];
}

/* ── Access ────────────────────────────────────────────────────────────── */

const ImuSample&   imu()        { return s_imu; }
const RangeSample& rangeFront() { return s_front; }
const RangeSample& rangeRear()  { return s_rear; }
float pressurePa()              { return s_press_pa; }
float baroTempC()               { return s_baro_t; }

bool imuPresent()              { return s_imu_ok; }
bool magPresent()              { return s_mag_ok; }
bool baroPresent()             { return s_baro_ok; }
bool ultrasonicFrontPresent()  { return s_us_front_ok; }
bool ultrasonicRearPresent()   { return s_us_rear_ok; }
uint8_t  ultrasonicIdle(uint8_t i) { return i < 4 ? s_us_idle[i] : 2; }
uint32_t imuRecoveries()           { return s_imu_recoveries; }
void ultrasonicStats(uint8_t idx, uint32_t &rx, uint32_t &frm, uint32_t &bad) {
  idx = idx ? 1 : 0;
  rx = s_us_rx[idx]; frm = s_us_frm[idx]; bad = s_us_bad[idx];
}

}  /* namespace Sensors */
