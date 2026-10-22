/* =============================================================================
 *  Battery.cpp — implementation
 * ============================================================================= */

#include "Battery.h"

namespace Battery {

/* ══════════════════════════════════════════════════════════════════════════
 *  MAX7219 — bit-banged 3-wire driver
 * ══════════════════════════════════════════════════════════════════════════
 *  Protocol: pull CS low, shift 16 bits MSB-first (address byte then data
 *  byte) on the rising edge of CLK, then raise CS to latch. That is the whole
 *  chip. It is deliberately not on the hardware SPI peripheral: the display
 *  updates four times a second, bit-banging costs nothing, and it leaves the
 *  SPI pins free.
 *
 *  Segment bit map in no-decode mode:
 *       bit7 = DP, bit6 = A, bit5 = B, bit4 = C,
 *       bit3 = D,  bit2 = E, bit1 = F, bit0 = G
 *
 *          AAA
 *         F   B
 *         F   B
 *          GGG
 *         E   C
 *         E   C
 *          DDD  DP
 * ══════════════════════════════════════════════════════════════════════════ */

#if ENABLE_DISPLAY

/* Segment font. Indexed through glyphFor() below. */
static const uint8_t SEG_DIGIT[10] = {
  0x7E, /* 0 */ 0x30, /* 1 */ 0x6D, /* 2 */ 0x79, /* 3 */ 0x33, /* 4 */
  0x5B, /* 5 */ 0x5F, /* 6 */ 0x70, /* 7 */ 0x7F, /* 8 */ 0x7B  /* 9 */
};

static const uint8_t SEG_BLANK = 0x00;
static const uint8_t SEG_DP    = 0x80;

/* Letters we can actually form on seven segments. Anything not listed here
 * renders blank rather than as a random pattern, which makes typos in status
 * strings obvious instead of mysterious. */
static uint8_t glyphFor(char c) {
  if (c >= '0' && c <= '9') return SEG_DIGIT[c - '0'];
  switch (c) {
    case ' ': return SEG_BLANK;
    case '-': return 0x01;
    case 'A': case 'a': return 0x77;
    case 'b': case 'B': return 0x1F;
    case 'C':           return 0x4E;
    case 'c':           return 0x0D;
    case 'd': case 'D': return 0x3D;
    case 'E': case 'e': return 0x4F;
    case 'F': case 'f': return 0x47;
    case 'H': case 'h': return 0x37;
    case 'I': case 'i': return 0x30;
    case 'L': case 'l': return 0x0E;
    case 'n':           return 0x15;
    case 'O':           return 0x7E;
    case 'o':           return 0x1D;
    case 'P': case 'p': return 0x67;
    case 'r': case 'R': return 0x05;
    case 'S': case 's': return 0x5B;
    case 't': case 'T': return 0x0F;
    case 'U':           return 0x3E;
    case 'u':           return 0x1C;
    case 'y': case 'Y': return 0x3B;
    default:            return SEG_BLANK;
  }
}

static void maxSend(uint8_t reg, uint8_t data) {
  digitalWrite(PIN_MAX7219_CS, LOW);
  for (int8_t i = 7; i >= 0; i--) {                 /* address byte           */
    digitalWrite(PIN_MAX7219_CLK, LOW);
    digitalWrite(PIN_MAX7219_DIN, (reg >> i) & 0x01);
    digitalWrite(PIN_MAX7219_CLK, HIGH);
  }
  for (int8_t i = 7; i >= 0; i--) {                 /* data byte              */
    digitalWrite(PIN_MAX7219_CLK, LOW);
    digitalWrite(PIN_MAX7219_DIN, (data >> i) & 0x01);
    digitalWrite(PIN_MAX7219_CLK, HIGH);
  }
  digitalWrite(PIN_MAX7219_CS, HIGH);               /* latch                  */
}

/* Write one digit position, 0 = leftmost as the user sees it. The MAX7219's
 * DIG0..DIG3 may be wired either way round; DISPLAY_DIGITS_REVERSED flips it
 * in software so you never have to unsolder anything. */
static void maxSetDigit(uint8_t pos, uint8_t segments) {
  if (pos >= MAX7219_NUM_DIGITS) return;
#if DISPLAY_DIGITS_REVERSED
  uint8_t hw = (uint8_t)(MAX7219_NUM_DIGITS - 1 - pos);
#else
  uint8_t hw = pos;
#endif
  maxSend((uint8_t)(MAX7219_REG_DIGIT0 + hw), segments);
}

static void maxInit() {
  pinMode(PIN_MAX7219_DIN, OUTPUT);
  pinMode(PIN_MAX7219_CS,  OUTPUT);
  pinMode(PIN_MAX7219_CLK, OUTPUT);
  digitalWrite(PIN_MAX7219_CS, HIGH);

  maxSend(MAX7219_REG_DISPLAYTEST, 0x00);                  /* test off        */
  maxSend(MAX7219_REG_SCANLIMIT,   MAX7219_NUM_DIGITS - 1);/* scan 4 digits   */
  maxSend(MAX7219_REG_DECODEMODE,  0x00);                  /* raw segments    */
  maxSend(MAX7219_REG_INTENSITY,   MAX7219_INTENSITY);
  maxSend(MAX7219_REG_SHUTDOWN,    0x01);                  /* normal operation*/
  for (uint8_t i = 0; i < MAX7219_NUM_DIGITS; i++) maxSetDigit(i, SEG_BLANK);
}

#endif /* ENABLE_DISPLAY */

/* ══════════════════════════════════════════════════════════════════════════
 *  LiPo discharge curve
 * ══════════════════════════════════════════════════════════════════════════
 *  Per-cell resting voltage → state of charge, for a lightly loaded LiPo.
 *  Descending voltage order; linear interpolation between neighbours.
 *
 *  Read the shape: from 100 % down to 20 % the cell falls only 0.49 V, then
 *  loses the last 20 % in 0.19 V. That knee is exactly why a linear gauge is
 *  useless and why "it showed 50 % and then died" is such a common complaint.
 * ══════════════════════════════════════════════════════════════════════════ */
struct CurvePoint { float v; float pct; };

static const CurvePoint LIPO_CURVE[] = {
  {4.20f, 100.0f}, {4.15f,  95.0f}, {4.11f,  90.0f}, {4.08f,  85.0f},
  {4.02f,  80.0f}, {3.98f,  75.0f}, {3.95f,  70.0f}, {3.91f,  65.0f},
  {3.87f,  60.0f}, {3.85f,  55.0f}, {3.84f,  50.0f}, {3.82f,  45.0f},
  {3.80f,  40.0f}, {3.79f,  35.0f}, {3.77f,  30.0f}, {3.75f,  25.0f},
  {3.73f,  20.0f}, {3.71f,  15.0f}, {3.69f,  10.0f}, {3.61f,   5.0f},
  {3.50f,   0.0f}
};
static const uint8_t CURVE_N = sizeof(LIPO_CURVE) / sizeof(LIPO_CURVE[0]);

static float curveLookup(float cell_v) {
  if (cell_v >= LIPO_CURVE[0].v)           return 100.0f;
  if (cell_v <= LIPO_CURVE[CURVE_N - 1].v) return 0.0f;
  for (uint8_t i = 0; i < CURVE_N - 1; i++) {
    float hi_v = LIPO_CURVE[i].v, lo_v = LIPO_CURVE[i + 1].v;
    if (cell_v <= hi_v && cell_v >= lo_v) {
      float t = (cell_v - lo_v) / (hi_v - lo_v);       /* 0 at lo, 1 at hi    */
      return LIPO_CURVE[i + 1].pct
           + t * (LIPO_CURVE[i].pct - LIPO_CURVE[i + 1].pct);
    }
  }
  return 0.0f;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Module state
 * ══════════════════════════════════════════════════════════════════════════ */
static float    s_v_filt     = 0.0f;   /* EMA of the measured terminal volts */
static float    s_v_rest     = 0.0f;   /* sag-compensated estimate           */
static float    s_percent    = 0.0f;
static uint32_t s_load_ms    = 0;      /* fw 3.5.3: last time the motor was loaded */
static bool     s_valid      = false;
static uint16_t s_warmup     = 0;
static uint32_t s_last_read  = 0;
static uint32_t s_last_disp  = 0;
static uint32_t s_cycle_t0   = 0;
static bool     s_show_volts = false;
static bool     s_blink_on   = true;
static uint32_t s_blink_t0   = 0;

/* Divider ratio: V_pack = V_adc * (R1 + R2) / R2 */
static const float DIVIDER_RATIO =
    (DIVIDER_R1_OHMS + DIVIDER_R2_OHMS) / DIVIDER_R2_OHMS;

/* ── One raw voltage sample at the ADC pin ────────────────────────────── */
static float readPinVolts() {
#if BOARD_IS_ESP32
  /* analogReadMilliVolts() applies the chip's factory ADC calibration, which
   * matters a lot on the ESP32-S3: the raw ADC is noticeably non-linear near
   * both rails and a plain analogRead() can be off by 5 % or more. */
  return (float)analogReadMilliVolts(PIN_BATTERY_SENSE) * 0.001f;
#else
  return ((float)analogRead(PIN_BATTERY_SENSE) / ADC_MAX_COUNTS) * ADC_VREF;
#endif
}

/* Median of BATTERY_SAMPLES readings. The median, not the mean: a single
 * commutation spike from the brushed motor moves a mean but cannot move a
 * median at all. Insertion sort because N is 15 and the array is tiny. */
/* Blocking version. BATTERY_SAMPLES reads back to back with a settle delay
 * between each - about 2.3 ms at 15 samples. Fine at boot, NOT fine in the
 * control loop: 2.3 ms is a quarter of the 10 ms drive period, arriving as a
 * jitter spike every BATTERY_READ_PERIOD_MS. Used only by begin(). */
static float medianPinVoltsBlocking() {
  float buf[BATTERY_SAMPLES];
  for (uint8_t i = 0; i < BATTERY_SAMPLES; i++) {
    buf[i] = readPinVolts();
    delayMicroseconds(150);
  }
  for (uint8_t i = 1; i < BATTERY_SAMPLES; i++) {
    float key = buf[i];
    int8_t j = (int8_t)i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  return buf[BATTERY_SAMPLES / 2];
}

/* ---------------------------------------------------------------------------
 *  Non-blocking sampler used at runtime.
 *
 *  One ADC read per loop iteration into a ring, median taken when the read
 *  period elapses. Consecutive samples are already milliseconds apart because
 *  the loop runs at ~100 Hz+, so the 150 us settle delay is not merely
 *  unnecessary - it is redundant by three orders of magnitude.
 *
 *  Same median-of-15 rejection of ESC switching spikes, zero added latency.
 * ------------------------------------------------------------------------- */
static float   s_ring[BATTERY_SAMPLES];
static uint8_t s_ring_n   = 0;      /* how many slots are populated */
static uint8_t s_ring_i   = 0;      /* next slot to write           */

static void sampleTick() {
  s_ring[s_ring_i] = readPinVolts();
  s_ring_i = (uint8_t)((s_ring_i + 1) % BATTERY_SAMPLES);
  if (s_ring_n < BATTERY_SAMPLES) s_ring_n++;
}

static bool ringMedian(float &out) {
  if (s_ring_n < 3) return false;           /* not enough to reject anything */
  float buf[BATTERY_SAMPLES];
  for (uint8_t i = 0; i < s_ring_n; i++) buf[i] = s_ring[i];
  for (uint8_t i = 1; i < s_ring_n; i++) {
    float key = buf[i];
    int8_t j = (int8_t)i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  out = buf[s_ring_n / 2];
  return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ══════════════════════════════════════════════════════════════════════════ */

bool begin() {
#if BOARD_IS_ESP32
  analogReadResolution(12);
#endif
  pinMode(PIN_BATTERY_SENSE, INPUT);

#if ENABLE_DISPLAY
  maxInit();
  /* Self-test: every segment, so a dead digit or a cold joint shows up in the
   * first second rather than as a wrong number three weeks later. */
  for (uint8_t i = 0; i < MAX7219_NUM_DIGITS; i++) maxSetDigit(i, 0xFF);
  delay(400);
  showText("bAt");
  delay(500);
#endif

  /* Prime the filter so the first displayed value is already sensible
   * instead of ramping up from zero. */
  float pin_v = medianPinVoltsBlocking();
  s_v_filt = (pin_v * DIVIDER_RATIO) * BATTERY_CAL_GAIN + BATTERY_CAL_OFFSET;
  s_v_rest = s_v_filt;
  s_percent = curveLookup(s_v_rest / (float)BATTERY_CELLS);
  s_cycle_t0 = millis();
  s_blink_t0 = s_cycle_t0;

#if ENABLE_DISPLAY
  return true;
#else
  return false;
#endif
}

void update(uint32_t now_ms, float throttle_frac) {
  /* One cheap ADC read per visit - no delay, no loop. */
  sampleTick();

  if (now_ms - s_last_read < BATTERY_READ_PERIOD_MS) return;
  s_last_read = now_ms;

  float pin_v;
  if (!ringMedian(pin_v)) return;
  float pack_v = (pin_v * DIVIDER_RATIO) * BATTERY_CAL_GAIN + BATTERY_CAL_OFFSET;

  /* Reject nonsense outright. An open divider reads ~0, a shorted one reads
   * the rail; neither should be fed into the filter and then displayed as a
   * confident percentage. */
  if (pack_v < 1.0f || pack_v > (float)BATTERY_CELLS * 4.6f) return;

  /* fw 3.5.3: compensate EACH sample for the load it was taken under, THEN
   * filter. Filtering first (as before) kept a sagged voltage in the ~4 s
   * EMA after the car stopped, while the load was already zero: the gauge
   * then "fell freely" on stale sag - 15-19 % shown at a resting 45 %. */
  float sag = BATTERY_SAG_V_AT_FULL * fabsf(throttle_frac);
  if (fabsf(throttle_frac) > 0.02f) s_load_ms = now_ms;
  s_v_filt += ((pack_v + sag) - s_v_filt) * BATTERY_EMA_ALPHA;

  float v_rest_raw = s_v_filt;                 /* already an open-circuit estimate */
  if (v_rest_raw > (float)BATTERY_CELLS * CELL_FULL_V)
    v_rest_raw = (float)BATTERY_CELLS * CELL_FULL_V;
  s_v_rest = v_rest_raw;

  float target = curveLookup(s_v_rest / (float)BATTERY_CELLS);

  /* Asymmetric rate limit: a gauge is allowed to fall as fast as the pack
   * does, but may only climb slowly. Batteries do recover when you let off
   * the throttle, but a percentage that jumps 15 points upward reads as a
   * broken gauge even when it is arithmetically correct. */
  float dt = (float)BATTERY_READ_PERIOD_MS * 0.001f;
  if (target > s_percent) {
    float max_rise = BATTERY_RISE_LIMIT_PPS * dt;
    s_percent += (target - s_percent > max_rise) ? max_rise : (target - s_percent);
  } else if (now_ms - s_load_ms < BATTERY_RECOVER_MS) {
    /* fw 3.5.3: under load, or while the voltage is still recovering, a
     * fall is mostly sag: a 6750 mAh pack cannot lose 30 % in a second.
     * At rest (true open-circuit voltage) it still falls freely. */
    float max_fall = BATTERY_FALL_LIMIT_PPS * dt;
    s_percent -= (s_percent - target > max_fall) ? max_fall : (s_percent - target);
  } else {
    s_percent = target;
  }

  if (!s_valid) {
    if (++s_warmup >= 8) s_valid = true;    /* ~1.6 s of agreeing samples     */
  }
}

void updateDisplay(uint32_t now_ms) {
#if !ENABLE_DISPLAY
  (void)now_ms;
  return;
#else
  if (now_ms - s_last_disp < DISPLAY_UPDATE_MS) return;
  s_last_disp = now_ms;

  if (!s_valid) { showText(" ---"); return; }

  /* Critical pack: flash "LO" and nothing else. At this point the only
   * number that matters is that you should stop driving. */
  if (isCritical()) {
    if (now_ms - s_blink_t0 >= DISPLAY_LOW_BLINK_MS) {
      s_blink_t0 = now_ms;
      s_blink_on = !s_blink_on;
    }
    showText(s_blink_on ? "LO  " : "    ");
    return;
  }

#if DISPLAY_SHOW_VOLTS
  uint32_t dwell = s_show_volts ? DISPLAY_VOLTS_MS : DISPLAY_PERCENT_MS;
  if (now_ms - s_cycle_t0 >= dwell) {
    s_cycle_t0 = now_ms;
    s_show_volts = !s_show_volts;
  }
#endif

  char buf[10];
#if DISPLAY_SHOW_VOLTS
  if (s_show_volts) {
    /* "16.8" — three digits plus a decimal point, which the font handles
     * without consuming a fourth digit position. */
    float v = s_v_filt;
    if (v < 0.0f)  v = 0.0f;
    if (v > 99.0f) v = 99.0f;          /* keeps the format at three digits   */
    int whole = (int)v;
    int frac  = (int)((v - (float)whole) * 10.0f + 0.5f);
    if (frac > 9) { frac = 0; whole += 1; }
    if (whole > 99) { whole = 99; frac = 9; }
    buf[0] = (char)('0' + (whole / 10) % 10);
    buf[1] = (char)('0' + whole % 10);
    buf[2] = '.';
    buf[3] = (char)('0' + frac % 10);
    buf[4] = '\0';
    showText(buf);
    return;
  }
#endif

  int pct = (int)(s_percent + 0.5f);
  if (pct < 0)   pct = 0;
  if (pct > 100) pct = 100;
  /* 100 fills all four digits as "100P" would not fit nicely; show "100 ".
   * Below that, right-align the number and mark it with a trailing 'P' so a
   * glance tells you percent from volts without waiting for the cycle. */
  if (pct >= 100) snprintf(buf, sizeof(buf), "100 ");
  else            snprintf(buf, sizeof(buf), "%3dP", pct);
  showText(buf);
#endif
}

void showText(const char *text) {
#if !ENABLE_DISPLAY
  (void)text;
#else
  uint8_t pos = 0;
  const char *p = text;
  while (*p && pos < MAX7219_NUM_DIGITS) {
    uint8_t seg = glyphFor(*p);
    p++;
    if (*p == '.') { seg |= SEG_DP; p++; }   /* point rides on the character  */
    maxSetDigit(pos++, seg);
  }
  while (pos < MAX7219_NUM_DIGITS) maxSetDigit(pos++, SEG_BLANK);
#endif
}

void showNumber(int value) {
  char buf[8];
  if (value >  9999) value =  9999;
  if (value <  -999) value =  -999;
  snprintf(buf, sizeof(buf), "%4d", value);
  showText(buf);
}

void blank() {
#if ENABLE_DISPLAY
  for (uint8_t i = 0; i < MAX7219_NUM_DIGITS; i++) maxSetDigit(i, SEG_BLANK);
#endif
}

void setIntensity(uint8_t level) {
#if ENABLE_DISPLAY
  if (level > 15) level = 15;
  maxSend(MAX7219_REG_INTENSITY, level);
#else
  (void)level;
#endif
}

float packVoltage()    { return s_v_filt; }
float restingVoltage() { return s_v_rest; }
float cellVoltage()    { return s_v_rest / (float)BATTERY_CELLS; }
float percent()        { return s_percent; }
float remainingMah()   { return s_percent * 0.01f * BATTERY_CAPACITY_MAH; }
bool  isValid()        { return s_valid; }
bool  isLow()          { return s_valid && cellVoltage() <= CELL_CRITICAL_V; }
bool  isCritical()     { return s_valid && cellVoltage() <= CELL_EMPTY_V; }

}  /* namespace Battery */
