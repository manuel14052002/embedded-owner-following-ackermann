/* =============================================================================
 *  Drive.cpp — implementation
 * ============================================================================= */

#include <Wire.h>
#include "Drive.h"

namespace Drive {

/* ── Module state ───────────────────────────────────────────────────────── */
static bool     s_pca_ok      = false;
static bool     s_armed       = false;
static bool     s_failsafe    = true;
static uint32_t s_arm_start   = 0;
static uint32_t s_last_cmd_ms = 0;
static uint32_t s_frames      = 0;
static uint32_t s_last_update = 0;

static float    s_req_v = 0.0f;      /* requested, from the Pi                */
static float    s_req_s = 0.0f;
static float    s_app_v = 0.0f;      /* applied, after limiting               */
static float    s_app_s = 0.0f;

static uint16_t s_esc_us   = ESC_NEUTRAL_US;
static uint32_t s_wd_last  = 0;      /* last watchdog re-arm */
static bool     s_wd_tripped = false;
static uint16_t s_servo_us = SERVO_CENTER_US;

/* Reverse handshake state machine. Brushed RC ESCs interpret the FIRST
 * backwards pulse as "brake" and only the second, after a return to neutral,
 * as "reverse". Skipping this is why an escape manoeuvre sometimes just
 * stopped the car dead instead of backing it out. */
enum RevState : uint8_t { REV_IDLE, REV_BRAKING, REV_NEUTRAL_GAP, REV_RUNNING };
static RevState s_rev       = REV_IDLE;
static uint32_t s_rev_timer = 0;
static bool     s_was_fwd   = false;

/* ══════════════════════════════════════════════════════════════════════════
 *  PCA9685 raw driver
 * ══════════════════════════════════════════════════════════════════════════ */

static bool pcaWrite8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static uint8_t pcaRead8(uint8_t reg) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom((uint8_t)PCA9685_ADDR, (uint8_t)1) != 1) return 0xFF;
  return Wire.read();
}

/* Set the PWM frame rate. The prescaler is an integer, so the achievable
 * frequency is quantised: at 50 Hz the real output is 50.0 Hz within ~0.4 %,
 * which no RC device cares about. */
static bool pcaSetFreq(float hz) {
  float prescale_f = (PCA9685_OSC_HZ / (4096.0f * hz)) - 1.0f;
  uint8_t prescale = (uint8_t)(prescale_f + 0.5f);
  if (prescale < 3) prescale = 3;         /* datasheet minimum                */

  uint8_t old_mode = pcaRead8(PCA_MODE1);
  if (old_mode == 0xFF) return false;

  /* PRESCALE is only writable in sleep. */
  if (!pcaWrite8(PCA_MODE1, (old_mode & 0x7F) | PCA_MODE1_SLEEP)) return false;
  if (!pcaWrite8(PCA_PRESCALE, prescale)) return false;
  if (!pcaWrite8(PCA_MODE1, old_mode)) return false;
  delay(1);                                /* oscillator restart, datasheet 7.3.1.1 */
  return pcaWrite8(PCA_MODE1, old_mode | PCA_MODE1_RESTART | PCA_MODE1_AI);
}

static bool pcaSetPWM(uint8_t ch, uint16_t on, uint16_t off) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(PCA_LED0_ON_L + 4 * ch);
  Wire.write((uint8_t)(on  & 0xFF));
  Wire.write((uint8_t)(on  >> 8));
  Wire.write((uint8_t)(off & 0xFF));
  Wire.write((uint8_t)(off >> 8));
  return Wire.endTransmission() == 0;
}

/* Microseconds → ticks. At 50 Hz one frame is 20000 µs across 4096 ticks, so
 * one tick is 4.88 µs — about 0.5 % of servo travel. Good enough that the
 * quantisation is invisible next to the servo's own backlash. */
static inline uint16_t usToTicks(uint16_t us) {
  float ticks = ((float)us * 4096.0f * PCA9685_PWM_FREQ_HZ) / 1000000.0f;
  if (ticks < 0) ticks = 0;
  if (ticks > 4095) ticks = 4095;
  return (uint16_t)(ticks + 0.5f);
}

static void writePulse(uint8_t ch, uint16_t us) {
  if (!s_pca_ok) return;
  pcaSetPWM(ch, 0, usToTicks(us));
}

#if ENABLE_PCA_WATCHDOG
/* Force every PCA9685 output fully off. Bit 4 of ALL_LED_OFF_H is the
 * datasheet's "full OFF" bit, which overrides whatever is in the per-channel
 * registers. An ESC seeing no pulses at all falls back to its own signal-loss
 * behaviour, which on the 880 is to cut the motor - the safe direction. */
static void pcaAllOutputsOff() {
  if (!s_pca_ok) return;
  pcaWrite8(PCA_ALL_LED_ON_L + 2, 0x00);
  pcaWrite8(PCA_ALL_LED_ON_L + 3, 0x10);
}

static void pcaOutputsResume() {
  if (!s_pca_ok) return;
  pcaWrite8(PCA_ALL_LED_ON_L + 2, 0x00);
  pcaWrite8(PCA_ALL_LED_ON_L + 3, 0x00);
}

/* Called once per drive cycle while the loop is healthy. If the loop stops
 * calling it - a hang, a brown-out, a blocked I2C transaction that never
 * returns - nothing re-arms the timer and the next call from ANY surviving
 * path trips it. The check is also made from update(), so a partially-alive
 * board still shuts the motor down rather than holding its last throttle. */
static void watchdogPet(uint32_t now_ms) {
  s_wd_last = now_ms;
  if (s_wd_tripped) {
    s_wd_tripped = false;
    pcaOutputsResume();
  }
}

static bool watchdogExpired(uint32_t now_ms) {
  return (uint32_t)(now_ms - s_wd_last) > PCA_WATCHDOG_MS;
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  Helpers
 * ══════════════════════════════════════════════════════════════════════════ */

static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

/* Move `cur` toward `target` no faster than `rate` units per second. */
static float slew(float cur, float target, float rate, float dt) {
  float max_step = rate * dt;
  float delta = target - cur;
  if (delta >  max_step) delta =  max_step;
  if (delta < -max_step) delta = -max_step;
  return cur + delta;
}

/* Steering fraction → pulse width, with travel limiting and optional invert. */
static uint16_t steerToUs(float s) {
  s = clampf(s, -1.0f, 1.0f) * SERVO_TRAVEL_LIMIT;
#if SERVO_INVERT
  s = -s;
#endif
  float span = (s >= 0.0f) ? (float)(SERVO_MAX_US - SERVO_CENTER_US)
                           : (float)(SERVO_CENTER_US - SERVO_MIN_US);
  return (uint16_t)(SERVO_CENTER_US + s * span + 0.5f);
}

/* Throttle fraction → ESC pulse width.
 *
 * The deadband skip is the important part. Mapping linearly from neutral
 * means the first ~6 % of commanded throttle does nothing but make the motor
 * sing. Instead, ANY non-zero request starts at neutral + deadband and the
 * remaining range is scaled into what is left. Exact zero still means an
 * exact neutral pulse, so "stop" is unambiguous. */
static uint16_t throttleToUs(float v) {
  v = clampf(v, -1.0f, 1.0f);
  if (fabsf(v) < 1e-3f) return ESC_NEUTRAL_US;

  if (v > 0.0f) {
    v = clampf(v, 0.0f, ESC_MAX_FWD_FRAC) / ESC_MAX_FWD_FRAC;
    float lo = (float)(ESC_NEUTRAL_US + ESC_DEADBAND_US);
    float hi = (float)ESC_NEUTRAL_US + (ESC_MAX_US - ESC_NEUTRAL_US) * ESC_MAX_FWD_FRAC;
    return (uint16_t)(lo + v * (hi - lo) + 0.5f);
  }
  float m = clampf(-v, 0.0f, ESC_MAX_REV_FRAC) / ESC_MAX_REV_FRAC;
  float lo = (float)(ESC_NEUTRAL_US - ESC_DEADBAND_US);
  float hi = (float)ESC_NEUTRAL_US - (ESC_NEUTRAL_US - ESC_MIN_US) * ESC_MAX_REV_FRAC;
  return (uint16_t)(lo + m * (hi - lo) - 0.5f);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ══════════════════════════════════════════════════════════════════════════ */

bool begin() {
  Wire.beginTransmission(PCA9685_ADDR);
  s_pca_ok = (Wire.endTransmission() == 0);
  if (!s_pca_ok) return false;

  /* Software reset, then totem-pole outputs and auto-increment on. */
  pcaWrite8(PCA_MODE1, PCA_MODE1_RESTART);
  delay(10);
  pcaWrite8(PCA_MODE1, PCA_MODE1_AI);
  pcaWrite8(PCA_MODE2, PCA_MODE2_OUTDRV);
  delay(5);

  if (!pcaSetFreq(PCA9685_PWM_FREQ_HZ)) { s_pca_ok = false; return false; }

  /* Park both channels before anything else can command them. */
  s_servo_us = SERVO_CENTER_US;
  s_esc_us   = ESC_NEUTRAL_US;
  writePulse(CH_SERVO, s_servo_us);
  writePulse(CH_ESC,   s_esc_us);

#if ENABLE_ESTOP_PIN
  pinMode(PIN_ESTOP, ESTOP_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
#endif

  s_arm_start   = millis();
  s_last_update = s_arm_start;
  s_wd_last     = s_arm_start;
  s_wd_tripped  = false;
  s_armed       = false;
  s_failsafe    = true;
  return true;
}

void setCommand(float v, float s, uint32_t now_ms) {
  s_req_v = clampf(v, -1.0f, 1.0f);
  s_req_s = clampf(s, -1.0f, 1.0f);
  s_last_cmd_ms = now_ms;
  s_frames++;
}

void stop() {
  s_req_v = 0.0f;
  s_req_s = 0.0f;
  s_app_v = 0.0f;
  s_esc_us = ESC_NEUTRAL_US;
  writePulse(CH_ESC, s_esc_us);
  s_rev = REV_IDLE;
}

bool watchdogTripped() {
#if ENABLE_PCA_WATCHDOG
  return s_wd_tripped;
#else
  return false;
#endif
}

void update(uint32_t now_ms) {
#if ENABLE_PCA_WATCHDOG
  /* Re-arm first: reaching here means the scheduler is alive. */
  watchdogPet(now_ms);
#endif
  float dt = (float)(now_ms - s_last_update) * 0.001f;
  s_last_update = now_ms;
  if (dt <= 0.0f || dt > 0.25f) dt = 1.0f / (float)LOOP_DRIVE_HZ;

  /* ── Arming: hold a steady neutral pulse until the ESC is happy ──────── */
  if (!s_armed) {
    writePulse(CH_ESC, ESC_NEUTRAL_US);
    writePulse(CH_SERVO, SERVO_CENTER_US);
    if (now_ms - s_arm_start >= ESC_ARM_MS) s_armed = true;
    return;
  }

  /* ── Hardware e-stop ─────────────────────────────────────────────────── */
#if ENABLE_ESTOP_PIN
  bool estop = (digitalRead(PIN_ESTOP) == (ESTOP_ACTIVE_LOW ? LOW : HIGH));
  if (estop) {
    s_app_v = 0.0f;
    s_esc_us = ESC_NEUTRAL_US;
    writePulse(CH_ESC, s_esc_us);
    writePulse(CH_SERVO, steerToUs(0.0f));
    s_rev = REV_IDLE;
    return;
  }
#endif

  /* ── Failsafe: the Pi went quiet ─────────────────────────────────────── */
  bool stale = (now_ms - s_last_cmd_ms) > CMD_TIMEOUT_MS;
  if (stale != s_failsafe) s_failsafe = stale;

  float tgt_v = s_failsafe ? 0.0f : s_req_v;
  float tgt_s = s_failsafe ? 0.0f : s_req_s;

  /* ── Reverse handshake ───────────────────────────────────────────────────
   * Entering reverse from forward motion: brake pulse, neutral gap, then
   * reverse. Entering reverse from a standstill still needs the gap, because
   * the ESC's own state machine does not know we were stopped.            */
#if ESC_REVERSE_HANDSHAKE
  bool want_rev = (tgt_v < -1e-3f);
  if (!want_rev) {
    s_rev = REV_IDLE;
    if (tgt_v > 1e-3f) s_was_fwd = true;
  } else {
    switch (s_rev) {
      case REV_IDLE:
        s_rev = REV_BRAKING;
        s_rev_timer = now_ms;
        break;
      case REV_BRAKING:
        /* One short backwards pulse — the ESC reads this as "brake". */
        s_app_v = 0.0f;
        s_esc_us = (uint16_t)(ESC_NEUTRAL_US - ESC_DEADBAND_US - 40);
        writePulse(CH_ESC, s_esc_us);
        writePulse(CH_SERVO, steerToUs(slew(s_app_s, tgt_s,
                    SERVO_MAX_SLEW_US_PER_S / 500.0f, dt)));
        if (now_ms - s_rev_timer >= ESC_BRAKE_PULSE_MS) {
          s_rev = REV_NEUTRAL_GAP;
          s_rev_timer = now_ms;
        }
        return;
      case REV_NEUTRAL_GAP:
        s_app_v = 0.0f;
        s_esc_us = ESC_NEUTRAL_US;
        writePulse(CH_ESC, s_esc_us);
        /* Steering keeps tracking during the gap: this is exactly when the
         * escape manoeuvre wants the wheels already swinging to lock. */
        s_app_s = slew(s_app_s, tgt_s, SERVO_MAX_SLEW_US_PER_S / 500.0f, dt);
        writePulse(CH_SERVO, steerToUs(s_app_s));
        if (now_ms - s_rev_timer >= ESC_BRAKE_PULSE_MS) {
          s_rev = REV_RUNNING;
          s_was_fwd = false;
        }
        return;
      case REV_RUNNING:
      default:
        break;
    }
  }
#endif

  /* ── Slew-limited application ────────────────────────────────────────── */
  float v_rate = ESC_MAX_SLEW_US_PER_S / 500.0f;    /* µs/s → fraction/s      */
  float s_rate = SERVO_MAX_SLEW_US_PER_S / 500.0f;
  s_app_v = slew(s_app_v, tgt_v, v_rate, dt);
  s_app_s = slew(s_app_s, tgt_s, s_rate, dt);

  /* Snap to exact zero once we are within a hair of it, so "stopped" really
   * writes 1500 µs rather than 1499 µs forever. */
  if (fabsf(s_app_v) < 0.004f) s_app_v = 0.0f;

  s_esc_us   = throttleToUs(s_app_v);
  s_servo_us = steerToUs(s_app_s);
  writePulse(CH_ESC,   s_esc_us);
  writePulse(CH_SERVO, s_servo_us);
}

bool     isArmed()          { return s_armed; }
bool     isFailsafe()       { return s_failsafe; }
bool     isPcaOk()          { return s_pca_ok; }
float    throttleFraction() { return s_app_v; }
float    steerFraction()    { return s_app_s; }
uint16_t escPulseUs()       { return s_esc_us; }
uint16_t servoPulseUs()     { return s_servo_us; }
uint32_t framesApplied()    { return s_frames; }

}  /* namespace Drive */
