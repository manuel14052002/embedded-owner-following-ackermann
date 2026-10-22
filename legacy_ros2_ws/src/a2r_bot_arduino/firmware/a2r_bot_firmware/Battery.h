/* =============================================================================
 *  Battery.h — pack voltage on A0 → percentage → MAX7219 / FYQ-5641 display
 * =============================================================================
 *  Pack: Tattu TAA67504S25X9 — 14.8 V 6750 mAh 4S1P 25C LiPo
 *
 *  Reading "14.2 V" off a divider and calling it a fuel gauge gives you a
 *  number that lies in three different ways at once, so this module deals
 *  with all three:
 *
 *    1. NOISE.   The ADC shares a ground with a brushed motor. Raw samples
 *                scatter by tenths of a volt. → median-of-15, then a slow EMA.
 *
 *    2. SAG.     Under throttle the pack drops by I·R_internal and the gauge
 *                appears to lose 30 % every time you accelerate. We have no
 *                current sensor, but we know the throttle we just applied, so
 *                we estimate the open-circuit voltage from it.
 *
 *    3. CURVE.   LiPo voltage is NOT linear in charge. A 4S pack spends most
 *                of its life between 15.2 V and 15.8 V; linear interpolation
 *                between "empty" and "full" reports ~50 % for most of the
 *                flight and then falls off a cliff. → 21-point discharge LUT.
 *
 *  The display is driven by a bit-banged MAX7219 driver with a full segment
 *  font, so it can show "16.8" with a decimal point, "LO" when the pack is
 *  nearly flat, and "bAt" / "Err" during start-up and faults — not just
 *  digits.
 * ============================================================================= */

#ifndef A2R_BATTERY_H
#define A2R_BATTERY_H

#include <Arduino.h>
#include "Config.h"

namespace Battery {

/* Brings up the MAX7219 (scan limit, intensity, no-decode mode) and runs a
 * brief self-test so a dead display is obvious before you go looking for a
 * software bug: all segments on, then "bAt", then the first reading.
 * Returns false only if the display is disabled at compile time. */
bool begin();

/* Sample the divider, filter, compensate for sag and update the LUT lookup.
 * Call at any rate; it rate-limits itself to BATTERY_READ_PERIOD_MS.
 * `throttle_frac` is the APPLIED throttle from Drive::throttleFraction(). */
void update(uint32_t now_ms, float throttle_frac);

/* Refresh what is on the 7-segment display. Separated from update() so the
 * display can run at its own comfortable rate and so you can call it during
 * a fault to show an error code without disturbing the filter state. */
void updateDisplay(uint32_t now_ms);

/* ── Readings ─────────────────────────────────────────────────────────── */

float    packVoltage();        /* filtered, at the terminals (volts)         */
float    restingVoltage();     /* sag-compensated open-circuit estimate      */
float    cellVoltage();        /* restingVoltage() / BATTERY_CELLS           */
float    percent();            /* 0..100, from the discharge curve           */
float    remainingMah();       /* percent × capacity, for convenience        */
bool     isLow();              /* below CELL_CRITICAL_V per cell             */
bool     isCritical();         /* at or below CELL_EMPTY_V per cell          */
bool     isValid();            /* enough samples collected to be believed    */

/* ── Display control ──────────────────────────────────────────────────── */

/* Show an arbitrary 4-character string. Supported glyphs: 0-9 A b C d E F H
 * I L n o P r S t U u y - and space. A '.' immediately after a character
 * lights that character's decimal point and does not consume a digit, so
 * "16.8" fits in three digits plus the point. */
void showText(const char *text);

/* Show a signed integer, right-aligned. Values outside -999..9999 clamp. */
void showNumber(int value);

/* Blank the display (used at shutdown so it does not sit lit on a flat pack). */
void blank();

/* Set brightness 0..15 at runtime — handy if the display washes out outdoors. */
void setIntensity(uint8_t level);

}  /* namespace Battery */

#endif /* A2R_BATTERY_H */
