/* =============================================================================
 *  Protocol.h — newline-delimited JSON over USB serial, Pi ↔ Arduino
 * =============================================================================
 *  One JSON object per line, both directions, 921600 8N1. Deliberately the
 *  same wire format the previous firmware used, so a2r_bot_follow's planner
 *  talks to this board without a single change.
 *
 *  ── PI → ARDUINO ──────────────────────────────────────────────────────────
 *    {"cmd":"DRIVE","v":0.42,"s":-0.31}   throttle and steering, both -1..1
 *    {"cmd":"STOP"}                       immediate neutral
 *    {"cmd":"CAL_IMU"}                    re-run the gyro bias estimate
 *    {"cmd":"PING"}                       ask for an immediate status line
 *    {"cmd":"DISP","text":"HI  "}         write 4 chars to the 7-segment
 *    {"cmd":"DIM","level":3}              display brightness 0..15
 *
 *  ── ARDUINO → PI ──────────────────────────────────────────────────────────
 *    {"t":123456,"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..,
 *                "mx":..,"my":..,"mz":..,"tc":..}          @ IMU_PUBLISH_HZ
 *    {"t":..,"us_f":1.83,"us_r":0.44}                      @ US_PUBLISH_HZ
 *    {"t":..,"bat_v":15.42,"bat_pct":68.0,"bat_cell":3.855,
 *            "bat_mah":4590,"bat_low":0}                   @ 2 Hz
 *    {"t":..,"press":100831.2,"btc":24.9}                  @ BARO_PUBLISH_HZ
 *    {"status":"...","fw":"3.0.0","board":"nano_esp32", ...} @ STATUS_PERIOD_MS
 *    {"e_stop":1}                                          on change
 *
 *  ── WHY NOT ArduinoJson ───────────────────────────────────────────────────
 *  The inbound grammar is six fixed commands with numeric fields. A 30-line
 *  scanner handles it with no heap, no library version to pin, and no
 *  possibility of a fragmentation stall three hours into a run. Outbound is
 *  hand-formatted for the same reason.
 *
 *  ── ROBUSTNESS ────────────────────────────────────────────────────────────
 *  A line longer than RX_BUFFER_LEN is discarded up to the next newline
 *  rather than silently truncated, because a truncated DRIVE frame that
 *  still parses is far worse than a dropped one.
 * ============================================================================= */

#ifndef A2R_PROTOCOL_H
#define A2R_PROTOCOL_H

#include <Arduino.h>
#include "Config.h"

namespace Protocol {

void begin();

/* Drain the serial RX buffer and dispatch any complete lines. Call often —
 * at LOOP_SERIAL_HZ or faster. Non-blocking. */
void poll(uint32_t now_ms);

/* Telemetry emitters. Each rate-limits itself, so calling them every loop is
 * correct and cheap. */
void publishImu(uint32_t now_ms);
void publishRanges(uint32_t now_ms);
void publishBattery(uint32_t now_ms);
void publishBaro(uint32_t now_ms);
void publishStatus(uint32_t now_ms, bool force = false);

/* Link beat (fw 3.4.0): {"lk":<last cmd seq>,"qa":<ms since it arrived>,
 * "t":<ms>,"fs":<failsafe 0/1>} at LINK_PUBLISH_HZ, always. Lets the Pi
 * measure command round-trip and detect a dead link within ~1 s. */
void publishLink(uint32_t now_ms);

/* millis() of the last command carrying a sequence number (0 = none yet) */
uint32_t lastCmdMs();

/* Counters, reported in the status line. rxErrors climbing while rxFrames is
 * flat is the signature of a baud mismatch or a noisy cable. */
uint32_t rxFrames();
uint32_t rxErrors();

}  /* namespace Protocol */

#endif /* A2R_PROTOCOL_H */
