/* =============================================================================
 *  Drive.h — PCA9685 → steering servo (ch0) + Hobbywing 880 ESC (ch1)
 * =============================================================================
 *  Takes the normalised command the Raspberry Pi sends
 *
 *      v  ∈ [-1, +1]   throttle   (+ forward)
 *      s  ∈ [-1, +1]   steering   (+ right, matching a negative yaw rate in
 *                                  the ROS convention — the planner has
 *                                  already done that conversion)
 *
 *  and turns it into two RC pulses. Everything that could damage hardware or
 *  surprise a bystander lives here rather than upstream: arming, the ESC
 *  deadband skip, slew limits, the reverse handshake and the failsafe.
 *
 *  DESIGN NOTE — WHY NO Adafruit_PWMServoDriver
 *  --------------------------------------------
 *  The PCA9685 needs seven register writes and a prescaler calculation. A
 *  library for that is an extra install step, an extra version to pin, and a
 *  layer between you and the datasheet when something misbehaves. The raw
 *  driver below is about 60 lines and depends on nothing but Wire.h.
 * ============================================================================= */

#ifndef A2R_DRIVE_H
#define A2R_DRIVE_H

#include <Arduino.h>
#include "Config.h"

namespace Drive {

/* Lifecycle ---------------------------------------------------------------- */

/* Brings up the PCA9685, parks the servo at centre, then holds the ESC at
 * neutral for ESC_ARM_MS. Returns false if the PCA9685 did not ACK, in which
 * case the car will not move and the firmware says so on every heartbeat. */
bool begin();

/* Call at LOOP_DRIVE_HZ. Applies slew limits, runs the arming and reverse
 * handshake state machines, and enforces the command timeout. This is the
 * function that actually writes pulses, so if it stops being called, the car
 * stops — which is the intended failure mode. */
void update(uint32_t now_ms);

/* Command ------------------------------------------------------------------ */

/* New DRIVE frame from the Pi. Values outside [-1, 1] are clamped, not
 * rejected, so a slightly out-of-range planner cannot stall the car. */
void setCommand(float v, float s, uint32_t now_ms);

/* Immediate neutral: motor to 1500 µs, steering to centre. Used by the STOP
 * frame, the failsafe and the e-stop. */
void stop();

/* State (read by Protocol for telemetry) ----------------------------------- */

bool  isArmed();
bool  isFailsafe();          /* true while the Pi has been silent too long   */
/* True when the PCA9685 watchdog cut the outputs because the main loop stopped
 * re-arming it (board hung / browned out) - the one failure the software
 * failsafe cannot cover, because it needs the software to still be running. */
bool  watchdogTripped();
bool  isPcaOk();
float throttleFraction();    /* last APPLIED throttle, after all limiting.
                                Battery sag compensation uses this — it must
                                be what the motor got, not what was asked. */
float steerFraction();
uint16_t escPulseUs();
uint16_t servoPulseUs();

/* Diagnostics: total DRIVE frames applied since boot. A Pi-side log showing
 * frames sent but this counter flat means the link, not the planner. */
uint32_t framesApplied();

}  /* namespace Drive */

#endif /* A2R_DRIVE_H */
