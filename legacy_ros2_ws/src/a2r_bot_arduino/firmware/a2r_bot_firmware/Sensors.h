/* =============================================================================
 *  Sensors.h — GY-91 (MPU-9250 + BMP280) and 2 × DYP-A21 ultrasonic
 * =============================================================================
 *  Everything here produces RAW, CALIBRATED-BUT-UNFILTERED physical units and
 *  hands them straight to the Pi. No orientation filter, no Kalman, no
 *  smoothing beyond what a sensor's own DLPF does.
 *
 *  That split is deliberate. Attitude estimation and sensor fusion live in
 *  a2r_bot_sensor_fusion on the Raspberry Pi, where there is double
 *  precision, numpy, a proper covariance representation and — above all — a
 *  place to log and replay them. Running a Madgwick filter here would mean
 *  the only copy of the raw data is the one nobody can inspect.
 *
 *  So the contract is:  this board measures, the Pi believes or disbelieves.
 *
 *  UNITS (REP-103, so nothing downstream has to convert)
 *  -----------------------------------------------------
 *      accel   m/s²        body frame, x forward, y left, z up
 *      gyro    rad/s       right-handed about the same axes
 *      mag     µT          sensor frame, see the axis note in readMag()
 *      range   metres
 *      temp    °C
 *      press   Pa
 * ============================================================================= */

#ifndef A2R_SENSORS_H
#define A2R_SENSORS_H

#include <Arduino.h>
#include "Config.h"

namespace Sensors {

struct ImuSample {
  float ax, ay, az;      /* m/s²                                             */
  float gx, gy, gz;      /* rad/s, gyro bias already removed                 */
  float mx, my, mz;      /* µT   (zero if ENABLE_MAG is 0 or no AK8963)      */
  float temp_c;
  bool  mag_valid;
};

struct RangeSample {
  float range_m;
  bool  valid;
  uint32_t stamp_ms;
};

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

/* Probes and configures every device on the I2C bus. Missing devices are
 * recorded, not fatal: a robot with a dead magnetometer should still drive,
 * and the Pi should be told which sensors it is actually getting. */
bool begin();

/* Re-scan the bus. Useful after a brownout, which knocks I2C devices off
 * more often than it resets the MCU. */
void rescan(); 

/* Free a hung I2C bus (fw 3.4.0). A slave that was interrupted mid-byte -
 * by a reset, a brownout or a vibrating connector - can hold SDA low forever,
 * and no amount of re-initialising the ESP32's I2C peripheral fixes that.
 * Clocking SCL up to 9 times lets the slave finish its byte and release SDA;
 * a STOP then returns the bus to idle. Re-initialises Wire afterwards. */
void i2cRecoverBus();

/* ── Per-cycle work ────────────────────────────────────────────────────── */

void updateImu(uint32_t now_ms);        /* at IMU_PUBLISH_HZ                 */
void updateBaro(uint32_t now_ms);       /* at BARO_PUBLISH_HZ                */
void updateUltrasonic(uint32_t now_ms); /* alternates front/rear             */

/* ── Access ────────────────────────────────────────────────────────────── */

const ImuSample&   imu();
const RangeSample& rangeFront();
const RangeSample& rangeRear();
float pressurePa();
float baroTempC();

bool imuPresent();
bool magPresent();
bool baroPresent();
bool ultrasonicFrontPresent();
bool ultrasonicRearPresent();
/* Cumulative sonar link counters (UART mode): raw bytes, valid frames,
 * checksum failures. idx 0 = front, 1 = rear. */
void ultrasonicStats(uint8_t idx, uint32_t &rx, uint32_t &frm, uint32_t &bad);
/* Boot-time idle level of sonar wire i (0 front rx, 1 front tx, 2 rear rx,
 * 3 rear tx): 1 driven high by a powered device, 0 low, 2 not probed. */
uint8_t  ultrasonicIdle(uint8_t i);
/* How many times the IMU dropped out and was re-initialised. */
uint32_t imuRecoveries();

/* ── Gyro bias ─────────────────────────────────────────────────────────── */

/* Kick off a bias estimate. Only completes while the car is actually still;
 * if it is moving, the samples are discarded and it keeps waiting, so
 * calibrating by accident on a moving robot is not possible.
 * `is_commanded_still` should be true when the throttle is at neutral. */
void beginGyroCalibration();
void updateGyroCalibration(bool is_commanded_still);
bool gyroCalibrated();
void gyroBias(float &bx, float &by, float &bz);

}  /* namespace Sensors */

#endif /* A2R_SENSORS_H */
