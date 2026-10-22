/**
 * a2r_bot_drive.ino  (Option B — IMU on Arduino I2C)
 * ─────────────────────────────────────────────────────────────────────────
 * Firmware for Arduino Nano ESP32
 *
 * CHANGES FROM v1:
 *   1. MPU6050 IMU wired to A4 (SDA) / A5 (SCL) — shared I2C bus with PCA9685
 *   2. ADC divisor fixed: 1023 → 4095  (ESP32 is 12-bit, not 10-bit)
 *   3. Telemetry rate: 50 ms → 20 ms  (20 Hz → 50 Hz for EKF latency)
 *   4. IMU fields (ax,ay,az,gx,gy,gz) added to every telemetry packet
 *
 * WIRING (I2C bus — A4=SDA, A5=SCL shared):
 *   PCA9685 : addr 0x40  (hardware default, A0-A5 pads unsoldered)
 *   MPU6050 : addr 0x68  (AD0 pin → GND)
 *             VCC → 3.3V,  GND → GND,  SDA → A4,  SCL → A5
 *             INT pin left unconnected (polling mode used here)
 *
 * SERIAL PROTOCOL (115200 baud, newline-delimited JSON):
 *   Receive:  {"cmd":"DRIVE","v":0.0,"s":0.0}
 *             {"cmd":"STOP"}
 *             {"cmd":"RESET_ENC"}
 *             {"cmd":"CAL_IMU"}   ← new: trigger IMU offset calibration
 *   Send:     {"enc":0,"v_cmd":0.0,"cur_l":0.0,"cur_r":0.0,"bat_v":7.4,
 *              "e_stop":0,
 *              "ax":0.0,"ay":0.0,"az":9.81,
 *              "gx":0.0,"gy":0.0,"gz":0.0}
 * ─────────────────────────────────────────────────────────────────────────
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <ArduinoJson.h>

// ── Pin definitions ───────────────────────────────────────────────────────
// BTS7960 Motor Driver
#define PIN_RPWM     5
#define PIN_LPWM     6
#define PIN_R_EN     7
#define PIN_L_EN     8
#define PIN_I_RS     A0   // Right current sense
#define PIN_L_RS     A1   // Left  current sense
#define PIN_BATT_V   A2   // Battery voltage divider

// ── PCA9685 channels ──────────────────────────────────────────────────────
#define CH_STEERING  0    // Servo (1000–2000 µs)

// ── Servo timing ──────────────────────────────────────────────────────────
#define SERVO_FREQ_HZ   50
#define SERVO_MIN_US    1000
#define SERVO_MID_US    1500
#define SERVO_MAX_US    2000
#define US_TO_TICK(us)  ((uint16_t)((uint32_t)(us) * 4096UL / 20000UL))

// ── Motor parameters ──────────────────────────────────────────────────────
#define MAX_PWM         255

// [FIX 2026-07-03] ESP32 LEDC PWM for the BTS7960.
// Arduino-ESP32's analogWrite() is unreliable for motor drivers (wrong
// default frequency/resolution on several core versions) — a frequent cause
// of "servo works, motor doesn't". Drive RPWM/LPWM via LEDC instead.
#define MOTOR_PWM_FREQ  20000   // 20 kHz — above audible, fine for BTS7960
#define MOTOR_PWM_RES   8       // 8-bit → duty 0..255 (matches MAX_PWM)
// [FIX 2026-07-03] ESP32 Arduino core 3.x LEDC API addresses PWM by PIN
// (no channel numbers). ledcAttach(pin,freq,res) + ledcWrite(pin,duty).
#define MAX_CURRENT_A   20.0f
#define MIN_BATT_V      6.0f
#define MAX_BATT_V      8.4f
#define BATT_V_DIVIDER  2.0f

// ── Timing ────────────────────────────────────────────────────────────────
#define WATCHDOG_MS     500
#define TELEMETRY_MS    20    // ← 50 Hz (was 50 ms / 20 Hz)
#define SERIAL_BAUD     115200

// ── MPU6050 I2C registers ─────────────────────────────────────────────────
#define MPU_ADDR        0x68
#define MPU_REG_PWR     0x6B   // Power management
#define MPU_REG_SMPLRT  0x19   // Sample rate divider
#define MPU_REG_CONFIG  0x1A   // DLPF config
#define MPU_REG_GYRO_CF 0x1B   // Gyro full-scale
#define MPU_REG_ACCEL_CF 0x1C  // Accel full-scale
#define MPU_REG_ACCEL_X 0x3B   // First data register (AX_H)

// Sensitivity at default full-scale settings
// Accel ±2g  → 16384 LSB/g
// Gyro  ±250°/s → 131 LSB/(°/s)
#define ACCEL_SCALE     (9.80665f / 16384.0f)   // converts LSB → m/s²
#define GYRO_SCALE      (1.0f / 131.0f * 0.017453293f) // LSB → rad/s

// ── Globals ───────────────────────────────────────────────────────────────
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

volatile long  g_encoderCount = 0;
unsigned long  g_lastCmdMs    = 0;
unsigned long  g_lastTelMs    = 0;
float          g_velCmd       = 0.0f;
float          g_steerCmd     = 0.0f;
bool           g_saftyStop    = false;

// IMU state
float g_ax = 0.0f, g_ay = 0.0f, g_az = 9.80665f;
float g_gx = 0.0f, g_gy = 0.0f, g_gz = 0.0f;

// IMU calibration offsets (set by CAL_IMU command — hold car still)
float g_ax_off = 0.0f, g_ay_off = 0.0f, g_az_off = 0.0f;
float g_gx_off = 0.0f, g_gy_off = 0.0f, g_gz_off = 0.0f;
bool  g_imu_ok  = false;

// JSON buffers — enlarged to accommodate IMU fields
StaticJsonDocument<256> rxDoc;
StaticJsonDocument<256> txDoc;
char txBuf[256];

// ─────────────────────────────────────────────────────────────────────────
// MPU6050 initialisation
// ─────────────────────────────────────────────────────────────────────────
bool mpuInit() {
    // Wake up — clear sleep bit
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_PWR);
    Wire.write(0x00);
    if (Wire.endTransmission(true) != 0) {
        return false;   // Device not found on bus
    }

    delay(100);

    // Sample rate divider: SMPLRT_DIV = 0 → 1 kHz / (1+0) = 1000 Hz internal
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_SMPLRT);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // DLPF: ~42 Hz bandwidth (reduces vibration noise)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_CONFIG);
    Wire.write(0x03);
    Wire.endTransmission(true);

    // Gyro  ±250°/s
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_GYRO_CF);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // Accel ±2g
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_CF);
    Wire.write(0x00);
    Wire.endTransmission(true);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Read raw 16-bit registers and convert to SI units
// ─────────────────────────────────────────────────────────────────────────
void mpuRead() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_X);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14, true);   // 14 bytes: accel + temp + gyro

    if (Wire.available() < 14) return;

    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read();   // Temperature — discard
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    g_ax = (float)raw_ax * ACCEL_SCALE - g_ax_off;
    g_ay = (float)raw_ay * ACCEL_SCALE - g_ay_off;
    g_az = (float)raw_az * ACCEL_SCALE - g_az_off;
    g_gx = (float)raw_gx * GYRO_SCALE  - g_gx_off;
    g_gy = (float)raw_gy * GYRO_SCALE  - g_gy_off;
    g_gz = (float)raw_gz * GYRO_SCALE  - g_gz_off;
}

// ─────────────────────────────────────────────────────────────────────────
// IMU calibration — average 200 samples while car is stationary
// ─────────────────────────────────────────────────────────────────────────
void mpuCalibrate() {
    const int N = 200;
    double sax = 0, say = 0, saz = 0, sgx = 0, sgy = 0, sgz = 0;
    g_ax_off = g_ay_off = g_az_off = 0;
    g_gx_off = g_gy_off = g_gz_off = 0;

    for (int i = 0; i < N; i++) {
        mpuRead();
        sax += g_ax; say += g_ay; saz += g_az;
        sgx += g_gx; sgy += g_gy; sgz += g_gz;
        delay(5);
    }
    g_ax_off = (float)(sax / N);
    g_ay_off = (float)(say / N);
    // Keep az offset so gravity (≈9.81) stays — subtract mean then add 9.81
    g_az_off = (float)(saz / N) - 9.80665f;
    g_gx_off = (float)(sgx / N);
    g_gy_off = (float)(sgy / N);
    g_gz_off = (float)(sgz / N);

    Serial.println("{\"status\":\"IMU calibration done\"}");
}

// ─────────────────────────────────────────────────────────────────────────
// ADC helpers  — FIXED: 4095 for ESP32 12-bit ADC  (was 1023)
// ─────────────────────────────────────────────────────────────────────────
float readCurrentA(int pin) {
    int raw = analogRead(pin);
    float voltage = (float)raw * (3.3f / 4095.0f);   // ← FIXED
    return voltage;   // BTS7960: ~1V per Amp
}

float readBatteryV() {
    int raw = analogRead(PIN_BATT_V);
    float measuredV = (float)raw * (3.3f / 4095.0f);  // ← FIXED
    return measuredV * BATT_V_DIVIDER;
}

// ─────────────────────────────────────────────────────────────────────────
// Motor + steering helpers
// ─────────────────────────────────────────────────────────────────────────
void driveMotorsBTS(float vel) {
    int pwm = (int)(fabsf(vel) * MAX_PWM);
    pwm = constrain(pwm, 0, MAX_PWM);

    if (vel > 0.02f) {
        ledcWrite(PIN_LPWM, 0);
        ledcWrite(PIN_RPWM, pwm);
    } else if (vel < -0.02f) {
        ledcWrite(PIN_RPWM, 0);
        ledcWrite(PIN_LPWM, pwm);
    } else {
        ledcWrite(PIN_RPWM, 0);
        ledcWrite(PIN_LPWM, 0);
    }
}

void setSteeringNorm(float norm) {
    int us = (int)(SERVO_MID_US + norm * (SERVO_MAX_US - SERVO_MID_US));
    us = constrain(us, SERVO_MIN_US, SERVO_MAX_US);
    pca9685.setPWM(CH_STEERING, 0, US_TO_TICK(us));
}

void setSteeringUs(int us) {
    pca9685.setPWM(CH_STEERING, 0, US_TO_TICK(us));
}

// ─────────────────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);

    // BTS7960
    pinMode(PIN_RPWM, OUTPUT);
    pinMode(PIN_LPWM, OUTPUT);
    pinMode(PIN_R_EN, OUTPUT);
    pinMode(PIN_L_EN, OUTPUT);
    digitalWrite(PIN_R_EN, HIGH);
    digitalWrite(PIN_L_EN, HIGH);

    // [FIX 2026-07-03] LEDC PWM setup for BTS7960 (replaces analogWrite)
    ledcAttach(PIN_RPWM, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcAttach(PIN_LPWM, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcWrite(PIN_RPWM, 0);
    ledcWrite(PIN_LPWM, 0);

    // [FIX 2026-07-03] Make the 12-bit ADC assumption explicit and set input
    // attenuation so the pin can read up to ~3.3 V (needed for batt/current).
    analogReadResolution(12);
  #if defined(ARDUINO_ARCH_ESP32)
    analogSetAttenuation(ADC_11db);
  #endif

    // I2C bus — shared between PCA9685 and MPU6050
    Wire.begin();
    Wire.setClock(400000);   // 400 kHz fast mode

    // PCA9685
    pca9685.begin();
    pca9685.setOscillatorFrequency(27000000);
    pca9685.setPWMFreq(SERVO_FREQ_HZ);
    setSteeringUs(SERVO_MID_US);

    // MPU6050
    g_imu_ok = mpuInit();
    if (g_imu_ok) {
        Serial.println("{\"status\":\"MPU6050 OK\"}");
        // Auto-calibrate on boot (car must be stationary for ~1 s)
        mpuCalibrate();
    } else {
        Serial.println("{\"status\":\"MPU6050 NOT FOUND — check wiring\"}");
    }

    g_lastCmdMs = millis();
    Serial.println("{\"status\":\"a2r_bot_drive ready\"}");
}

// ─────────────────────────────────────────────────────────────────────────
// loop()
// ─────────────────────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();

    // ── Parse incoming command ────────────────────────────────────────
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() > 2) {
            DeserializationError err = deserializeJson(rxDoc, line);
            if (!err) {
                const char* cmd = rxDoc["cmd"] | "";
                if (strcmp(cmd, "DRIVE") == 0) {
                    g_velCmd   = constrain((float)(rxDoc["v"] | 0.0f), -1.0f, 1.0f);
                    g_steerCmd = constrain((float)(rxDoc["s"] | 0.0f), -1.0f, 1.0f);
                    g_lastCmdMs = now;
                } else if (strcmp(cmd, "STOP") == 0) {
                    g_velCmd = 0.0f; g_steerCmd = 0.0f;
                    g_lastCmdMs = now;
                } else if (strcmp(cmd, "RESET_ENC") == 0) {
                    g_encoderCount = 0;
                    g_lastCmdMs = now;
                } else if (strcmp(cmd, "CAL_IMU") == 0) {
                    // Recalibrate: keep the car still for ~1 s
                    mpuCalibrate();
                }
            }
        }
    }

    // ── Watchdog ─────────────────────────────────────────────────────
    if ((now - g_lastCmdMs) > WATCHDOG_MS) {
        g_velCmd = 0.0f; g_steerCmd = 0.0f;
    }

    // ── Read IMU ─────────────────────────────────────────────────────
    if (g_imu_ok) {
        mpuRead();
    }

    // ── Safety ───────────────────────────────────────────────────────
    float curL  = readCurrentA(PIN_I_RS);
    float curR  = readCurrentA(PIN_L_RS);
    float battV = readBatteryV();

    // [FIX 2026-07-03] Robust e-stop.
    // Previous logic tripped whenever battV < MIN_BATT_V — but if the battery
    // voltage divider saturates the ADC (BATT_V_DIVIDER too small for a 2S
    // pack) the reading is garbage and could sit right at the threshold,
    // latching the motor off intermittently ("works then doesn't").
    // A reading below a hard floor is treated as an IMPLAUSIBLE sensor fault
    // and IGNORED for battery cutoff, not as a real flat battery.
    // NOTE: set BATT_V_DIVIDER to match your ACTUAL resistors so a full pack
    // (~8.4 V) lands near ~3.0 V on the pin. Until then battery cutoff is
    // effectively disabled to avoid false trips.
    bool battPlausible = (battV > 3.0f);      // < 3 V ⇒ divider/wiring fault
    bool battLow       = battPlausible && (battV < MIN_BATT_V);
    bool overCurrent   = (curL > MAX_CURRENT_A) || (curR > MAX_CURRENT_A);

    g_saftyStop = overCurrent || battLow;

    if (g_saftyStop) {
        g_velCmd = 0.0f;
    }

    // ── Apply commands ───────────────────────────────────────────────
    driveMotorsBTS(g_velCmd);
    setSteeringNorm(g_steerCmd);

    // ── Telemetry (50 Hz) ────────────────────────────────────────────
    if ((now - g_lastTelMs) >= TELEMETRY_MS) {
        g_lastTelMs = now;
        txDoc.clear();

        // Drive telemetry
        txDoc["enc"]    = g_encoderCount;
        txDoc["v_cmd"]  = roundf(g_velCmd   * 1000.0f) / 1000.0f;
        txDoc["cur_l"]  = roundf(curL        * 100.0f)  / 100.0f;
        txDoc["cur_r"]  = roundf(curR        * 100.0f)  / 100.0f;
        txDoc["bat_v"]  = roundf(battV       * 100.0f)  / 100.0f;
        txDoc["e_stop"] = g_saftyStop ? 1 : 0;

        // IMU telemetry (zeroed if MPU not found)
        txDoc["ax"] = roundf(g_ax * 1000.0f) / 1000.0f;
        txDoc["ay"] = roundf(g_ay * 1000.0f) / 1000.0f;
        txDoc["az"] = roundf(g_az * 1000.0f) / 1000.0f;
        txDoc["gx"] = roundf(g_gx * 1000.0f) / 1000.0f;
        txDoc["gy"] = roundf(g_gy * 1000.0f) / 1000.0f;
        txDoc["gz"] = roundf(g_gz * 1000.0f) / 1000.0f;

        serializeJson(txDoc, txBuf);
        Serial.println(txBuf);
    }
}
