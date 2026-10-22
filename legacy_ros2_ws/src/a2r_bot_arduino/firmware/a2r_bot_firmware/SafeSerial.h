/* =============================================================================
 *  SafeSerial.h — telemetry output that can NEVER stall the main loop
 * =============================================================================
 *  WHY THIS EXISTS (fw 3.4.0)
 *  ──────────────────────────
 *  On the Nano ESP32 (TinyUSB CDC, arduino-esp32 core 2.0.x) Serial.write()
 *  is:
 *
 *      while (to_send) {
 *          space = tud_cdc_n_write_available(itf);
 *          if (!space) { tud_cdc_n_write_flush(itf); continue; }   <-- forever
 *          ...
 *
 *  There is no timeout. The USB TX FIFO is only 64 bytes and one IMU line is
 *  ~140, so whenever the Pi stops draining the link - a busy moment on the
 *  Pi, the bridge sleeping while it reopens the port, a TinyUSB hiccup - the
 *  first Serial.print() that finds the FIFO full never returns. loop() stops,
 *  telemetry stops, the drive failsafe stops, and the PCA9685 keeps driving
 *  the ESC at the last pulse it was given. That is the "Arduino silent for
 *  8 s" in the bridge log, and why only unplugging the cable fixed it.
 *  (Espressif later added a timeout to this loop upstream for this reason.)
 *
 *  WHAT THIS DOES
 *  ──────────────
 *  `LINK` is a Print object. All telemetry goes through it (LINK.print(...)).
 *  It collects one line, then hands it to USB only as fast as the FIFO has
 *  room, with a hard time budget per line:
 *
 *      host not connected (DTR low)     -> line dropped, zero wait
 *      FIFO full before the line starts -> line dropped, zero wait
 *      FIFO fills mid-line, budget hit  -> rest dropped, and the NEXT line
 *                                          starts with '\n' so the Pi can
 *                                          resynchronise instead of gluing
 *                                          two half-lines into bad JSON
 *
 *  Old telemetry is worthless: a 50 Hz IMU sample that could not be sent now
 *  is replaced 20 ms later. Dropping it is the correct behaviour; waiting
 *  for it is what froze the car. Drops are counted and reported in the
 *  heartbeat ("txdrop"/"torn") so a bad link is visible, not silent.
 *
 *  On AVR (classic Nano, hardware UART that always drains) this is a plain
 *  pass-through to Serial.
 * ============================================================================= */
#ifndef A2R_SAFESERIAL_H
#define A2R_SAFESERIAL_H

#include <Arduino.h>
#include "Config.h"

class LineOut : public Print {
public:
  size_t write(uint8_t c) override;
  size_t write(const uint8_t *buf, size_t n) override;
  using Print::write;

  uint32_t drops() const { return m_drops; }
  uint32_t torn()  const { return m_torn;  }
  uint32_t lastOkMs() const { return m_last_ok; }   /* last line fully sent */

private:
#if BOARD_IS_ESP32
  void    flushLine();
  char    m_buf[TX_LINE_MAX];
  size_t  m_len      = 0;
  bool    m_mid_line = false;   /* last byte actually sent was not '\n'    */
  bool    m_need_nl  = false;   /* a line was cut short: resync with '\n'  */
#endif
  uint32_t m_drops = 0;
  uint32_t m_torn  = 0;
  uint32_t m_last_ok = 0;
};

extern LineOut LINK;

#endif
