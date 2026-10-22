/* =============================================================================
 *  SafeSerial.cpp — see SafeSerial.h for why
 * ============================================================================= */
#include "SafeSerial.h"

LineOut LINK;

#if BOARD_IS_ESP32

size_t LineOut::write(uint8_t c) {
  m_buf[m_len++] = (char)c;
  if (c == '\n' || m_len >= TX_LINE_MAX) flushLine();
  return 1;
}

size_t LineOut::write(const uint8_t *buf, size_t n) {
  for (size_t i = 0; i < n; i++) write(buf[i]);
  return n;
}

void LineOut::flushLine() {
  const size_t len = m_len;
  m_len = 0;
  if (len == 0) return;

  /* NOTE (fw 3.4.1): no `if (!Serial)` check here. On the ESP32 core that flag
   * needs DTR *and* RTS and can stay false after some DTR/RTS toggles even
   * while the port is open, which silenced ALL telemetry. availableForWrite()
   * below already returns 0 when the host has the port closed (DTR low),
   * which is exactly what the stock firmware relied on. */

  if (m_need_nl) {
    if (LINK_PORT.availableForWrite() <= 0) { m_drops++; return; }
    LINK_PORT.write((uint8_t)'\n');
    m_need_nl  = false;
    m_mid_line = false;
  }

  const uint8_t *p = (const uint8_t *)m_buf;
  size_t left = len;
  bool started = false;
  const uint32_t t0 = micros();

  while (left) {
    int room = LINK_PORT.availableForWrite();
    if (room > 0) {
      size_t n = (size_t)room < left ? (size_t)room : left;
      size_t w = LINK_PORT.write(p, n);      /* n <= room: cannot spin */
      if (w == 0) break;
      p += w; left -= w; started = true;
      m_mid_line = (p[-1] != '\n');
    } else if (!started) {
      m_drops++;                          /* clean drop, nothing sent */
      if (m_mid_line) m_need_nl = true;
      return;
    }
    if ((uint32_t)(micros() - t0) > TX_LINE_BUDGET_US) break;
  }

  if (left) {                             /* cut short mid-line */
    m_torn++;
    m_need_nl = true;
  } else {
    m_last_ok = millis();
  }
}

#else  /* AVR: hardware UART always drains, plain pass-through */

size_t LineOut::write(uint8_t c) { return LINK_PORT.write(c); }
size_t LineOut::write(const uint8_t *buf, size_t n) { return LINK_PORT.write(buf, n); }

#endif
