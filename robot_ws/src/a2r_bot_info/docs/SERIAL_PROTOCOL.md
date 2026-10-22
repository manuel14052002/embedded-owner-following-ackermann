# Serial protocol

Newline-delimited JSON, 115200 8N1, both directions.  Unchanged from the
original workspace, so existing firmware and existing tools still work.

JSON was kept rather than replaced with a binary framing because at 20 Hz and
115200 baud the overhead is irrelevant, and being able to debug the link with
nothing but a serial monitor is worth a great deal at 2 a.m.

## Pi → Arduino

| Command | Fields | Meaning |
|---|---|---|
| `{"cmd":"DRIVE","v":0.4,"s":-0.2}` | `v`,`s` ∈ [-1, 1] | Throttle and steer, normalised. `+s` steers **left** (REP-103). |
| `{"cmd":"STOP"}` | — | Neutral immediately. |
| `{"cmd":"CAL_IMU"}` | — | Average the gyro for 2 s and store the bias. Robot must be still and level. |
| `{"cmd":"RESET_ENC"}` | — | Zero the encoder count. |
| `{"cmd":"PING"}` | — | Reply with a status line. |
| `{"cmd":"DISP","t":"87P"}` | `t`, ≤4 chars | Write to the MAX7219. |
| `{"cmd":"DIM","v":7}` | `v` ∈ [0, 15] | Display brightness. |

## Arduino → Pi

| Line | Fields | Rate |
|---|---|---|
| `{"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..}` | m/s², rad/s, body frame | ~50 Hz |
| `{"mx":..,"my":..,"mz":..}` | µT, body frame | ~25 Hz |
| `{"f":1.23,"r":0.85}` | metres, or `-1` for no echo | ~10 Hz |
| `{"bat_v":16.42,"bat_p":87}` | volts, percent | 1 Hz |
| `{"baro":101325.0,"temp":21.5}` | Pa, °C | 2 Hz |
| `{"enc":1234}` | counts | 10 Hz |
| `{"e_stop":true}` | — | on change |
| `{"status":"armed"}` | free text | 1 Hz |

## The three rules that matter

**1. The failsafe is in the firmware, not the Pi.**  If no `DRIVE` arrives
within `CMD_TIMEOUT_MS` (500 ms), the firmware goes to neutral on its own.
The bridge has its own 350 ms timeout above that, so a hung bridge stops the
car and a crashed Pi also stops the car.  Two independent timeouts, because
the failure you did not think of is the one that happens.

**2. Reverse is a handshake, not a negative number.**  The HobbyWing 880, like
most car ESCs, needs brake → neutral → reverse.  Sending 1300 µs from forward
just brakes.  The firmware implements the sequence; the protocol hides it.
`v = -0.4` means reverse and the firmware works out how.

**3. Arming is not optional.**  The ESC needs continuous neutral at power-up
before it will respond to anything.  The firmware holds channel 1 at exactly
1500 µs for 3000 ms and refuses all `DRIVE` commands until then.  If the ESC
beeps forever, either the Arduino was already sending throttle at power-up, or
the ESC's own neutral is not 1500 µs.

## Testing it by hand

```bash
# Who owns the port? Nothing should, before you start.
sudo lsof /dev/ttyARDUINO

screen /dev/ttyARDUINO 115200     # or: picocom -b 115200 /dev/ttyARDUINO
```

Then, with the car **on blocks**:

```json
{"cmd":"PING"}
{"cmd":"CAL_IMU"}
{"cmd":"DRIVE","v":0,"s":0.3}
{"cmd":"DRIVE","v":0.15,"s":0}
{"cmd":"STOP"}
```

Positive `s` must steer left.  If it steers right, flip `SERVO_REVERSED` in
`Config.h` rather than negating it somewhere in the ROS stack — the sign
convention should be true at the lowest level, or every layer above inherits
the confusion.
