#!/usr/bin/env python3
"""gps_owner_node - OUTDOOR mode (a2r_step3f): where is the owner when neither camera nor lidar sees them?

Sources: the SIM7600 on the Pi gives the ROBOT's fix (AT+CGNSSINFO: position + HDOP + satellites) and
reads the OWNER's phone position from Firebase live_location (the A2R phone app) over its own SIM.

What a2r_step3f changed, after the 2026-10-09 outdoor run and Lee, Cho & Lim, "GPS-Based Human
Tracking Methods for Outdoor Robots", J. KSPE 35(4), 2018 ("the paper"):

  * The robot's own GNSS was the big error source: it jumped 40 m and 77 m within 15 s next to the
    building and spread over 47 x 66 m while the robot stood still, while the phone stayed within
    15 x 11 m. That is why "36 m" was shown when the owner was 3 m away.
      - the fix now comes WITH its quality (AT+CGNSSINFO: HDOP, satellites); sigma = 4 m x HDOP;
        fixes with < 5 satellites or HDOP > 5 are not used for navigation at all;
      - like the paper's EKF (encoder + DGPS + compass), GPS is FUSED with odometry: the GPS->odom
        offset is a small Kalman filter, so one fix moves it only a little, and a fix that disagrees
        with odometry by more than a 3-sigma gate (a jump) is rejected outright;
      - north (where GPS north lies in the odom frame) still comes from GPS-vs-odometry displacement
        pairs and the GNSS course while driving - the paper uses a digital compass; ours (the IMU
        magnetometer) is unusable next to the motors - but course samples are only taken when the
        GNSS speed agrees with the odometry speed (the log showed 2.4 m/s "GNSS speed" while crawling).
  * The owner's positions are SMOOTHED with the paper's moving average (eq. 10-12, m = 3) after
    rejecting impossible jumps (> 3 m/s), and kept as a TRAIL.
  * The trail is published as a GLOBAL PATH (/gps/owner_path, odom). follow_manager follows it
    point by point (the paper's "trajectory tracking", around buildings the owner walked round) and
    cuts straight to a later point when the way is clear ("direct tracking"); a point counts as
    reached within the paper's eq. 14 radius (robot sigma + owner sigma).
  * The REAL error is measured: whenever lidar/camera see the owner, their position is compared with
    the GPS estimate, and the RMS of that difference becomes the minimum sigma. So the robot learns
    how far it may trust GPS at this place, and hands over to the lidar/camera search inside it.

Topics  in : /a2r/mode_request (String), /follow/owner (PointStamped base_link: owner seen)
        out: /a2r/mode (String, latched), /gps/robot_fix, /gps/owner_fix (NavSatFix, raw),
             /gps/owner_estimate (PointStamped odom, point.z = 1-sigma metres),
             /gps/owner_path (nav_msgs/Path odom: smoothed owner trail = global path), /gps/status (JSON)
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import deque

R_EARTH = 6371000.0
UERE_M = 4.0           # user equivalent range error: 1-sigma position ~ UERE x HDOP (consumer GNSS)


# ============================================================ parsers (unit-tested without ROS)

def _dm(v: float) -> float:
    d = int(v // 100)
    return d + (v - d * 100) / 60.0


def parse_cgpsinfo(response: str):
    """+CGPSINFO ddmm.mmmm,N,dddmm.mmmm,E,... -> (lat, lon) or None (GPS.py)."""
    m = re.search(r"\+CGPSINFO:\s*(\d+(?:\.\d+)?),([NS]),(\d+(?:\.\d+)?),([EW])", response)
    if not m:
        return None
    lat, lon = _dm(float(m.group(1))), _dm(float(m.group(3)))
    lat, lon = (-lat if m.group(2) == "S" else lat), (-lon if m.group(4) == "W" else lon)
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def parse_cgpsinfo_full(response: str):
    """(lat, lon, speed m/s|None, course deg|None, hdop None, sats None) from +CGPSINFO."""
    ll = parse_cgpsinfo(response)
    if ll is None:
        return None
    m = re.search(r"\+CGPSINFO:\s*[^,]*,[NS],[^,]*,[EW],[^,]*,[^,]*,[^,]*,([\d.]*),([\d.]*)", response)
    spd = crs = None
    if m:
        try:
            spd = float(m.group(1)) * 0.514444 if m.group(1) else None
            crs = float(m.group(2)) if m.group(2) else None
        except ValueError:
            spd = crs = None
    return ll[0], ll[1], spd, crs, None, None


def parse_cgnssinfo(response: str):
    """+CGNSSINFO: mode,GPS-SVs,GLONASS-SVs,BEIDOU-SVs[,GALILEO-SVs],lat,N,lon,E,date,utc,alt,speed(kn),
    course,PDOP,HDOP,VDOP -> (lat, lon, speed m/s|None, course|None, hdop|None, sats|None) or None."""
    m = re.search(r"\+CGNSSINFO:\s*([^\r\n]*)", response)
    if not m:
        return None
    f = [x.strip() for x in m.group(1).split(",")]
    try:
        i = next(k for k in range(1, len(f) - 3) if f[k + 1] in ("N", "S") and f[k + 3] in ("E", "W"))
    except StopIteration:
        return None
    try:
        lat, lon = _dm(float(f[i])), _dm(float(f[i + 2]))
    except ValueError:
        return None
    lat, lon = (-lat if f[i + 1] == "S" else lat), (-lon if f[i + 3] == "W" else lon)
    sats = 0
    for x in f[1:i]:
        try:
            sats += int(x)
        except ValueError:
            pass
    rest = f[i + 4:]                      # date, utc, alt, speed, course, pdop, hdop, vdop

    def fl(k):
        try:
            return float(rest[k]) if len(rest) > k and rest[k] != "" else None
        except ValueError:
            return None
    spd = fl(3)
    spd = spd * 0.514444 if spd is not None else None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon, spd, fl(4), fl(6), (sats or None)


def parse_firebase(body: str):
    """Firebase live_location (A2R app: latitude, longitude, speed_mps, heading_degrees, last_updated)
    -> (lat, lon, acc|None, ts|None, speed m/s, heading deg|None) or None."""
    try:
        o = json.loads(body)
        lat, lon = float(o["latitude"]), float(o["longitude"])
        acc = o.get("accuracy")
        acc = float(acc) if acc is not None else None
        ts = o.get("timestamp", o.get("last_updated"))
        spd = float(o.get("speed_mps") or 0.0)
        hdg = o.get("heading_degrees")
        hdg = float(hdg) if hdg is not None else None
    except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if not (math.isfinite(spd) and 0.0 <= spd < 10.0):
        spd = 0.0
    return lat, lon, acc, ts, spd, hdg


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.atan2(math.sqrt(min(1.0, a)), math.sqrt(max(0.0, 1 - a)))


def ll_to_enu(lat, lon, lat0, lon0):
    return (math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * R_EARTH)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def robot_sigma(hdop, sats, default=5.0):
    """1-sigma horizontal error of a robot fix (m); None = do not use for navigation."""
    if sats is not None and sats < 5:
        return None
    if hdop is None:
        return default
    if hdop > 5.0:
        return None
    return max(2.5, UERE_M * hdop)


# ============================================================ estimation

class GpsAligner:
    """odom = R(psi) * enu + b.
    psi : north in odom - from GPS/odometry displacement pairs (2-D Kabsch) and GNSS-course samples.
    b   : the GPS->odom offset, a 2-D Kalman filter (random walk = odometry drift); each robot fix is a
          measurement with its own sigma, gated at 3 sigma - a GPS jump is rejected, not averaged in
          (the paper's EKF fuses DGPS with dead reckoning the same way)."""

    def __init__(self, min_base_m=6.0, window_s=180.0, max_scale_err=0.4, max_psi_sigma_deg=12.0,
                 min_pairs=3, drift_mps=0.05):
        self.min_base, self.window, self.max_scale = min_base_m, window_s, max_scale_err
        self.max_sig, self.min_pairs = math.radians(max_psi_sigma_deg), min_pairs
        self.samples = deque(maxlen=600)      # (t, e, n, ox, oy, sigma, kind)
        self.headings = deque(maxlen=300)     # (t, psi, weight)
        self.psi, self.psi_sigma, self.npairs = None, math.inf, 0
        self.b, self.Pb, self.b_t = None, None, None
        self.drift = drift_mps
        self.accepted = self.rejected = 0

    def add(self, t, e, n, ox, oy, sigma_m, kind="robot"):
        self.samples.append((t, e, n, ox, oy, max(0.5, sigma_m), kind))

    def add_heading(self, t, psi, w=1.0):
        self.headings.append((t, _wrap(psi), w))

    def heading_estimate(self, now, min_n=8):
        H = [h for h in self.headings if now - h[0] <= self.window]
        if len(H) < min_n:
            return None
        sw = sum(w for _, _, w in H)
        sx, cx = sum(w * math.sin(p) for _, p, w in H), sum(w * math.cos(p) for _, p, w in H)
        R = min(1.0, math.hypot(sx, cx) / sw)
        std = math.sqrt(max(1e-9, -2.0 * math.log(max(R, 1e-9))))
        n_eff = sw * sw / sum(w * w for _, _, w in H)
        return math.atan2(sx, cx), max(std / math.sqrt(n_eff), math.radians(3.0))

    def update(self, now):
        S = [s for s in self.samples if now - s[0] <= self.window]
        pairs = []
        for kind in ("robot", "owner"):
            K = [s for s in S if s[6] == kind]
            for i in range(len(K)):
                for j in range(i + 1, len(K)):
                    do = (K[j][3] - K[i][3], K[j][4] - K[i][4])
                    lo = math.hypot(*do)
                    if lo < self.min_base:
                        continue
                    de = (K[j][1] - K[i][1], K[j][2] - K[i][2])
                    le = math.hypot(*de)
                    if le > 1e-6 and abs(le / lo - 1.0) <= self.max_scale:
                        pairs.append((de, do, lo * lo / (K[i][5] ** 2 + K[j][5] ** 2)))
                    break
        self.npairs = len(pairs)
        est = []
        if self.npairs >= self.min_pairs:
            sx = sum(w * (de[0] * do[1] - de[1] * do[0]) for de, do, w in pairs)
            sy = sum(w * (de[0] * do[0] + de[1] * do[1]) for de, do, w in pairs)
            psi = math.atan2(sx, sy)
            res = [_wrap(math.atan2(do[1], do[0]) - math.atan2(de[1], de[0]) - psi) for de, do, _ in pairs]
            rms = math.sqrt(sum(r * r for r in res) / len(res))
            est.append((psi, max(rms / math.sqrt(len(res)), math.radians(2.0))))
        hd = self.heading_estimate(now)
        if hd is not None:
            est.append(hd)
        if not est:
            self.psi, self.psi_sigma = None, math.inf
            return False
        ws = [1.0 / (s * s) for _, s in est]
        self.psi = math.atan2(sum(w * math.sin(p) for (p, _), w in zip(est, ws)),
                              sum(w * math.cos(p) for (p, _), w in zip(est, ws)))
        self.psi_sigma = 1.0 / math.sqrt(sum(ws))
        return self.aligned

    def fuse_fix(self, t, e, n, ox, oy, sigma_m):
        """One robot fix into the offset filter. Returns 'init' | 'ok' | 'jump' | 'no-north'."""
        if self.psi is None:
            return "no-north"
        c, s = math.cos(self.psi), math.sin(self.psi)
        z = (ox - (c * e - s * n), oy - (s * e + c * n))
        r = sigma_m * sigma_m
        if self.b is None:
            self.b, self.Pb, self.b_t = list(z), [r, r], t
            return "init"
        q = (self.drift * max(0.0, t - self.b_t)) ** 2 + 0.05     # odometry drift since the last fix
        self.b_t = t
        P = [self.Pb[0] + q, self.Pb[1] + q]
        d2 = sum((z[k] - self.b[k]) ** 2 / (P[k] + r) for k in range(2))
        if d2 > 11.8:                                # chi2(2) 99.7 %: a GPS jump, not a move
            self.Pb = P
            self.rejected += 1
            if self.rejected >= 8 and self.rejected > 2 * self.accepted:   # everything "jumps":
                self.b, self.Pb = list(z), [r, r]                          # odometry drifted - re-anchor
                self.accepted = self.rejected = 0
                return "init"
            return "jump"
        for k in range(2):
            K = P[k] / (P[k] + r)
            self.b[k] += K * (z[k] - self.b[k])
            P[k] *= (1.0 - K)
        self.Pb = P
        self.accepted += 1
        return "ok"

    @property
    def aligned(self):
        return self.psi is not None and self.psi_sigma <= self.max_sig and self.b is not None

    @property
    def b_sigma(self):
        return math.sqrt(max(self.Pb)) if self.Pb else math.inf

    def to_odom(self, e, n):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return c * e - s * n + self.b[0], s * e + c * n + self.b[1]


class OwnerTrail:
    """The owner's GPS positions (ENU) -> outlier-rejected, moving-average smoothed trail
    (the paper's eq. 10-11: centred average of 2m+1 points, fewer at the ends; m = 3, eq. 12)."""

    def __init__(self, m=3, max_speed=3.0, keep_s=600.0, spacing_m=1.0):
        self.m, self.vmax, self.keep, self.spacing = m, max_speed, keep_s, spacing_m
        self.raw = deque(maxlen=400)          # (t, e, n)
        self.pending = None                   # one suspicious jump, confirmed by the next fix
        self.newest_m = m                     # trailing window for the newest point (node: 1 while walking)

    def add(self, t, e, n):
        if self.raw:
            t0, e0, n0 = self.raw[-1]
            d = math.hypot(e - e0, n - n0)
            if d > self.vmax * max(t - t0, 1.0) + 4.0:
                p = self.pending
                if p is not None and math.hypot(e - p[1], n - p[2]) < 4.0 + self.vmax * (t - p[0]):
                    self.raw.append(p)        # two fixes agree: the owner really is there
                    self.pending = None
                else:
                    self.pending = (t, e, n)
                    return False
        self.raw.append((t, e, n))
        while self.raw and t - self.raw[0][0] > self.keep:
            self.raw.popleft()
        return True

    def smoothed(self):
        R = list(self.raw)
        out = []
        for i in range(len(R)):
            k = min(self.m, i, len(R) - 1 - i) if i < len(R) - 1 else min(self.m, i)
            if i == len(R) - 1:               # newest point: trailing average (no future yet); short
                win = R[max(0, i - self.newest_m):i + 1]   # while the owner walks, or it lags behind
            else:
                win = R[i - k:i + k + 1]
            out.append((R[i][0], sum(p[1] for p in win) / len(win), sum(p[2] for p in win) / len(win)))
        return out

    def path(self):
        S = self.smoothed()
        if not S:
            return []
        keep = [S[0]]
        for p in S[1:]:
            if math.hypot(p[1] - keep[-1][1], p[2] - keep[-1][2]) >= self.spacing:
                keep.append(p)
        if keep[-1] is not S[-1]:
            keep.append(S[-1])
        return keep


# ============================================================ SIM7600 modem

def find_at_port(preferred, baud, log=print):
    """The SIM7600 command port: only /dev/serial/by-id SimTech ports are probed (never the lidar)."""
    import glob
    import os
    import serial
    sim = []
    for p in sorted(glob.glob("/dev/serial/by-id/*")):
        n = os.path.basename(p).lower()
        if "simtech" in n or "sim7600" in n or "1e0e" in n or "qualcomm" in n:
            rp = os.path.realpath(p)
            if rp not in sim:
                sim.append(rp)
    cands = []
    if preferred and str(preferred).lower() != "auto":
        rp = os.path.realpath(preferred)
        if not sim or rp in sim:
            cands.append(rp)
        else:
            log(f"{preferred} is not a SIM7600 port (by-id) - not probed, it may be the lidar")
    cands += [p for p in sim if p not in cands]
    tried = []
    for p in cands:
        try:
            s = serial.Serial(p, baud, timeout=0.3, write_timeout=1)
        except Exception as exc:          # noqa: BLE001
            tried.append(f"{p}: {exc}")
            continue
        try:
            s.reset_input_buffer()
            r = ""
            for _ in range(2):
                s.write(b"AT\r")
                time.sleep(0.4)
                r = s.read(512).decode("utf-8", errors="ignore")
                if "OK" in r:
                    return p
            tried.append(f"{p}: " + ((" ".join(r.split()))[:24] or "no answer"))
        finally:
            s.close()
    if not cands:
        raise RuntimeError("no SIM7600 in /dev/serial/by-id - is it plugged in?")
    raise RuntimeError("no SIM7600 port answered AT (" + "; ".join(tried) + ")")


class Sim7600:
    def __init__(self, port, baud, sim_pin, apn, firebase_url, log=print, http_timeout=12.0):
        import serial
        self.port = find_at_port(port, baud, log)
        self.ser = serial.Serial(self.port, baud, timeout=0.2, write_timeout=2)
        self.lock = threading.Lock()
        self.sim_pin, self.apn, self.url = sim_pin, apn, firebase_url
        self.log, self.http_timeout = log, http_timeout
        self.http_ok, self.cgnss = False, True
        self.gnss = self.net = self.last_gps_raw = self.last_http = ""
        time.sleep(1.0)

    def at(self, cmd, timeout=2.0, expected=None):
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + "\r").encode("ascii"))
            self.ser.flush()
            t0, resp = time.monotonic(), ""
            while time.monotonic() - t0 < timeout:
                k = self.ser.in_waiting
                if k:
                    resp += self.ser.read(k).decode("utf-8", errors="ignore")
                    if expected and expected in resp:
                        break
                    if "\r\nERROR\r\n" in resp or "+CME ERROR:" in resp:
                        break
                    if not expected and "\r\nOK\r\n" in resp:
                        break
                time.sleep(0.02)
            return resp

    def init(self, use_http=True):
        if "OK" not in self.at("AT") + self.at("AT"):
            raise RuntimeError(f"{self.port} does not answer AT")
        self.at("ATE0")
        self.at("AT+CMEE=2")
        if "SIM PIN" in self.at("AT+CPIN?") and self.sim_pin:
            self.at('AT+CPIN="{}"'.format(self.sim_pin), timeout=5)
            time.sleep(2)
        if "+CGPS: 1" not in self.at("AT+CGPS?"):
            self.log("GNSS start: " + " ".join(self.at("AT+CGPS=1", timeout=3).split()))
        self.gnss = " ".join(self.at("AT+CGPS?").split())[:40]
        if not use_http:
            return
        self.at("AT+CNMP=2")
        self.at('AT+CGDCONT=1,"IP","{}"'.format(self.apn))
        for _ in range(30):
            if re.search(r"\+C(E)?REG:\s*\d+,[15]", self.at("AT+CREG?", 1) + self.at("AT+CEREG?", 1)):
                break
            time.sleep(1)
        self.net = " ".join((self.at("AT+CPIN?") + self.at("AT+CSQ") + self.at("AT+CREG?")).split())\
            .replace(" OK", "")[:90]
        if not re.search(r"\+CGACT:\s*1,1", self.at("AT+CGACT?")):
            self.at("AT+CGACT=1,1", timeout=8)
        self.at("AT+HTTPTERM")
        if "ERROR" in self.at("AT+HTTPINIT", timeout=3):
            self.last_http = "HTTPINIT failed"
            return
        self.at('AT+HTTPPARA="CID",1')
        self.http_ok = "ERROR" not in self.at('AT+HTTPPARA="URL","{}"'.format(self.url), timeout=3)
        self.last_http = "ready" if self.http_ok else "URL rejected"

    def robot_fix(self):
        """(lat, lon, speed, course, hdop, sats) - AT+CGNSSINFO (with quality), else AT+CGPSINFO."""
        if self.cgnss:
            r = self.at("AT+CGNSSINFO", timeout=1.5)
            if "ERROR" in r:
                self.cgnss = False                    # firmware without CGNSSINFO
            else:
                self.last_gps_raw = " ".join(r.split())[:110]
                return parse_cgnssinfo(r)
        r = self.at("AT+CGPSINFO", timeout=1.5)
        self.last_gps_raw = " ".join(r.split())[:110]
        return parse_cgpsinfo_full(r)

    def firebase(self):
        if not self.http_ok:
            return None
        r = self.at("AT+HTTPACTION=0", timeout=self.http_timeout, expected="+HTTPACTION:")
        m = re.search(r"\+HTTPACTION:\s*0,(\d+),(\d+)", r)
        self.last_http = (f"HTTP {m.group(1)}, {m.group(2)} bytes" if m
                          else "no reply: " + " ".join(r.split())[:50])
        if not m or int(m.group(1)) != 200 or int(m.group(2)) <= 0:
            return None
        r = self.at("AT+HTTPREAD=0,{}".format(int(m.group(2))), timeout=5)
        mk = re.search(r"\+HTTPREAD:\s*DATA,\d+\s*\r?\n", r)
        if not mk:
            return None
        body = re.split(r"\r?\n\+HTTPREAD:\s*0", r[mk.end():], maxsplit=1)[0].strip()
        return parse_firebase(body)

    def close(self):
        try:
            self.at("AT+HTTPTERM", timeout=2)
            self.at("AT+CGPS=0", timeout=2)
        except Exception:                  # noqa: BLE001
            pass
        self.ser.close()


# ============================================================ ROS node

def main():  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import NavSatFix
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    def yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    class GpsOwner(Node):
        def __init__(self):
            super().__init__("gps_owner")
            p = self.declare_parameter
            p("mode", "indoor")
            p("port", "auto")
            p("baud", 115200)
            p("sim_pin", "")  # set via params.yaml / env, never commit
            p("apn", "online")
            p("firebase_url", "https://YOUR-FIREBASE-PROJECT-default-rtdb.REGION.firebasedatabase.app/"
                              "live_location.json")
            p("firebase_enabled", True)
            p("firebase_period_s", 3.0)
            p("firebase_latency_s", 1.0)
            p("gps_period_s", 1.0)
            p("robot_sigma_m", 5.0)                   # used only when the modem gives no HDOP
            p("firebase_sigma_m", 6.0)                # phone (fused GNSS/Wi-Fi), no accuracy field in the app
            p("owner_max_age_s", 15.0)
            p("min_baseline_m", 6.0)
            p("smooth_m", 3)                          # the paper's m (eq. 12: m = D*Ns/v)
            p("odom_frame", "odom")
            p("base_frame", "base_link")
            p("phone_fresh_s", 5.0)                   # kept for old params files (unused)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.g = g
            self.mode = "outdoor" if str(g("mode")).strip().lower() == "outdoor" else "indoor"
            self.odom, self.base = g("odom_frame"), g("base_frame")
            self.al = GpsAligner(min_base_m=float(g("min_baseline_m")))
            self.trail = OwnerTrail(m=int(g("smooth_m")))
            self.ref = None
            self.robot = None                          # (t, lat, lon, sigma|None, verdict)
            self.owner = None                          # (t, lat, lon, sigma, "firebase")
            self.owner_vel = (0.0, None)
            self.err = deque(maxlen=40)                # |GPS estimate - seen owner| (m), measured
            self._fb_ts, self._fb_same, self._rf_prev = None, 0, None
            self.modem, self.modem_err = None, ""
            self.lock = threading.Lock()
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self._PoseStamped, self._Path = PoseStamped, Path
            self.pub_mode = self.create_publisher(String, "/a2r/mode", latched)
            self.pub_rfix = self.create_publisher(NavSatFix, "/gps/robot_fix", 10)
            self.pub_ofix = self.create_publisher(NavSatFix, "/gps/owner_fix", 10)
            self.pub_est = self.create_publisher(PointStamped, "/gps/owner_estimate", 10)
            self.pub_path = self.create_publisher(Path, "/gps/owner_path", 10)
            self.pub_stat = self.create_publisher(String, "/gps/status", 10)
            self.create_subscription(String, "/a2r/mode_request", self._on_mode_req, 10)
            self.create_subscription(PointStamped, "/follow/owner", self._on_owner_seen, 10)
            self.stop = threading.Event()
            threading.Thread(target=self._modem_loop, daemon=True).start()
            self.create_timer(0.5, self._tick)
            self.create_timer(1.0, self._status)
            self._publish_mode()
            self.get_logger().info(f"gps_owner (a2r_step3f) up - mode {self.mode.upper()}")

        # ---------------------------------------------------------------- mode / frames
        def _publish_mode(self):
            self.pub_mode.publish(String(data=self.mode))

        def _on_mode_req(self, msg):
            m = str(msg.data).strip().lower()
            if m in ("indoor", "outdoor") and m != self.mode:
                self.mode = m
                self.get_logger().info(f"mode -> {m.upper()} (phone page)")
            self._publish_mode()

        def _odom_base(self):
            try:
                t = self.tf.lookup_transform(self.odom, self.base, rclpy.time.Time())
            except Exception:                  # noqa: BLE001
                return None
            return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

        def _enu(self, lat, lon):
            if self.ref is None:
                self.ref = (lat, lon)
            return ll_to_enu(lat, lon, *self.ref)

        # ---------------------------------------------------------------- owner seen: measure the real error
        def _on_owner_seen(self, msg):
            if self.mode != "outdoor" or not self.al.aligned:
                return
            est = getattr(self, "_last_est", None)
            T = self._odom_base()
            if est is None or T is None or time.monotonic() - est[2] > 1.0:
                return
            c, s = math.cos(T[2]), math.sin(T[2])
            ox, oy = T[0] + c * msg.point.x - s * msg.point.y, T[1] + s * msg.point.x + c * msg.point.y
            now = time.monotonic()
            if not self.err or now - self.err[-1][0] > 1.0:          # at most one sample per second
                self.err.append((now, math.hypot(est[0] - ox, est[1] - oy)))

        def _measured_sigma(self):
            E = [e for t, e in self.err if time.monotonic() - t < 300.0]
            return math.sqrt(sum(e * e for e in E) / len(E)) if len(E) >= 5 else None

        # ---------------------------------------------------------------- modem thread
        def _modem_loop(self):
            last_gps = last_fb = 0.0
            while not self.stop.is_set():
                if self.mode != "outdoor":
                    if self.modem is not None:
                        self.modem.close()
                        self.modem = None
                        self.get_logger().info("indoor: modem closed")
                    self.stop.wait(0.5)
                    continue
                if self.modem is None:
                    try:
                        self.modem = Sim7600(self.g("port"), int(self.g("baud")), str(self.g("sim_pin")),
                                             str(self.g("apn")), str(self.g("firebase_url")),
                                             log=self.get_logger().info)
                        self.modem.init(use_http=bool(self.g("firebase_enabled")))
                        self.modem_err = ""
                        self.get_logger().info(f"SIM7600 ready on {self.modem.port} (Firebase "
                                               f"{'on' if self.modem.http_ok else 'off'})")
                    except Exception as exc:   # noqa: BLE001
                        self.modem, self.modem_err = None, f"{type(exc).__name__}: {exc}"
                        self.get_logger().warn(f"SIM7600 not available ({self.modem_err}) - retry in 10 s",
                                               throttle_duration_sec=30.0)
                        self.stop.wait(10.0)
                        continue
                now = time.monotonic()
                try:
                    if now - last_gps >= float(self.g("gps_period_s")):
                        last_gps = now
                        fix = self.modem.robot_fix()
                        if fix is not None:
                            self._on_robot_fix(*fix)
                    if self.modem.http_ok and now - last_fb >= float(self.g("firebase_period_s")):
                        last_fb = now
                        r = self.modem.firebase()
                        if r is not None:
                            self._on_firebase(r)
                except Exception as exc:        # noqa: BLE001
                    if self.stop.is_set():
                        break
                    self.get_logger().warn(f"modem error ({exc}) - reopening", throttle_duration_sec=10.0)
                    try:
                        self.modem.close()
                    except Exception:           # noqa: BLE001
                        pass
                    self.modem = None
                self.stop.wait(0.1)

        def _on_firebase(self, r):
            same = r[3] is not None and r[3] == self._fb_ts
            self._fb_ts = r[3]
            self._fb_same = self._fb_same + 1 if same else 0
            sig = r[2] if r[2] else float(self.g("firebase_sigma_m"))
            with self.lock:
                t_rx = self.owner[0] if (same and self.owner is not None) else time.monotonic()
                self.owner = (t_rx, r[0], r[1], sig, "firebase")
                self.owner_vel = (r[4], r[5])
            if not same:
                e, n = self._enu(r[0], r[1])
                if not self.trail.add(t_rx, e, n):
                    self.get_logger().info("owner GPS jump ignored (needs a second fix to confirm)",
                                           throttle_duration_sec=10.0)
            if self._fb_same == 5:
                self.get_logger().warn("Firebase live_location has not changed for 5 reads - "
                                       "is the owner's phone still uploading?")

        def _on_robot_fix(self, lat, lon, spd=None, course=None, hdop=None, sats=None):
            now = time.monotonic()
            sig = robot_sigma(hdop, sats, float(self.g("robot_sigma_m")))
            T = self._odom_base()
            verdict = "poor fix (HDOP/satellites) - not used" if sig is None else "no odometry"
            prev, self._rf_prev = self._rf_prev, (now, T)
            if sig is not None and T is not None:
                e, n = self._enu(lat, lon)
                # north from the GNSS course - only while driving straight forward AND the GNSS speed
                # matches the odometry speed (multipath makes up speeds while the car crawls)
                if prev is not None and prev[1] is not None and spd is not None and course is not None \
                        and 0.0 < now - prev[0] < 2.5 and sig <= 6.0:
                    dx, dy = T[0] - prev[1][0], T[1] - prev[1][1]
                    v_odo = math.hypot(dx, dy) / (now - prev[0])
                    fwd = dx * math.cos(T[2]) + dy * math.sin(T[2])
                    turn = abs(_wrap(T[2] - prev[1][2])) / (now - prev[0])
                    if fwd > 0.0 and turn < 0.3 and v_odo >= 0.4 and abs(spd - v_odo) < 0.35:
                        self.al.add_heading(now, T[2] - (math.pi / 2.0 - math.radians(course)),
                                            min(1.0, v_odo * v_odo))
                verdict = self.al.fuse_fix(now, e, n, T[0], T[1], sig)
                if verdict != "jump":
                    self.al.add(now, e, n, T[0], T[1], sig, "robot")
            self.robot = (now, lat, lon, sig, verdict, hdop, sats)

        # ---------------------------------------------------------------- outputs
        def _fix_msg(self, lat, lon, sig, frame):
            m = NavSatFix()
            m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), frame
            m.latitude, m.longitude = float(lat), float(lon)
            m.position_covariance = [sig * sig, 0.0, 0.0, 0.0, sig * sig, 0.0, 0.0, 0.0, 1e6]
            m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
            return m

        def _tick(self):
            if self.mode != "outdoor":
                return
            now = time.monotonic()
            self.al.update(now)
            r = self.robot
            if r is not None and now - r[0] < 5.0:
                self.pub_rfix.publish(self._fix_msg(r[1], r[2], r[3] or 99.0, "gps_robot"))
            with self.lock:
                o = self.owner
            if o is None or now - o[0] > float(self.g("owner_max_age_s")):
                return
            self.pub_ofix.publish(self._fix_msg(o[1], o[2], o[3], "gps_owner"))
            if not self.al.aligned or self.ref is None:
                return
            self.trail.newest_m = 1 if self.owner_vel[0] >= 0.3 else int(self.g("smooth_m"))
            trail = self.trail.path()
            if not trail:
                return
            _, e, n = trail[-1]                       # newest SMOOTHED owner position
            spd, hdg = self.owner_vel
            pred = spd >= 0.3 and hdg is not None
            if pred:
                dt = min(now - o[0] + float(self.g("firebase_latency_s")), 4.0)
                e += spd * dt * math.sin(math.radians(hdg))
                n += spd * dt * math.cos(math.radians(hdg))
            x, y = self.al.to_odom(e, n)
            T = self._odom_base()
            dist = math.hypot(x - T[0], y - T[1]) if T is not None else 0.0
            age = now - o[0]
            rs = self.robot[3] if self.robot and self.robot[3] else float(self.g("robot_sigma_m"))
            b_sig = max(self.al.b_sigma, 0.6 * rs)     # GPS errors near buildings are correlated (bias)
            sig = math.sqrt(o[3] ** 2 / 2.0 + b_sig ** 2                # /2: averaged over m+1 fixes
                            + (dist * self.al.psi_sigma) ** 2 + ((0.5 if pred else 1.4) * age) ** 2)
            ms = self._measured_sigma()
            if ms is not None:
                sig = max(sig, ms)                     # what GPS really achieved here (vs lidar/camera)
            self._last_est = (x, y, now)
            stamp = self.get_clock().now().to_msg()
            m = PointStamped()
            m.header.stamp, m.header.frame_id = stamp, self.odom
            m.point.x, m.point.y, m.point.z = float(x), float(y), float(sig)
            self.pub_est.publish(m)
            path = self._Path()                        # the global path: the owner's smoothed trail
            path.header.stamp, path.header.frame_id = stamp, self.odom
            for _, pe, pn in trail[-60:]:
                px, py = self.al.to_odom(pe, pn)
                ps = self._PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(px), float(py)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            ps = self._PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
            self.pub_path.publish(path)

        def _status(self):
            now = time.monotonic()
            with self.lock:
                o = self.owner
            r = self.robot
            st = {"mode": self.mode, "modem": "ok" if self.modem is not None else (self.modem_err or "off"),
                  "robot_fix_age_s": round(now - r[0], 1) if r else None,
                  "robot_sigma_m": None if not r or r[3] is None else round(r[3], 1),
                  "robot_fix": r[4] if r else None, "hdop": r[5] if r else None, "sats": r[6] if r else None,
                  "jumps_rejected": self.al.rejected,
                  "owner_age_s": round(now - o[0], 1) if o else None,
                  "aligned": self.al.aligned, "pairs": self.al.npairs,
                  "north_in_odom_deg": None if self.al.psi is None else round(math.degrees(self.al.psi), 1),
                  "north_sigma_deg": None if self.al.psi is None else round(math.degrees(self.al.psi_sigma), 1),
                  "north_samples": len(self.al.headings),
                  "measured_error_m": None if self._measured_sigma() is None else round(self._measured_sigma(), 1),
                  "trail_points": len(self.trail.raw),
                  "owner_lat": o[1] if o else None, "owner_lon": o[2] if o else None,
                  "owner_speed_mps": round(self.owner_vel[0], 2), "owner_heading_deg": self.owner_vel[1]}
            md = self.modem
            if md is not None:
                st.update(port=md.port, sim_net=md.net, gnss=md.gnss, gps_raw=md.last_gps_raw,
                          firebase=md.last_http)
            if r and o:
                st["owner_dist_gps_m"] = round(haversine(r[1], r[2], o[1], o[2]), 1)   # raw, unfiltered
            est = getattr(self, "_last_est", None)
            T = self._odom_base()
            if est is not None and T is not None and now - est[2] < 2.0:
                st["owner_dist_m"] = round(math.hypot(est[0] - T[0], est[1] - T[1]), 1)    # filtered
            self.pub_stat.publish(String(data=json.dumps(st)))
            if self.mode == "outdoor":
                self.get_logger().info("gps: " + json.dumps(st), throttle_duration_sec=15.0)
            self._publish_mode()

        def destroy_node(self):
            self.stop.set()
            m = self.modem
            if m is not None:
                if m.lock.acquire(timeout=0.5):
                    m.lock.release()
                    m.close()
                else:
                    m.ser.close()
            return super().destroy_node()

    rclpy.init()
    node = GpsOwner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
