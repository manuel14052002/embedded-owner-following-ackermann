#!/usr/bin/env python3
"""phone_pdr_node - track the owner with THEIR PHONE's motion sensors while the robot cannot see them.

Works with any phone, no app: the owner opens  https://<jetson-ip>:8443/  in the phone's browser
(Android Chrome / Firefox, iPhone Safari), taps Start and keeps the page open. The page counts
steps (accelerometer) and integrates turning (gyroscope about the gravity axis, so the phone may be
in a pocket in any orientation; compass fallback for phones without a gyroscope) and sends
{steps, yaw} 10 times a second.

The robot aligns the phone's walk with what its camera sees:
  * while FollowMe detects the owner, pairs (phone track point, owner position in odom) are
    collected; a 2-D similarity fit (rotation, step-length scale, translation - and a mirror flag,
    because browsers disagree on sign conventions) maps the phone track into the robot's odom frame;
  * a camera detection that does not fit the phone's walk (a person crossing the owner's path taken
    for the owner) is not used for the alignment: it is compared with where the phone says the owner
    is, anchored on detections 0.4-2 s older (before the crossing). After reset_s of disagreement
    the alignment is dropped and rebuilt (then the phone was the one that was wrong).
  * when the owner is lost (/followme/owner_lost, or no owner detection for lost_after_s), the
    owner's position is predicted as
        last seen position + fitted transform x (phone displacement since then)
    and published to the Pi's search on /phone_owner/last_seen (base_link). The Pi's follow_manager
    then drives toward where the phone says the owner went, instead of the last spot seen.
  * when the phone estimate is not usable (no alignment yet, phone silent, too long out of sight),
    FollowMe's own /followme/owner_last_seen is forwarded unchanged.

Topics
  in : /followme/owner_position (PointStamped, camera frame, real detections only)
       /followme/owner_lost (Bool), /followme/owner_last_seen (PointStamped, base_link)
  out: /phone_owner/last_seen (PointStamped, base_link) -> Pi follow_manager last_seen_topic
       /phone_owner/estimate (PointStamped, odom, only while lost), /phone_owner/status (String, JSON)
       /phone_owner/view (PointStamped, odom, z = sigma m) - always, for the web view minimap
"""

from __future__ import annotations

import json
import math
import os
import ssl
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

# ============================================================================== core (no ROS)


def fit_similarity_2d(P: np.ndarray, Q: np.ndarray, allow_mirror: bool = False):
    """Least-squares Q ~ s*R*P + t. R is a proper rotation unless allow_mirror (then rotation OR
    reflection, whichever fits). A mostly straight walk fits its own mirror image equally well, so a
    free reflection picks the wrong side half the time - only allow it for a known-mirrored phone.
    Returns (s, R, t, rms, mirrored)."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - mp, Q - mq
    var = float(np.sum(Pc ** 2))
    if var < 1e-9:
        raise ValueError("phone track has no extent")
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ U.T                                  # best orthogonal map (Kabsch/Umeyama)
    if not allow_mirror and np.linalg.det(R) < 0:   # best PROPER rotation (Umeyama det fix)
        D = np.diag([1.0, -1.0])
        R = Vt.T @ D @ U.T
        S = S * np.array([1.0, -1.0])
    s = float(np.sum(S)) / var
    t = mq - s * R @ mp
    res = Q - (s * (R @ P.T).T + t)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return s, R, t, rms, bool(np.linalg.det(R) < 0)


class PhoneTracker:
    """Phone dead reckoning aligned to the robot's odom frame by the camera's owner detections."""

    def __init__(self, step_len_m: float = 0.70, window_s: float = 30.0, min_pairs: int = 8,
                 min_extent_m: float = 1.0, max_rms_m: float = 0.45, scale_range=(0.45, 1.7),
                 max_lost_s: float = 45.0, max_walk_m: float = 25.0, phone_timeout_s: float = 1.5,
                 history_s: float = 180.0, allow_mirror: bool = False, vision_latency_s: float = 0.3,
                 gate_min_m: float = 1.0, reset_s: float = 6.0) -> None:
        self.L0, self.window_s, self.min_pairs = step_len_m, window_s, min_pairs
        self.min_extent, self.max_rms, self.scale_range = min_extent_m, max_rms_m, scale_range
        self.max_lost_s, self.max_walk, self.phone_timeout = max_lost_s, max_walk_m, phone_timeout_s
        self.track: deque = deque()                 # (t, px, py, cumulative walk m)
        self.vision: deque = deque()                # (t, x_odom, y_odom)
        self.history_s = history_s
        self.allow_mirror = allow_mirror            # True only for a phone whose turn sign is inverted
        self.latency = vision_latency_s             # camera detection age when it arrives (image + net)
        self.steps = None
        self.px = self.py = self.walk = 0.0
        self.phone_t = -1e9
        self.fit = None                             # dict(s, R, t, rms, mirrored, n, t_fit)
        self.gate_min, self.reset_s = gate_min_m, reset_s
        self.rejected = self.resets = 0             # camera detections not used / alignments dropped
        self._rej_since, self._rej_n = None, 0
        self.lock = threading.Lock()

    # -- inputs -------------------------------------------------------------------------
    def on_phone(self, t: float, steps: int, yaw: float) -> None:
        with self.lock:
            if self.steps is None or steps < self.steps:      # first sample / page reloaded
                self.steps = steps
            n = min(int(steps - self.steps), 10)              # cap: a glitch must not teleport
            for _ in range(max(0, n)):
                self.px += self.L0 * math.cos(yaw)
                self.py += self.L0 * math.sin(yaw)
                self.walk += self.L0
            self.steps = steps
            self.phone_t = t
            self.track.append((t, self.px, self.py, self.walk))
            while self.track and t - self.track[0][0] > self.history_s:
                self.track.popleft()

    def on_vision(self, t: float, x: float, y: float) -> bool:
        """One camera detection of 'the owner' (odom). False = it does not fit the phone's walk
        and was not used."""
        with self.lock:
            tv = t - self.latency                          # when the camera actually saw the owner
            if not self._vision_ok(tv, x, y):
                return False
            self.vision.append((tv, x, y))
            while self.vision and t - self.vision[0][0] > max(self.window_s, 5.0):
                self.vision.popleft()
            return True

    def _anchor_locked(self, sR: np.ndarray, t_hi: float, span: float) -> Optional[np.ndarray]:
        """Mean offset (detection - fitted phone point) of the detections in [t_hi - span, t_hi]."""
        offs = []
        for (t, x, y) in reversed(self.vision):
            if t > t_hi:
                continue
            if t_hi - t > span:
                break
            p = self._phone_at(t)
            if p is not None:
                offs.append(np.array([x, y]) - sR @ np.array(p[:2]))
        return np.mean(offs, axis=0) if offs else None

    def _vision_ok(self, tv: float, x: float, y: float) -> bool:
        f = self.fit
        if f is None or not self.vision or self.phone_t < tv - self.phone_timeout:
            self._rej_since = None
            return True                                    # nothing to judge with
        sR = f["s"] * f["R"]
        t_last = self.vision[-1][0]
        # judged against where the owner was BEFORE (0.4-2 s ago): a crossing person cannot drag it
        a, t_a = self._anchor_locked(sR, tv - 0.4, 1.6), tv - 0.4
        if a is None:
            a, t_a = self._anchor_locked(sR, t_last, 1.0), t_last
        p, pa = self._phone_at(tv), self._phone_at(min(t_a, t_last))
        if a is None or p is None:
            self._rej_since = None
            return True
        walked = f["s"] * max(0.0, p[2] - pa[2]) if pa is not None else 0.0
        pred = sR @ np.array(p[:2]) + a
        gate = max(self.gate_min, 2.5 * f["rms"] + 0.1 * walked)
        if math.hypot(x - pred[0], y - pred[1]) <= gate:
            self._rej_since = None
            return True
        self.rejected += 1
        if self._rej_since is None:
            self._rej_since, self._rej_n = tv, 0
        self._rej_n += 1
        if tv - self._rej_since < self.reset_s or self._rej_n < 10:
            return False
        self.fit, self._rej_since = None, None             # long disagreement: realign from scratch
        self.vision.clear()
        self.resets += 1
        return True

    # -- helpers ------------------------------------------------------------------------
    def _phone_at(self, t: float) -> Optional[tuple]:
        tr = self.track
        if not tr or t < tr[0][0] - 0.2:
            return None
        if t >= tr[-1][0]:
            return tr[-1][1], tr[-1][2], tr[-1][3]
        lo, hi = 0, len(tr) - 1
        while hi - lo > 1:                                     # binary search on time
            mid = (lo + hi) // 2
            if tr[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        (t0, x0, y0, w0), (t1, x1, y1, w1) = tr[lo], tr[hi]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return x0 + a * (x1 - x0), y0 + a * (y1 - y0), w0 + a * (w1 - w0)

    def update_fit(self, now: float) -> Optional[dict]:
        """Refit from the last window_s of pairs; keeps the previous fit if this one is not good."""
        with self.lock:
            P, Q = [], []
            for (t, x, y) in self.vision:
                if now - t > self.window_s:
                    continue
                p = self._phone_at(t)
                if p is not None:
                    P.append(p[:2])
                    Q.append((x, y))
            if len(P) < self.min_pairs:
                return self.fit
            P, Q = np.array(P), np.array(Q)
            if np.max(np.linalg.norm(Q - Q.mean(axis=0), axis=1)) < self.min_extent or \
                    np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)) < 0.5 * self.min_extent:
                return self.fit                                # owner has not walked enough yet
            try:
                s, R, t, rms, mirrored = fit_similarity_2d(P, Q, False)
                # a phone that reports turns inverted fits only its mirror image - accept that ONLY when
                # the walk had real turns (a straight walk fits both equally) and it is clearly better
                minor = float(np.linalg.svd(P - P.mean(axis=0), compute_uv=False)[-1]) / np.sqrt(len(P))
                if self.allow_mirror or minor >= 0.5:
                    m_fit = fit_similarity_2d(P, Q, True)
                    if m_fit[4] and (self.allow_mirror or m_fit[3] < 0.6 * rms):
                        s, R, t, rms, mirrored = m_fit
            except ValueError:
                return self.fit
            if rms <= self.max_rms and self.scale_range[0] <= s <= self.scale_range[1]:
                self.fit = dict(s=s, R=R, t=t, rms=rms, mirrored=mirrored, n=len(P), t_fit=now)
            return self.fit

    def predict(self, now: float) -> Optional[dict]:
        """Owner position (odom) from the phone, anchored at the last camera detections."""
        with self.lock:
            f = self.fit
            if f is None or not self.vision or now - self.phone_t > self.phone_timeout:
                return None
            t_last = self.vision[-1][0]
            if now - t_last > self.max_lost_s:
                return None
            # anchor: mean offset of the last ~1 s of detections (robust to one noisy detection)
            sR = f["s"] * f["R"]
            offs = []
            for (t, x, y) in list(self.vision)[-15:]:
                if t_last - t > 1.0:
                    continue
                p = self._phone_at(t)
                if p is not None:
                    offs.append(np.array([x, y]) - sR @ np.array(p[:2]))
            if not offs:
                return None
            anchor = np.mean(offs, axis=0)
            p_now, p_then = self._phone_at(now), self._phone_at(t_last)
            if p_now is None or p_then is None:
                return None
            walked = p_now[2] - p_then[2]
            if walked > self.max_walk:
                return None
            est = sR @ np.array(p_now[:2]) + anchor
            return dict(x=float(est[0]), y=float(est[1]), walked=float(f["s"] * walked),
                        since=float(now - t_last), sigma=float(f["rms"] + 0.06 * f["s"] * walked))

    def status(self, now: float) -> dict:
        f = self.fit
        return dict(phone=bool(now - self.phone_t <= self.phone_timeout),
                    steps=int(self.steps or 0),
                    aligned=f is not None,
                    rejected=self.rejected, resets=self.resets,
                    rms=round(f["rms"], 2) if f else None,
                    step_len=round(f["s"] * self.L0, 2) if f else None,
                    mirrored=f["mirrored"] if f else None)


# ============================================================================== phone page

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A2RBOT phone tracking</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee;text-align:center}
button{font-size:1.4em;padding:.8em 1.6em;margin:1.2em;border-radius:12px;border:0;background:#2e7d32;color:#fff}
#s{font-size:1.1em;margin:.6em}.big{font-size:2.4em;margin:.2em}.dim{color:#999;font-size:.9em;padding:0 1em}
#arrow{display:inline-block;font-size:3em;transition:transform .1s}
.m{font-size:1em;padding:.5em 1.2em;margin:.4em;background:#444}.m.on{background:#2e7d32}
</style></head><body>
<h2>A2RBOT phone tracking</h2>
<div>Mode: <b id="mode">?</b><br><button id="indoor" class="m">Indoor</button><button id="outdoor" class="m">Outdoor</button></div>
<button id="go">Start step count</button>
<!-- a2r_step3c: no phone GPS here - the owner's phone app writes to Firebase, the robot's SIM7600 reads it -->
<div id="s">Tap Start, then keep this page open and the phone with you.</div>
<div class="big"><span id="steps">0</span> steps</div>
<div id="arrow">&#8593;</div>
<div id="robot" class="dim"></div>
<p class="dim">The robot uses your steps and turns only while it cannot see you.
Keep the screen on (the page asks the phone to); a locked screen stops the sensors.</p>
<script>
/* ---- pure step + heading tracker (also run by the robot's tests in Node) ---- */
function makePdr(){
  var st={steps:0,yaw:0,mode:"gyro",gx:0,gy:0,gz:9.81,base:9.81,lp:9.81,high:false,last:-1e9,t:null,haveGyro:false};
  st.onMotion=function(ax,ay,az,ra,rb,rg,tms){
    if(st.t===null){st.t=tms;st.gx=ax;st.gy=ay;st.gz=az;}
    var dt=Math.min(Math.max((tms-st.t)/1000,0),0.2); st.t=tms;
    st.gx+=0.05*(ax-st.gx); st.gy+=0.05*(ay-st.gy); st.gz+=0.05*(az-st.gz);
    var gn=Math.sqrt(st.gx*st.gx+st.gy*st.gy+st.gz*st.gz)||1;
    if(ra!==null&&rb!==null&&rg!==null){            /* gyro (deg/s) about the gravity axis */
      st.haveGyro=true;
      var rate=(rb*st.gx+rg*st.gy+ra*st.gz)/gn;
      st.yaw+=rate*Math.PI/180*dt;
    }
    var m=Math.sqrt(ax*ax+ay*ay+az*az);
    st.lp+=0.25*(m-st.lp); st.base+=0.01*(m-st.base);
    if(!st.high&&st.lp>st.base+0.9&&tms-st.last>280){st.high=true;}
    else if(st.high&&st.lp<st.base-0.4){st.high=false;st.steps++;st.last=tms;}
  };
  st.onCompass=function(headingDeg){            /* phones without a gyroscope */
    if(!st.haveGyro&&headingDeg!==null){st.mode="compass";st.yaw=-headingDeg*Math.PI/180;}
  };
  return st;
}
if(typeof module!=="undefined"){module.exports={makePdr:makePdr};}
/* ---- page wiring ---- */
if(typeof document!=="undefined"){
  var pdr=makePdr(), sending=false, ok=0;
  /* a2r_step3c: indoor/outdoor switch only. The owner's position comes from Firebase through the
     robot's SIM7600 - this page sends no GPS. */
  var curMode="";
  function showMode(m){
    if(!m)return; curMode=m;
    document.getElementById("mode").textContent=m.toUpperCase();
    document.getElementById("indoor").className="m"+(m==="indoor"?" on":"");
    document.getElementById("outdoor").className="m"+(m==="outdoor"?" on":"");
  }
  function setMode(m){
    fetch("/mode",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:m})})
     .then(function(r){return r.json();}).then(function(j){showMode(j.mode);}).catch(function(){});
  }
  document.getElementById("indoor").onclick=function(){setMode("indoor");};
  document.getElementById("outdoor").onclick=function(){setMode("outdoor");};
  setInterval(function(){
    if(sending)return;                              /* /data answers carry the mode then */
    fetch("/mode").then(function(r){return r.json();}).then(function(j){showMode(j.mode);}).catch(function(){});
  },2000);
  fetch("/mode").then(function(r){return r.json();}).then(function(j){showMode(j.mode);}).catch(function(){});
  function onMotion(e){
    var a=e.accelerationIncludingGravity||{}, r=e.rotationRate||{};
    if(a.x===null||a.x===undefined)return;
    pdr.onMotion(a.x,a.y,a.z,
      (r.alpha===undefined)?null:r.alpha,(r.beta===undefined)?null:r.beta,(r.gamma===undefined)?null:r.gamma,
      performance.now());
  }
  function onOrient(e){
    var h=(e.webkitCompassHeading!==undefined)?e.webkitCompassHeading:(e.alpha===null?null:360-e.alpha);
    pdr.onCompass(h);
  }
  async function wake(){try{if(navigator.wakeLock){await navigator.wakeLock.request("screen");}}catch(err){}}
  document.addEventListener("visibilitychange",function(){if(document.visibilityState==="visible")wake();});
  document.getElementById("go").onclick=async function(){
    try{
      if(typeof DeviceMotionEvent!=="undefined"&&typeof DeviceMotionEvent.requestPermission==="function"){
        var p=await DeviceMotionEvent.requestPermission(); if(p!=="granted"){throw new Error("motion permission denied");}
        if(typeof DeviceOrientationEvent.requestPermission==="function"){await DeviceOrientationEvent.requestPermission();}
      }
      window.addEventListener("devicemotion",onMotion);
      window.addEventListener("deviceorientation",onOrient);
      wake(); this.style.display="none"; sending=true;
      document.getElementById("s").textContent="Running - walk normally.";
    }catch(err){document.getElementById("s").textContent="Sensors not available: "+err.message+
      " (open the page over https and allow motion access)";}
  };
  setInterval(function(){
    if(!sending)return;
    var body={steps:pdr.steps,yaw:pdr.yaw,mode:pdr.mode,pdr:true};         /* a2r_step3c */
    fetch("/data",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)})
     .then(function(r){return r.json();}).then(function(j){
       ok=Date.now(); showMode(j.mode);
       document.getElementById("robot").textContent=j.text||"";})
     .catch(function(){});
    document.getElementById("steps").textContent=pdr.steps;
    document.getElementById("arrow").style.transform="rotate("+(-pdr.yaw*180/Math.PI)+"deg)";
    if(Date.now()-ok>3000){document.getElementById("robot").textContent="No connection to the robot";}
  },100);
}
</script></body></html>"""


def status_text(st: dict, lost: bool, using_phone: bool) -> str:
    if not st["phone"]:
        return "Robot: waiting for phone data"
    if using_phone:
        return "Robot: cannot see you - following your phone"
    if lost:
        return "Robot: cannot see you - phone not aligned yet, searching"
    if not st["aligned"]:
        return "Robot: sees you - walk a few metres so it can learn your phone"
    return f"Robot: sees you - phone aligned (step {st['step_len']} m)"


class PhoneServer:
    """HTTPS server: GET / -> page, POST /data -> feeds the tracker, answers with the status text."""

    def __init__(self, tracker: PhoneTracker, port: int, certfile: str, keyfile: str,
                 text_fn=lambda: "", gps_fn=None, mode_fn=lambda: "", mode_set_fn=None,
                 page: str = PAGE) -> None:
        server = self
        self.tracker, self.text_fn, self.posts = tracker, text_fn, 0
        self.gps_fn, self.mode_fn, self.mode_set_fn, self.page = gps_fn, mode_fn, mode_set_fn, page   # a2r_step3

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, server.page.encode(), "text/html; charset=utf-8")
                elif self.path == "/mode":                        # a2r_step3
                    self._send(200, json.dumps({"mode": server.mode_fn()}).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path == "/mode":                          # a2r_step3: indoor / outdoor switch
                    try:
                        n = min(int(self.headers.get("Content-Length", 0)), 256)
                        m = str(json.loads(self.rfile.read(n) or b"{}")["mode"]).strip().lower()
                        if m not in ("indoor", "outdoor"):
                            raise ValueError
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        return self._send(400, b'{"error":"mode must be indoor or outdoor"}', "application/json")
                    if server.mode_set_fn is not None:
                        server.mode_set_fn(m)
                    return self._send(200, json.dumps({"mode": server.mode_fn()}).encode(), "application/json")
                if self.path != "/data":
                    return self._send(404, b"not found", "text/plain")
                try:
                    n = min(int(self.headers.get("Content-Length", 0)), 4096)
                    d = json.loads(self.rfile.read(n) or b"{}")
                    steps, yaw = int(d["steps"]), float(d["yaw"])
                    if not math.isfinite(yaw) or steps < 0:
                        raise ValueError
                    gps = d.get("gps")                           # a2r_step3: optional phone GPS fix
                    if gps is not None:
                        gps = (float(gps["lat"]), float(gps["lon"]), float(gps.get("acc") or 30.0))
                        if not (-90 <= gps[0] <= 90 and -180 <= gps[1] <= 180 and math.isfinite(gps[2])):
                            gps = None
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    return self._send(400, b'{"error":"bad data"}', "application/json")
                if d.get("pdr", True):                          # a2r_step3: GPS-only posts carry no steps
                    server.tracker.on_phone(time.monotonic(), steps, yaw)
                if gps is not None and server.gps_fn is not None:
                    server.gps_fn(*gps)
                server.posts += 1
                self._send(200, json.dumps({"text": server.text_fn(), "mode": server.mode_fn()}).encode(),
                           "application/json")

        class Quiet(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass                                   # phones drop connections constantly

        self.httpd = Quiet(("0.0.0.0", port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


def ensure_cert(cert_dir: str) -> tuple:
    """Self-signed certificate (browsers only give motion sensors to https pages)."""
    cert_dir = os.path.expanduser(cert_dir)
    os.makedirs(cert_dir, exist_ok=True)
    cert, key = os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem")
    if not (os.path.isfile(cert) and os.path.isfile(key)):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
                        "-out", cert, "-days", "3650", "-subj", "/CN=a2rbot"],
                       check=True, capture_output=True)
    return cert, key


# ============================================================================== ROS node


def main() -> None:  # pragma: no cover - needs ROS
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformException, TransformListener

    def tf_matrix(tf) -> np.ndarray:
        q, t = tf.transform.rotation, tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
                         [0, 0, 0, 1.0]])

    class PhonePdrNode(Node):
        def __init__(self) -> None:
            super().__init__("phone_pdr")
            p = self.declare_parameter
            p("port", 8443)
            p("cert_dir", "~/.a2rbot_phone")
            p("odom_frame", "odom")
            p("base_frame", "base_link")
            p("step_length_m", 0.70)
            p("max_lost_s", 45.0)
            p("max_walk_m", 25.0)
            p("allow_mirror", False)      # True = always allow the mirror fit; False = only after clear turns
            p("vision_latency_s", 0.3)    # camera detection delay (image -> /followme/owner_position)
            p("owner_topic", "/followme/owner_position")
            p("lost_topic", "/followme/owner_lost")
            p("lost_after_s", 1.0)        # no owner detection this long = lost (works without /followme/owner_lost)
            p("vision_gate_m", 1.0)       # detection further than max(this, 2.5 rms) from the phone: not used
            p("vision_reset_s", 6.0)      # camera and phone disagree this long: realign
            p("last_seen_in", "/followme/owner_last_seen")
            p("last_seen_out", "/phone_owner/last_seen")
            p("firebase_url", "https://YOUR-FIREBASE-PROJECT-default-rtdb.REGION.firebasedatabase.app/"
                              "live_location.json")      # a2r_step3: the page also writes here (mobile data)
            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.odom, self.base = g("odom_frame"), g("base_frame")
            self.tracker = PhoneTracker(step_len_m=float(g("step_length_m")),
                                        max_lost_s=float(g("max_lost_s")), max_walk_m=float(g("max_walk_m")),
                                        allow_mirror=bool(g("allow_mirror")),
                                        vision_latency_s=float(g("vision_latency_s")),
                                        gate_min_m=float(g("vision_gate_m")), reset_s=float(g("vision_reset_s")))
            self.lost = False
            self.lost_after = float(g("lost_after_s"))
            self.t_owner, self.lost_flag, self.lost_flag_t = -math.inf, False, -math.inf
            self.using_phone = False
            cert, key = ensure_cert(g("cert_dir"))
            # a2r_step3: mode switch + phone GPS -> the Pi's gps_owner node
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from sensor_msgs.msg import NavSatFix
            self._NavSatFix, self.mode, self.mode_req = NavSatFix, "", ""
            self.pub_mode_req = self.create_publisher(String, "/a2r/mode_request", 10)
            self.create_subscription(String, "/a2r/mode", self._on_mode,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.server = PhoneServer(self.tracker, int(g("port")), cert, key, self._text,
                                      mode_fn=lambda: self.mode or self.mode_req or "indoor",   # a2r_step3c: no gps_fn
                                      mode_set_fn=self._set_mode,
                                      page=PAGE.replace("__FIREBASE_URL__", str(g("firebase_url"))))
            self.tf = Buffer()
            self.tfl = TransformListener(self.tf, self, spin_thread=False)
            self.pub_seen = self.create_publisher(PointStamped, g("last_seen_out"), 10)
            self.pub_est = self.create_publisher(PointStamped, "/phone_owner/estimate", 10)
            self.pub_status = self.create_publisher(String, "/phone_owner/status", 10)
            self.pub_view = self.create_publisher(PointStamped, "/phone_owner/view", 10)   # display only
            self.create_subscription(PointStamped, g("owner_topic"), self._on_owner, 10)
            self.create_subscription(Bool, g("lost_topic"), self._on_lost, 10)
            self.create_subscription(PointStamped, g("last_seen_in"), self._on_last_seen, 10)
            self.create_timer(0.5, self._fit_tick)
            self.create_timer(0.2, self._predict_tick)
            self.create_timer(1.0, self._status_tick)
            self.get_logger().info(f"phone page: https://<this-ip>:{self.server.port}/  "
                                   f"(open on the owner's phone, accept the certificate warning, tap Start)")

        def _on_mode(self, msg) -> None:                       # a2r_step3: the Pi's gps_owner decides
            self.mode = str(msg.data).strip().lower()

        def _set_mode(self, m: str) -> None:
            self.mode_req = m
            self.pub_mode_req.publish(String(data=m))
            self.get_logger().info(f"phone page: mode {m} requested")

        def _text(self) -> str:
            return status_text(self.tracker.status(time.monotonic()), self.lost, self.using_phone)

        def _T(self, target: str, source: str) -> Optional[np.ndarray]:
            try:
                return tf_matrix(self.tf.lookup_transform(target, source, rclpy.time.Time()))
            except TransformException:
                return None

        def _on_owner(self, msg) -> None:
            self.t_owner = time.monotonic()
            T = self._T(self.odom, msg.header.frame_id)
            if T is None:
                self.get_logger().warn(f"no TF {self.odom} <- {msg.header.frame_id}: cannot align the phone",
                                       throttle_duration_sec=10.0)
                return
            p = T @ np.array([msg.point.x, msg.point.y, msg.point.z, 1.0])
            if not self.tracker.on_vision(time.monotonic(), float(p[0]), float(p[1])):
                self.get_logger().info("camera owner point does not fit the phone's walk (crossing person?) - "
                                       "not used for the phone alignment", throttle_duration_sec=3.0)

        def _on_lost(self, msg) -> None:
            self.lost_flag, self.lost_flag_t = bool(msg.data), time.monotonic()

        def _on_last_seen(self, msg) -> None:
            if not self.using_phone:
                self.pub_seen.publish(msg)                  # phone estimate not usable: pass through

        def _fit_tick(self) -> None:
            self.tracker.update_fit(time.monotonic())

        def _predict_tick(self) -> None:
            now = time.monotonic()
            # lost: FollowMe says so, or no owner detection for lost_after_s (FollowMe builds without
            # /followme/owner_lost never said it - the Pi then never got a phone estimate)
            self.lost = (self.lost_flag and now - self.lost_flag_t < 3.0) or now - self.t_owner > self.lost_after
            view = self.tracker.predict(now)          # always (web view), not only while lost
            if view is not None:
                v = PointStamped()
                v.header.stamp, v.header.frame_id = self.get_clock().now().to_msg(), self.odom
                v.point.x, v.point.y, v.point.z = view["x"], view["y"], view["sigma"]   # z = sigma (m)
                self.pub_view.publish(v)
            est = view if self.lost else None
            was = self.using_phone
            self.using_phone = est is not None
            if was != self.using_phone:
                self.get_logger().info("owner lost: following the phone estimate" if self.using_phone
                                       else "phone estimate no longer used")
            if est is None:
                return
            stamp = self.get_clock().now().to_msg()
            m = PointStamped()
            m.header.stamp, m.header.frame_id = stamp, self.odom
            m.point.x, m.point.y = est["x"], est["y"]
            self.pub_est.publish(m)
            T = self._T(self.base, self.odom)
            if T is None:
                return
            b = T @ np.array([est["x"], est["y"], 0.0, 1.0])
            out = PointStamped()
            out.header.stamp, out.header.frame_id = stamp, self.base
            out.point.x, out.point.y = float(b[0]), float(b[1])
            self.pub_seen.publish(out)

        def _status_tick(self) -> None:
            now = time.monotonic()
            st = self.tracker.status(now)
            st.update(lost=self.lost, using_phone=self.using_phone, posts=self.server.posts)
            est = self.tracker.predict(now)
            if est:
                st.update(walked=round(est["walked"], 1), sigma=round(est["sigma"], 2))
            self.pub_status.publish(String(data=json.dumps(st)))

        def destroy_node(self):
            self.server.close()
            return super().destroy_node()

    rclpy.init()
    node = PhonePdrNode()
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
