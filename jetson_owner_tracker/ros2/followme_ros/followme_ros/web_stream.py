"""MJPEG web server: view the FollowMe overlay in any browser, in real time.

    http://<jetson-ip>:8081/          page with the live view + robot state
    http://<jetson-ip>:8081/stream    raw MJPEG stream (VLC, <img src=...>, etc.)

Design for smooth, low-latency video:
  * push() only stores a reference to the newest frame - it never encodes, so the
    perception loop is not slowed down (encoding used to cost several ms per frame there);
  * one encoder thread converts the newest frame to JPEG at up to ``max_fps``;
  * every client always receives the NEWEST jpeg (older ones are skipped, never queued),
    with a send buffer of a few frames so short Wi-Fi hiccups do not cause stutter;
  * nothing is encoded while nobody watches.
No ROS dependency.
"""

from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A2RBOT FollowMe</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto;margin-top:8px;border:1px solid #333;image-rendering:auto}
#s{font-size:1.2em;margin:8px}</style></head><body>
<div id="s">A2RBOT FollowMe</div><img id="v" src="/stream" alt="stream">
<script>
setInterval(()=>fetch('/state').then(r=>r.text()).then(t=>{document.getElementById('s').textContent=t}),500);
// reconnect automatically if the stream stalls (e.g. Wi-Fi drop)
const v=document.getElementById('v');
v.onerror=()=>setTimeout(()=>{v.src='/stream?'+Date.now()},500);
</script>
</body></html>"""


class MjpegServer:
    def __init__(self, port: int = 8081, quality: int = 60, max_fps: float = 25.0,
                 max_width: int = 480, send_buffer: int = 256 * 1024) -> None:
        self.quality = int(quality)
        self.min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
        self.max_width = int(max_width)
        send_buf = int(send_buffer)
        self._cond = threading.Condition()
        self._raw: Optional[np.ndarray] = None      # newest frame (not yet encoded)
        self._raw_seq = 0
        self._jpeg: Optional[bytes] = None           # newest encoded frame
        self._seq = 0
        self.clients = 0
        self.frames_encoded = 0
        self.state_text = "starting"
        self._running = True
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/state":
                    body = server.state_text.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/stream"):
                    try:
                        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf)
                        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except OSError:
                        pass
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    with server._cond:
                        server.clients += 1
                        server._cond.notify_all()
                    seen = -1
                    try:
                        while server._running:
                            with server._cond:
                                server._cond.wait_for(lambda: server._seq != seen or not server._running,
                                                      timeout=2.0)
                                jpeg, seen = server._jpeg, server._seq
                            if jpeg is None:
                                continue
                            # one write per frame: header + data together (no torn frames)
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with server._cond:
                            server.clients -= 1
                else:
                    self.send_error(404)

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                import sys
                if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
                    return                              # browser closed / reloaded the page: normal
                super().handle_error(request, client_address)

        self.httpd = QuietServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._encoder, daemon=True).start()

    @property
    def watched(self) -> bool:
        return self.clients > 0

    def push(self, bgr: np.ndarray, state_text: Optional[str] = None) -> None:
        """Offer the newest frame. Never encodes here (cheap for the caller); the encoder
        thread picks up only the newest frame, so slow encoding skips frames, never lags."""
        if state_text is not None:
            self.state_text = state_text
        if not self.watched:
            return
        with self._cond:
            self._raw = bgr
            self._raw_seq += 1
            self._cond.notify_all()

    def _encoder(self) -> None:
        last_seq, last_t = 0, 0.0
        while self._running:
            with self._cond:
                self._cond.wait_for(lambda: (self._raw_seq != last_seq and self.clients > 0) or not self._running,
                                    timeout=1.0)
                if not self._running:
                    return
                if self._raw_seq == last_seq or self.clients == 0:
                    continue
                frame, last_seq = self._raw, self._raw_seq
            wait = self.min_dt - (time.monotonic() - last_t)
            if wait > 0:
                time.sleep(wait)
                with self._cond:                         # take the NEWEST frame after waiting
                    frame, last_seq = self._raw, self._raw_seq
            if self.max_width and frame.shape[1] > self.max_width:
                h = int(frame.shape[0] * self.max_width / frame.shape[1])
                frame = cv2.resize(frame, (self.max_width, h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            last_t = time.monotonic()
            if not ok:
                continue
            with self._cond:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self.frames_encoded += 1
                self._cond.notify_all()

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self.httpd.shutdown()
