#!/usr/bin/env python3
"""camera_stream.py — MJPEG web stream for a2r_bot_follow

Run standalone:
    python3 camera_stream.py

Or launched automatically by follow_camera.launch.py.

View from any device on the LAN:
    http://<pi-ip>:8080/             <- live stream page
    http://<pi-ip>:8080/stream       <- raw MJPEG  (VLC, cv2, browser img tag)
    http://<pi-ip>:8080/snapshot     <- single JPEG frame
"""

import cv2
import time
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── config ────────────────────────────────────────────────────────────────────
DEVICE   = 4          # /dev/video0
WIDTH    = 640
HEIGHT   = 480
FPS      = 15
PORT     = 8080
QUALITY  = 80         # JPEG quality 1-100
# ─────────────────────────────────────────────────────────────────────────────

# shared latest frame + lock so HTTP threads don't fight the capture thread
_lock  = threading.Lock()
_frame = None          # bytes: latest JPEG

INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>a2r_bot camera</title>
  <style>
    body  { background:#111; margin:0; display:flex; flex-direction:column;
            align-items:center; justify-content:center; height:100vh; }
    img   { max-width:100%; border:2px solid #0f0; }
    h2    { color:#0f0; font-family:monospace; margin-bottom:12px; }
    p     { color:#888; font-family:monospace; font-size:0.85em; }
  </style>
</head>
<body>
  <h2>a2r_bot live camera</h2>
  <img src="/stream" alt="stream loading...">
  <p>/stream &nbsp;|&nbsp; /snapshot</p>
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request access log

    def do_GET(self):
        if self.path == '/':
            self._send_index()
        elif self.path == '/stream':
            self._send_stream()
        elif self.path == '/snapshot':
            self._send_snapshot()
        else:
            self.send_error(404)

    def _send_index(self):
        body = INDEX_HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        try:
            while True:
                with _lock:
                    frame = _frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b'--jpgboundary\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected — normal

    def _send_snapshot(self):
        with _lock:
            frame = _frame
        if frame is None:
            self.send_error(503, 'No frame yet')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(frame))
        self.end_headers()
        self.wfile.write(frame)


def capture_loop():
    global _frame
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    if not cap.isOpened():
        print(f'[camera_stream] ERROR: cannot open /dev/video{DEVICE}', flush=True)
        sys.exit(1)

    print(f'[camera_stream] camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}'
          f'x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
          f' @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps', flush=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
    while True:
        ok, bgr = cap.read()
        if not ok:
            print('[camera_stream] WARNING: dropped frame', flush=True)
            time.sleep(0.05)
            continue
        _, jpg = cv2.imencode('.jpg', bgr, encode_params)
        with _lock:
            _frame = jpg.tobytes()


def main():
    # capture runs in background thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # wait for first frame before accepting connections
    print('[camera_stream] waiting for first frame...', flush=True)
    while _frame is None:
        time.sleep(0.05)

    server = HTTPServer(('0.0.0.0', PORT), StreamHandler)

    def _shutdown(sig, frame):
        print('\n[camera_stream] shutting down', flush=True)
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f'[camera_stream] stream live at http://<pi-ip>:{PORT}/', flush=True)
    print(f'[camera_stream] raw MJPEG  at http://<pi-ip>:{PORT}/stream', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
