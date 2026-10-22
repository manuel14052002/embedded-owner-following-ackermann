"""Navigation view for the FollowMe web stream (no ROS dependency).

* ``render_minimap``  top-down "virtual map" in base_link: lidar points, the robot footprint,
  the planner's path, the owner and the follow goal, plus the planner state.
* ``project_ground_path``  the same planned path projected onto the FLOOR in the camera image
  (augmented-reality path), using the camera intrinsics and the TF base_link <- camera.
* ``composite``  pastes the minimap into a corner of the camera frame.

All inputs are plain numpy arrays in base_link (x forward, y left, metres).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np


def scan_to_base(ranges: Sequence[float], angle_min: float, angle_inc: float, T_base_laser: np.ndarray,
                 max_range: float = 6.0) -> np.ndarray:
    """LaserScan -> (N, 2) points in base_link."""
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(r.size)
    ok = np.isfinite(r) & (r > 0.05) & (r < max_range)
    if not ok.any():
        return np.zeros((0, 2))
    P = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum()), np.ones(ok.sum())])
    return (T_base_laser @ P)[:2].T


def render_minimap(size: int = 220, fwd_m: float = 4.5, side_m: float = 2.5, back_m: float = 1.0,
                   scan_pts: Optional[np.ndarray] = None, path_pts: Optional[np.ndarray] = None,
                   owner: Optional[tuple] = None, goal: Optional[tuple] = None, status: str = "",
                   footprint=(0.215, -0.210, 0.155)) -> np.ndarray:
    """Top-down map, robot at the bottom centre looking up. Returns a BGR image (size x size)."""
    h = w = size
    img = np.full((h, w, 3), 24, np.uint8)
    scale = min(w / (2 * side_m), h / (fwd_m + back_m))          # px per metre
    ox, oy = w / 2.0, h - back_m * scale                          # base_link origin in pixels

    def px(x, y):                                                 # base (x fwd, y left) -> pixel
        return int(round(ox - y * scale)), int(round(oy - x * scale))

    for r in np.arange(1.0, fwd_m + 0.01, 1.0):                   # range rings every metre
        cv2.circle(img, (int(ox), int(oy)), int(r * scale), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r:.0f}m", (int(ox) + 3, int(oy - r * scale) + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (90, 90, 90), 1, cv2.LINE_AA)
    if scan_pts is not None and len(scan_pts):
        for x, y in scan_pts:
            u, v = px(x, y)
            if 0 <= u < w and 0 <= v < h:
                img[max(0, v - 1):v + 1, max(0, u - 1):u + 1] = (200, 200, 200)
    if path_pts is not None and len(path_pts) >= 2:
        pts = np.array([px(x, y) for x, y in path_pts], np.int32)
        cv2.polylines(img, [pts], False, (60, 220, 60), 2, cv2.LINE_AA)
    if goal is not None:
        u, v = px(*goal[:2])
        cv2.drawMarker(img, (u, v), (255, 200, 0), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
    if owner is not None:
        u, v = px(*owner[:2])
        cv2.circle(img, (u, v), 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "owner", (u + 9, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    front, rear, half = footprint                                 # robot footprint
    corners = np.array([px(front, half), px(front, -half), px(rear, -half), px(rear, half)], np.int32)
    cv2.fillPoly(img, [corners], (70, 70, 200))
    cv2.arrowedLine(img, px(0.0, 0.0), px(front + 0.15, 0.0), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)
    if status:
        cv2.rectangle(img, (0, 0), (w, 16), (0, 0, 0), -1)
        cv2.putText(img, status[:34], (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    return img


def project_ground_path(path_pts: np.ndarray, T_base_cam: np.ndarray, fx: float, fy: float, cx: float,
                        cy: float, ground_z: float = -0.09, width: int = 640, height: int = 480,
                        min_depth: float = 0.25) -> np.ndarray:
    """Planned path (base_link, on the floor at z = ground_z) -> pixel polyline in the camera
    image. T_base_cam maps camera-optical points into base_link; points behind or too close to
    the camera are dropped. Returns (M, 2) int pixels (may be empty)."""
    if path_pts is None or len(path_pts) < 2 or T_base_cam is None:
        return np.zeros((0, 2), np.int32)
    T_cam_base = np.linalg.inv(T_base_cam)
    n = len(path_pts)
    P = np.stack([path_pts[:, 0], path_pts[:, 1], np.full(n, ground_z), np.ones(n)])
    C = (T_cam_base @ P)[:3]
    ok = C[2] > min_depth
    if not ok.any():
        return np.zeros((0, 2), np.int32)
    u = fx * C[0, ok] / C[2, ok] + cx
    v = fy * C[1, ok] / C[2, ok] + cy
    keep = (u > -width) & (u < 2 * width) & (v > -height) & (v < 2 * height)
    return np.stack([u[keep], v[keep]], 1).astype(np.int32)


def draw_ground_path(img: np.ndarray, pix: np.ndarray) -> None:
    """Semi-transparent green ribbon along the projected path."""
    if pix is None or len(pix) < 2:
        return
    over = img.copy()
    cv2.polylines(over, [pix.reshape(-1, 1, 2)], False, (60, 220, 60), 6, cv2.LINE_AA)
    cv2.addWeighted(over, 0.6, img, 0.4, 0, dst=img)


def composite(img: np.ndarray, panel: np.ndarray, corner: str = "br", margin: int = 8) -> None:
    """Paste the minimap into a corner of the camera frame (in place)."""
    ph, pw = panel.shape[:2]
    H, W = img.shape[:2]
    if ph + margin > H or pw + margin > W:
        return
    y0 = H - ph - margin if corner[0] == "b" else margin
    x0 = W - pw - margin if corner[1] == "r" else margin
    roi = img[y0:y0 + ph, x0:x0 + pw]
    cv2.addWeighted(panel, 0.85, roi, 0.15, 0, dst=roi)


def path_from_arc(v: float, w: float, seconds: float = 2.5, dt: float = 0.1) -> np.ndarray:
    """Fallback 'simulated path' when the planner publishes none: integrate (v, w) forward."""
    x = y = th = 0.0
    out = [(0.0, 0.0)]
    for _ in range(int(seconds / dt)):
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        out.append((x, y))
    return np.array(out)


# a2r_phone_view: phone (PDR) estimate + data strip on the minimap -----------------------------
def draw_gps(img: np.ndarray, gps: Optional[tuple], sigma: float, status_json: str,
             fwd_m: float = 4.5, side_m: float = 2.5, back_m: float = 1.0) -> None:
    """a2r_step3d: orange dot where the owner's GPS (Firebase, read by the Pi's SIM7600) says the
    owner is - at the map edge with the distance when further away - and one status line."""
    import json as _json
    h, w = img.shape[:2]
    scale = min(w / (2 * side_m), h / (fwd_m + back_m))
    ox, oy = w / 2.0, h - back_m * scale
    try:
        st = _json.loads(status_json) if status_json else {}
    except ValueError:
        st = {}
    if not st or st.get("mode") != "outdoor":
        return
    col = (0, 165, 255)
    if gps is not None:
        d = math.hypot(gps[0], gps[1])
        u, v = ox - gps[1] * scale, oy - gps[0] * scale
        out = not (6 <= u < w - 6 and 34 <= v < h - 30)
        if out:                                                   # on the edge, along the bearing
            b = math.atan2(gps[1], gps[0])
            if math.cos(b) > 0.0:
                k = min((w / 2 - 8) / max(abs(math.sin(b)), 1e-6), (oy - 36) / max(math.cos(b), 1e-6))
                u, v = ox - math.sin(b) * k, oy - math.cos(b) * k
            else:                                                 # behind: lower corner on that side
                u, v = (8 if math.sin(b) > 0 else w - 9), h - 38
        u, v = int(round(u)), int(round(v))
        if sigma > 0 and not out:
            cv2.circle(img, (u, v), max(4, int(sigma * scale)), col, 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 5, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"GPS {d:.0f}m", (min(u + 8, w - 58), max(v - 6, 44)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, col, 1, cv2.LINE_AA)
    if st.get("modem") != "ok":
        line = f"GPS: modem {str(st.get('modem'))[:28]}"
    elif st.get("robot_fix_age_s") is None:
        line = "GPS: robot has no fix yet (sky view?)"
    elif st.get("owner_age_s") is None:
        line = "GPS: nothing from Firebase yet"
    elif not st.get("aligned"):
        line = f"GPS: owner {st.get('owner_dist_gps_m')}m - drive to find north"
    else:
        line = (f"GPS: owner {st.get('owner_dist_gps_m')}m, {st.get('owner_age_s')}s old, "
                f"N+/-{st.get('north_sigma_deg')}")
    cv2.rectangle(img, (0, 18), (w, 32), (0, 0, 0), -1)
    cv2.putText(img, line[:42], (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)


def draw_phone(img: np.ndarray, phone: Optional[tuple], sigma: float, status_json: str,
               fwd_m: float = 4.5, side_m: float = 2.5, back_m: float = 1.0) -> None:
    """Magenta dot (+ uncertainty circle) where the phone says the owner is, and a 2-line data strip
    at the bottom of the minimap. Same geometry as render_minimap (its default extents)."""
    import json as _json
    h, w = img.shape[:2]
    scale = min(w / (2 * side_m), h / (fwd_m + back_m))
    ox, oy = w / 2.0, h - back_m * scale
    try:
        st = _json.loads(status_json) if status_json else {}
    except ValueError:
        st = {}
    col = (255, 0, 255)
    if phone is not None:
        u, v = ox - phone[1] * scale, oy - phone[0] * scale
        out = not (6 <= u < w - 6 and 20 <= v < h - 30)
        u, v = int(round(min(max(u, 6), w - 7))), int(round(min(max(v, 20), h - 31)))
        if sigma > 0 and not out:
            cv2.circle(img, (u, v), max(4, int(sigma * scale)), col, 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 5, col, -1 if not out else 2, cv2.LINE_AA)
        cv2.putText(img, "phone" + (" >" if out else ""), (min(u + 8, w - 52), v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, col, 1, cv2.LINE_AA)
    if not st:
        line1, line2 = "phone: no data - open :8443 on the phone", ""
    elif not st.get("phone"):
        line1, line2 = "phone: not connected - tap Start", ""
    else:
        line1 = f"phone: {st.get('steps', 0)} steps"
        if st.get("aligned"):
            line1 += f"  step {st.get('step_len')}m  fit {st.get('rms')}m" + ("  mirr" if st.get("mirrored") else "")
        else:
            line1 += "  calibrating (walk in view)"
        if st.get("using_phone"):
            line2 = f"FOLLOWING PHONE  {st.get('walked', 0)}m walked  +/-{st.get('sigma', 0)}m"
        elif st.get("lost"):
            line2 = "owner lost - phone estimate not usable"
        else:
            line2 = "camera sees owner - phone standing by"
    y0 = h - 28
    cv2.rectangle(img, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.putText(img, line1[:40], (4, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    if line2:
        cv2.putText(img, line2[:40], (4, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (0, 220, 255) if st.get("using_phone") else (200, 200, 200), 1, cv2.LINE_AA)
