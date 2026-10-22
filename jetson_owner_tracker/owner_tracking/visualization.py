"""OpenCV overlay rendering for the owner tracker.

The visualizer draws into a caller-supplied display buffer, never into the
frame that the Re-ID model sees, so overlays cannot contaminate embeddings.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from owner_tracking.datatypes import FrameResult, TrackState

BGR = tuple[int, int, int]

STATE_COLORS: dict[TrackState, BGR] = {
    TrackState.UNINITIALIZED: (200, 200, 200),
    TrackState.TRACKING: (60, 200, 60),
    TrackState.LOW_CONFIDENCE: (0, 200, 255),
    TrackState.AMBIGUOUS: (255, 0, 255),
    TrackState.REIDENTIFYING: (255, 180, 0),
    TrackState.LOST: (0, 0, 255),
}
OTHER_COLOR: BGR = (150, 150, 150)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def draw_dashed_rect(img: np.ndarray, bbox: np.ndarray, color: BGR, thickness: int = 1,
                     dash: int = 8) -> None:
    """Draw a dashed rectangle (used for the Kalman prediction)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(np.hypot(bx - ax, by - ay))
        for s in range(0, max(length, 1), dash * 2):
            e = min(s + dash, length)
            p = (int(ax + (bx - ax) * s / max(length, 1)), int(ay + (by - ay) * s / max(length, 1)))
            q = (int(ax + (bx - ax) * e / max(length, 1)), int(ay + (by - ay) * e / max(length, 1)))
            cv2.line(img, p, q, color, thickness, cv2.LINE_AA)


class Visualizer:
    """Draws owner box, state, confidence channels, scores and FPS."""

    def __init__(self, show_debug: bool = False, font_scale: float = 0.5) -> None:
        self.show_debug = show_debug
        self.font_scale = font_scale

    def toggle_debug(self) -> None:
        self.show_debug = not self.show_debug

    def _text(self, img: np.ndarray, text: str, org: tuple[int, int], color: BGR = (255, 255, 255),
              scale: Optional[float] = None) -> None:
        scale = scale or self.font_scale
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def draw(self, canvas: np.ndarray, result: FrameResult, fps: float = 0.0,
             latency_ms: float = 0.0, message: str = "") -> np.ndarray:
        """Render overlays onto ``canvas`` in place and return it."""
        color = STATE_COLORS.get(result.state, (255, 255, 255))
        # Other detections (thin grey), with debug scores if requested.
        for i, det in enumerate(result.detections):
            if i == result.owner_detection_index and result.reported:
                continue
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), OTHER_COLOR, 1)
            if self.show_debug and i < len(result.candidates):
                c = result.candidates[i]
                tag = f"s={_fmt(c.total)} a={_fmt(c.appearance_similarity)}"
                if c.rejected_reason:
                    tag += f" [{c.rejected_reason}]"
                self._text(canvas, tag, (x1, max(12, y1 - 4)), OTHER_COLOR, 0.4)
        # Kalman prediction (dashed).
        if result.predicted_bbox is not None and result.owner_id is not None:
            draw_dashed_rect(canvas, result.predicted_bbox, color)
        # Owner box - only when the tracker asserts it.
        if result.reported and result.owner_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in result.owner_bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            self._text(canvas, f"OWNER {result.owner_id}  {result.state.value}",
                       (x1, max(14, y1 - 6)), color, 0.55)
        self._panel(canvas, result, color, fps, latency_ms, message)
        return canvas

    def _panel(self, canvas: np.ndarray, r: FrameResult, color: BGR, fps: float,
               latency_ms: float, message: str) -> None:
        c = r.confidence
        best = r.best_candidate
        lines = [
            (f"Owner ID: {r.owner_id or '-'}   State: {r.state.value}", color),
            (f"Confidence: {_fmt(c.overall)}  (app {_fmt(c.appearance)} mot {_fmt(c.motion)} "
             f"spa {_fmt(c.spatial)} det {_fmt(c.detection)} id {_fmt(c.identity)})", (255, 255, 255)),
            (f"Re-ID sim: {_fmt(best.appearance_similarity if best else None)}   "
             f"Assoc score: {_fmt(best.total if best else None)}   margin: {_fmt(r.score_margin)}",
             (255, 255, 255)),
            (f"Decision: {r.decision.value}   gallery: {r.gallery_size}   "
             f"FPS: {fps:.1f}   latency: {latency_ms:.1f} ms", (255, 255, 255)),
        ]
        if r.identity_switch_attempt:
            lines.append(("ID-switch attempt blocked", (0, 0, 255)))
        if message:
            lines.append((message, (0, 255, 255)))
        if r.owner_id is None:
            lines.append(("Click a person to select the owner. R=reset Q=quit Space=pause D=debug",
                          (0, 255, 255)))
        height = 18 * len(lines) + 10
        width = min(canvas.shape[1], 640)
        roi = canvas[0:height, 0:width]
        cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0, dst=roi)
        for k, (text, col) in enumerate(lines):
            self._text(canvas, text, (8, 18 + 18 * k), col)
