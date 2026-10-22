"""Overlay: masks, Eq. 1 distances vs lambda_d, target, state, gestures."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from followme.pipeline import Mode, PerceptionResult
from followme.state_machine import Action, State

STATE_COLORS = {State.FOLLOW: (0, 200, 0), State.STEADY: (0, 200, 255),
                State.SEARCH: (0, 0, 255), State.WAIT: (255, 150, 0)}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _text(img, s, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(img: np.ndarray, r: PerceptionResult, state: Optional[State], action: Optional[Action],
         safety_radius: float, fps: float = 0.0, message: str = "") -> None:
    """Draw in place on ``img`` (a copy of the frame)."""
    for i, p in enumerate(r.persons):
        is_target = i == r.target_index
        color = (0, 220, 0) if is_target else (60, 60, 230)
        if p.mask.shape == img.shape[:2]:
            contours, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 3 if is_target else 2)
        x1, y1, x2, y2 = p.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_target else 1)
        d = r.distances[i] if i < len(r.distances) else math.inf
        label = "TARGET" if is_target else "person"
        if np.isfinite(d):
            label += f" d={d:.3f}"
        _text(img, label, (x1 + 2, max(15, y1 - 6)), color)
    if r.gesture is not None and r.gesture.hand is not None:
        ox, oy = r.gesture.crop_origin
        pts = (r.gesture.hand.landmarks[:, :2] + [ox, oy]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 0), 1, cv2.LINE_AA)
    y = 22
    col = STATE_COLORS.get(state, (255, 255, 255)) if state else (255, 255, 255)
    _text(img, f"{state.value if state else '-'}  [{r.mode.value}]  {fps:4.1f} Hz", (8, y), col, 0.7)
    y += 24
    if r.mode == Mode.CALIBRATING:
        w = img.shape[1] - 16
        cv2.rectangle(img, (8, y - 12), (8 + w, y), (80, 80, 80), -1)
        cv2.rectangle(img, (8, y - 12), (8 + int(w * r.calibration_progress), y), (0, 200, 255), -1)
        y += 20
        _text(img, "CALIBRATING: target alone, walk near/far, turn around", (8, y), (0, 200, 255))
        y += 22
    if np.isfinite(r.threshold):
        _text(img, f"threshold = {r.threshold:.3f}", (8, y))
        y += 20
    if r.identity_info:
        _text(img, r.identity_info, (8, y), (200, 200, 255))
        y += 20
    if r.p_filt is not None:
        _text(img, f"target  x={r.p_filt[0]:+.2f} z={r.p_filt[2]:.2f} m  theta={math.degrees(r.theta):+.0f} deg",
              (8, y))
        y += 20
    _text(img, f"safety circle {safety_radius:.2f} m", (8, y))
    y += 20
    if r.gesture is not None:
        _text(img, f"gesture {r.gesture.raw or '-'}" + (f" -> {r.gesture.command}" if r.gesture.command else ""),
              (8, y), (255, 255, 0))
        y += 20
    if action is not None:
        _text(img, f"action {action.kind.value}", (8, y))
        y += 20
    if message:
        _text(img, message, (8, img.shape[0] - 12), (0, 255, 255), 0.6)
