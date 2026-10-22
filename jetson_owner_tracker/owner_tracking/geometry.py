"""Bounding-box geometry helpers (boxes are ``x1, y1, x2, y2``)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

EPS = 1e-9


def bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > EPS else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` arrays."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > EPS, inter / np.maximum(union, EPS), 0.0).astype(np.float32)


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    out = np.empty_like(box)
    out[..., 0] = (box[..., 0] + box[..., 2]) * 0.5
    out[..., 1] = (box[..., 1] + box[..., 3]) * 0.5
    out[..., 2] = box[..., 2] - box[..., 0]
    out[..., 3] = box[..., 3] - box[..., 1]
    return out


def cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    out = np.empty_like(state)
    out[..., 0] = state[..., 0] - state[..., 2] * 0.5
    out[..., 1] = state[..., 1] - state[..., 3] * 0.5
    out[..., 2] = state[..., 0] + state[..., 2] * 0.5
    out[..., 3] = state[..., 1] + state[..., 3] * 0.5
    return out


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)


def contains_point(box: np.ndarray, x: float, y: float) -> bool:
    return bool(box[0] <= x <= box[2] and box[1] <= y <= box[3])


def clip_to_frame(box: np.ndarray, width: int, height: int,
                  min_size: int = 2) -> Optional[tuple[int, int, int, int]]:
    """Integer crop coordinates clipped to the frame, or None if too small."""
    x1 = int(max(0, math.floor(box[0])))
    y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(width, math.ceil(box[2])))
    y2 = int(min(height, math.ceil(box[3])))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2
