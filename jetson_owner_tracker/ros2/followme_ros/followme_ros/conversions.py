"""Message <-> numpy helpers with no ROS imports (unit-testable anywhere).

cv_bridge is deliberately avoided: on Jetson it is often built against a
different NumPy / OpenCV than the one the perception stack uses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def image_to_bgr(msg: Any) -> np.ndarray:
    """``sensor_msgs/Image`` (bgr8 / rgb8 / bgra8 / rgba8 / mono8) -> BGR uint8."""
    enc = msg.encoding.lower()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}.get(enc)
    if channels is None:
        raise ValueError(f"unsupported colour encoding '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if enc == "bgr8":
        return img.copy()
    if enc == "rgb8":
        return img[..., ::-1].copy()
    if enc == "bgra8":
        return img[..., :3].copy()
    if enc == "rgba8":
        return img[..., 2::-1].copy()
    return np.repeat(img, 3, axis=2)


def depth_to_meters(msg: Any) -> np.ndarray:
    """``sensor_msgs/Image`` 16UC1 (mm) or 32FC1 (m) -> float32 metres, 0 = invalid."""
    enc = msg.encoding.upper()
    if enc in ("16UC1", "MONO16"):
        dtype, scale = np.uint16, 0.001
    elif enc == "32FC1":
        dtype, scale = np.float32, 1.0
    else:
        raise ValueError(f"unsupported depth encoding '{msg.encoding}'")
    dt = np.dtype(dtype).newbyteorder(">" if getattr(msg, "is_bigendian", 0) else "<")
    itemsize = dt.itemsize
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    arr = buf[:, :msg.width * itemsize].copy().view(dt).reshape(msg.height, msg.width)
    out = arr.astype(np.float32) * scale
    out[~np.isfinite(out)] = 0.0
    return out


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_to_matrix(t: Any) -> np.ndarray:
    """``geometry_msgs/Transform`` -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    q = t.rotation
    T[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def yaw_from_matrix(T: np.ndarray) -> float:
    return math.atan2(T[1, 0], T[0, 0])
