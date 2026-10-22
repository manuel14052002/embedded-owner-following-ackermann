"""Camera sources with a ``cv2.VideoCapture``-like interface.

:class:`RealSenseCapture` reads the colour stream of an Intel RealSense
camera (D415/D435/D455...) through ``pyrealsense2``. It optionally streams
depth aligned to the colour image, exposed as :attr:`last_depth_m` (metres,
float32, 0 = no data) for future use by depth-aware association and by the FollowMe localisation. :attr:`intrinsics`
holds the colour-stream ``rs.intrinsics`` (``fx, fy, ppx, ppy``).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class RealSenseCapture:
    """Minimal ``read()/release()/get()/isOpened()`` wrapper around pyrealsense2."""

    def __init__(self, serial: Optional[str] = None, width: int = 640, height: int = 480,
                 fps: int = 30, enable_depth: bool = False, warmup_frames: int = 30,
                 timeout_ms: int = 5000) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("pyrealsense2 is not installed: pip install pyrealsense2") from exc
        self._rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.timeout_ms = timeout_ms
        self.enable_depth = enable_depth
        self.last_depth_m: Optional[np.ndarray] = None
        self._buffer: Optional[np.ndarray] = None
        self._opened = False

        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError("No RealSense device found (check the USB 3 cable/port)")
        names = [f"{d.get_info(rs.camera_info.name)} (serial {d.get_info(rs.camera_info.serial_number)})"
                 for d in devices]
        LOGGER.info("RealSense devices: %s", "; ".join(names))

        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        try:
            profile = self._pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not start RealSense colour stream {width}x{height}@{fps}: {exc}. "
                "Try 640x480@30 or 1280x720@30, and make sure the camera is on a USB 3 port "
                "and not open in another program (e.g. realsense-viewer).") from exc
        self._opened = True
        # Colour intrinsics (depth is aligned to colour, so these deproject depth too).
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._align = rs.align(rs.stream.color) if enable_depth else None
        self._depth_scale = 0.0
        if enable_depth:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # Let auto-exposure / white balance settle so the owner's first
        # embedding (taken at the click) is not from a dark, washed-out frame.
        for _ in range(max(0, warmup_frames)):
            try:
                self._pipeline.wait_for_frames(self.timeout_ms)
            except RuntimeError:
                break

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self._opened

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, bgr_frame)``. The frame is owned by this object and
        overwritten on the next call (one copy out of the driver's buffer)."""
        if not self._opened:
            return False, None
        try:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
        except RuntimeError as exc:
            LOGGER.error("RealSense frame timeout: %s", exc)
            return False, None
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return False, None
        src = np.asanyarray(color.get_data())
        if self._buffer is None or self._buffer.shape != src.shape:
            self._buffer = np.empty_like(src)
        # librealsense recycles its buffers, so the data must be copied out.
        np.copyto(self._buffer, src)
        if self._align is not None:
            depth = frames.get_depth_frame()
            self.last_depth_m = (np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
                                 if depth else None)
        return True, self._buffer

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self) -> None:
        if self._opened:
            self._pipeline.stop()
            self._opened = False
