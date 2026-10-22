#!/usr/bin/env python3
"""Interactive owner tracker.

Usage::

    python app.py --source 0                       # webcam
    python app.py --source video.mp4 --config config/default.yaml --save-video out.mp4
    python app.py --source realsense                  # Intel RealSense (D435 etc.)
    python app.py --source realsense:123456789012     # a specific RealSense by serial

Controls: left-click a person to select the owner (ignored while an owner is
locked), R = reset owner, Space = pause, D = toggle debug scores, Q/Esc = quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from owner_tracking.config import TrackerConfig
from owner_tracking.factory import build_detector, build_reid
from owner_tracking.logging_utils import CsvFrameLogger
from owner_tracking.runtime import resolve_device, seed_everything
from owner_tracking.stats import PerformanceStats, StageTimer
from owner_tracking.tracker import OwnerTracker
from owner_tracking.visualization import Visualizer

LOGGER = logging.getLogger("owner_tracker.app")
WINDOW = "Owner Tracker"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0",
                   help="camera index, video file, stream URL, or 'realsense[:SERIAL]'")
    p.add_argument("--rs-width", type=int, default=640, help="RealSense colour width")
    p.add_argument("--rs-height", type=int, default=480, help="RealSense colour height")
    p.add_argument("--rs-fps", type=int, default=30, help="RealSense colour frame rate")
    p.add_argument("--config", default=None, help="YAML config (defaults built in)")
    p.add_argument("--device", default=None, help="override runtime.device (auto|cpu|cuda[:N])")
    p.add_argument("--detector-model", default=None, help="override detector.model_path")
    p.add_argument("--reid-model", default=None, help="override reid.model_path (ReID-trained weights)")
    p.add_argument("--reid-backend", default=None, help="override reid.backend")
    p.add_argument("--log-csv", default=None, help="override logging.csv_path ('' disables)")
    p.add_argument("--save-video", default=None, help="write the annotated stream to this file")
    p.add_argument("--no-display", action="store_true", help="headless (no window; no clicks)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


class ClickBuffer:
    """Holds the most recent mouse click until the main loop consumes it."""

    def __init__(self) -> None:
        self.point: Optional[tuple[int, int]] = None

    def callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)

    def pop(self) -> Optional[tuple[int, int]]:
        point, self.point = self.point, None
        return point


def open_source(source: str, args: Optional[argparse.Namespace] = None):
    """Open a camera index, file/URL, or ``realsense[:SERIAL]``."""
    if source.lower().startswith("realsense"):
        from owner_tracking.camera import RealSenseCapture
        _, _, serial = source.partition(":")
        return RealSenseCapture(serial=serial or None,
                                width=getattr(args, "rs_width", 640),
                                height=getattr(args, "rs_height", 480),
                                fps=getattr(args, "rs_fps", 30))
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source '{source}'")
    return cap


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = TrackerConfig.from_yaml(args.config) if args.config else TrackerConfig()
    if args.device:
        config.runtime.device = args.device
    if args.detector_model:
        config.detector.model_path = args.detector_model
    if args.reid_model is not None:
        config.reid.model_path = args.reid_model
    if args.reid_backend:
        config.reid.backend = args.reid_backend
    if args.log_csv is not None:
        config.logging.enabled = bool(args.log_csv)
        config.logging.csv_path = args.log_csv or config.logging.csv_path
    config.validate()
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    LOGGER.info("Using device %s", device)

    try:
        detector = build_detector(config.detector, device)
        reid = build_reid(config.reid, device)
    except Exception as exc:  # noqa: BLE001 - surface a clear message and exit
        LOGGER.error("Model loading failed: %s", exc)
        return 2
    tracker = OwnerTracker(config, reid, detector=detector)
    try:
        cap = open_source(args.source, args)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2
    visualizer = Visualizer()
    stats = PerformanceStats()
    clicks = ClickBuffer()
    csv_logger = (CsvFrameLogger(config.logging.csv_path, config.logging.interval,
                                 config.logging.flush_every) if config.logging.enabled else None)
    writer: Optional[cv2.VideoWriter] = None
    display: Optional[np.ndarray] = None   # preallocated; the raw frame is never drawn on
    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, clicks.callback)
    paused, message, message_until = False, "", 0.0
    frame: Optional[np.ndarray] = None
    result = None
    try:
        while True:
            if not paused or frame is None:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.info("End of stream")
                    break
                with StageTimer(stats, "total"):
                    result = tracker.update(frame)
                stats.tick()
                if csv_logger is not None:
                    csv_logger.log(result)
            # Owner selection: clicks are consumed only when no owner is locked.
            point = clicks.pop()
            if point is not None:
                ok_sel, msg = tracker.select_owner(frame, point)
                message, message_until = msg, time.time() + 3.0
                LOGGER.info("Click at %s: %s", point, msg)
            if display is None or display.shape != frame.shape:
                display = np.empty_like(frame)
            np.copyto(display, frame)
            visualizer.draw(display, result, fps=stats.fps, latency_ms=stats.latency("total"),
                            message=message if time.time() < message_until else "")
            if args.save_video:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (display.shape[1], display.shape[0]))
                if not paused:
                    writer.write(display)
            if args.no_display:
                continue
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                tracker.reset()
                message, message_until = "Owner reset - click a person", time.time() + 3.0
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("d"), ord("D")):
                visualizer.toggle_debug()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_logger is not None:
            csv_logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    LOGGER.info("Performance: %s", stats.summary())
    LOGGER.info("Tracker counters: %s", dict(tracker.counters))
    return 0


if __name__ == "__main__":
    sys.exit(main())
