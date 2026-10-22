#!/usr/bin/env python3
"""FollowMe without ROS: perception + state machine with a recording navigation backend.

Usage::

    python followme_app.py --source realsense --config config/followme.yaml
    python followme_app.py --source realsense --target-model models/target.npz   # skip calibration
    python followme_app.py --source 0 --rgb-only                                  # webcam, no depth

Keys: C = start calibration (target alone in view), S = save target model,
F / W = inject a Follow / Wait command, R = drop calibration, Q/Esc = quit.
Goals are expressed in the camera-centred robot frame (x forward, y left);
on the robot, use the ROS 2 node in ``ros2/followme_ros`` instead.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from followme.config import FollowMeConfig
from followme.factory import build_perception
from followme.gestures import FOLLOW, WAIT
from followme.localization import CameraIntrinsics, intrinsics_from_fov
from followme.navigation import RecordingNavigation
from followme.ownership import OwnershipManager
from followme.pipeline import FollowMeController
from followme.visualization import draw
from owner_tracking.runtime import resolve_device

LOGGER = logging.getLogger("followme.app")
WINDOW = "FollowMe"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="realsense", help="'realsense[:SERIAL]', camera index or video file")
    p.add_argument("--config", default=None)
    p.add_argument("--target-model", default="models/target.npz", help="load if present; S saves here")
    p.add_argument("--rgb-only", action="store_true", help="no depth: range from box height (testing)")
    p.add_argument("--hfov", type=float, default=69.0, help="webcam horizontal FOV for --rgb-only")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--log-csv", default="logs/followme.csv")
    p.add_argument("--save-video", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = FollowMeConfig.from_yaml(args.config) if args.config else FollowMeConfig()
    device = resolve_device(cfg.device)

    depth_enabled = False
    if args.source.lower().startswith("realsense"):
        from owner_tracking.camera import RealSenseCapture
        _, _, serial = args.source.partition(":")
        cap = RealSenseCapture(serial=serial or None, width=args.width, height=args.height, fps=args.fps,
                               enable_depth=not args.rgb_only)
        intr = CameraIntrinsics.from_realsense(cap.intrinsics)
        depth_enabled = not args.rgb_only
    else:
        cap = cv2.VideoCapture(int(args.source)) if args.source.isdigit() else cv2.VideoCapture(args.source)
        if not cap.isOpened():
            LOGGER.error("Cannot open %s", args.source)
            return 2
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        intr = intrinsics_from_fov(w, h, args.hfov)
    if not depth_enabled:
        cfg.localization.bbox_range_fallback = True
        LOGGER.warning("No depth: using the box-height range fallback (testing only)")

    try:
        perception = build_perception(cfg, device, intr, args.target_model)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Model loading failed: %s", exc)
        return 2
    nav = RecordingNavigation()
    controller = FollowMeController(cfg, nav)
    ownership = None
    if cfg.ownership.enabled:
        if perception.gestures is None:
            LOGGER.error("ownership.enabled needs the gesture classifier (models/gesture_svm.joblib)")
        else:
            ownership = OwnershipManager(cfg, perception)
            LOGGER.info("Ownership by gesture enabled: raise an open palm to claim the robot")

    log_file = None
    writer_csv = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        writer_csv = csv.writer(log_file)
        writer_csv.writerow(["t", "mode", "state", "n_persons", "target", "min_distance", "lambda_d",
                             "px", "py", "pz", "theta", "gesture_raw", "gesture_cmd", "action",
                             "goal_x", "goal_y", "total_ms"])
    video = None
    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    message, message_until = "Press C to calibrate the target" if perception.model is None else "", time.time() + 5
    t_prev, fps = time.perf_counter(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            depth = getattr(cap, "last_depth_m", None) if depth_enabled else None
            now = time.monotonic()
            result = perception.process(frame, depth, now)
            action = controller.step(result)
            if ownership is not None:
                for ev in ownership.update(frame, depth, result, now):
                    if ev == "follow":
                        action = controller.command(FOLLOW, now)
                    elif ev in ("released", "claim_started", "claim_aborted"):
                        controller.machine.reset()
                        controller.stop(now)
            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(1e-6, t - t_prev) if fps else 1.0 / max(1e-6, t - t_prev)
            t_prev = t
            if writer_csv is not None:
                p = result.p_filt if result.p_filt is not None else [np.nan] * 3
                g = action.goal
                writer_csv.writerow([f"{now:.3f}", result.mode.value, controller.state.value, len(result.persons),
                                     "" if result.target_index is None else result.target_index,
                                     f"{np.min(result.distances):.3f}" if len(result.distances) else "",
                                     f"{result.threshold:.3f}", *[f"{v:.3f}" for v in p],
                                     "" if result.theta is None else f"{result.theta:.3f}",
                                     (result.gesture.raw or "") if result.gesture else "",
                                     (result.gesture.command or "") if result.gesture else "",
                                     action.kind.value, f"{g.x:.3f}" if g else "", f"{g.y:.3f}" if g else "",
                                     f"{result.timings_ms['total']:.1f}"])
            display = frame.copy()
            text = message if time.time() < message_until else ""
            if ownership is not None:
                text = text or f"[{ownership.state.value}] {ownership.message}"
                if ownership.claimant_box is not None and ownership.state.value == "CLAIMING":
                    x1, y1, x2, y2 = ownership.claimant_box.astype(int)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
            draw(display, result, controller.state, action, controller.safety_radius, fps, text)
            if args.save_video:
                if video is None:
                    video = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                            (display.shape[1], display.shape[0]))
                video.write(display)
            if args.no_display:
                continue
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                perception.start_calibration()
                message, message_until = "Calibrating...", time.time() + 3
            elif key == ord("s") and perception.model is not None:
                perception.model.save(args.target_model)
                message, message_until = f"Saved {args.target_model}", time.time() + 3
            elif key == ord("e"):
                ok_fin = perception.finish_calibration()
                message = "Calibration finished early" if ok_fin else "Need >= 6 samples first"
                message_until = time.time() + 3
            elif key == ord("r"):
                if ownership is not None:
                    ownership.release(now, "R key")
                else:
                    perception.clear_target()
                controller.machine.reset()
                message, message_until = "Calibration dropped - press C", time.time() + 3
            elif key == ord("f"):
                controller.command(FOLLOW, now)
            elif key == ord("w"):
                controller.command(WAIT, now)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if log_file:
            log_file.close()
        if video is not None:
            video.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    LOGGER.info("Transitions: %s", controller.machine.transitions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
