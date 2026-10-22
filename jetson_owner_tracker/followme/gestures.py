"""Hand gesture commands (paper Sect. III-D).

MediaPipe Hands gives 21 ordered 3-D hand key points; flattened relative to
the palm centre they form a 63-d vector (21 x 3) that an RBF-kernel,
one-versus-one multi-class SVM classifies as

    * ``WAIT``   - open hand   (Fig. 4a)
    * ``FOLLOW`` - closed hand (Fig. 4b)
    * ``OTHER``  - anything else, ignored (Fig. 4c)

A command is sent to the state machine only after ``xi`` (default 5) equal
consecutive classifications. Gestures are read from the re-identified
target's crop only (Fig. 1: Re-ID -> gesture detection), so bystanders
cannot command the robot.
"""

from __future__ import annotations

import logging
import threading
import importlib
import multiprocessing as mp_proc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from followme.config import GestureConfig

LOGGER = logging.getLogger(__name__)

WAIT, FOLLOW, OTHER = "WAIT", "FOLLOW", "OTHER"
CLASSES = (WAIT, FOLLOW, OTHER)
PALM_IDS = (0, 5, 9, 13, 17)      # wrist + four MCP joints


@dataclass
class HandObservation:
    landmarks: np.ndarray          # (21, 3): x, y in crop pixels, z in the same pixel scale
    handedness: str                # "Left" | "Right" (as reported by MediaPipe)
    score: float


def landmark_features(landmarks: np.ndarray, handedness: str = "Right", mirror_left: bool = True) -> np.ndarray:
    """63-d feature: key points relative to the palm centre, scale-normalised.

    Translation is removed by subtracting the palm centre and scale by
    dividing by the wrist -> middle-finger-MCP length, so the vector does not
    depend on where the hand is or how far it is from the camera. Left hands
    are mirrored (x -> -x) so one classifier serves both hands.
    """
    pts = np.asarray(landmarks, np.float64).reshape(21, 3).copy()
    pts -= pts[list(PALM_IDS)].mean(axis=0)
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale < 1e-9:
        scale = 1.0
    pts /= scale
    if mirror_left and handedness.lower().startswith("left"):
        pts[:, 0] = -pts[:, 0]
    return pts.reshape(63).astype(np.float32)


class HandLandmarkDetector:
    """MediaPipe Hands, via the legacy ``solutions`` API or the Tasks API.

    ``mediapipe >= 0.10.30`` wheels ship only the Tasks API, which needs the
    ``hand_landmarker.task`` model file (download from the MediaPipe model
    card). Older wheels (typical on Jetson) have ``mp.solutions.hands``.
    """

    def __init__(self, cfg: GestureConfig, static_images: bool = True) -> None:
        import mediapipe as mp

        self.mp = mp
        self.cfg = cfg
        use_solutions = cfg.backend == "solutions" or (cfg.backend == "auto" and hasattr(mp, "solutions"))
        if use_solutions:
            self.kind = "solutions"
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=static_images, max_num_hands=1, model_complexity=0,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not Path(cfg.task_model_path).is_file():
                raise FileNotFoundError(
                    f"MediaPipe Tasks API needs '{cfg.task_model_path}' (hand_landmarker.task); "
                    "download it from the MediaPipe Hand Landmarker model card")
            self.kind = "tasks"
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=cfg.task_model_path),
                running_mode=vision.RunningMode.IMAGE, num_hands=1,
                min_hand_detection_confidence=cfg.min_detection_confidence,
                min_hand_presence_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence)
            self.hands = vision.HandLandmarker.create_from_options(opts)
        LOGGER.info("MediaPipe hands backend: %s", self.kind)

    _lock = threading.Lock()                    # MediaPipe graphs are not thread-safe

    def detect(self, bgr: np.ndarray) -> Optional[HandObservation]:
        with self._lock:
            return self._detect(bgr)

    def _detect(self, bgr: np.ndarray) -> Optional[HandObservation]:
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.kind == "solutions":
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lms = res.multi_hand_landmarks[0].landmark
            cls = res.multi_handedness[0].classification[0]
            label, score = cls.label, float(cls.score)
        else:
            res = self.hands.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                return None
            lms = res.hand_landmarks[0]
            cat = res.handedness[0][0]
            label, score = cat.category_name, float(cat.score)
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], np.float64)
        return HandObservation(pts, label, score)

    def close(self) -> None:
        close = getattr(self.hands, "close", None)
        if close:
            close()


class GestureClassifier:
    """RBF-kernel one-versus-one SVM (Sect. III-D) on 63-d landmark vectors."""

    def __init__(self, c: float = 10.0, gamma: Union[str, float] = "scale") -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        self.model = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", decision_function_shape="ovo", C=c, gamma=gamma))

    def fit(self, x: np.ndarray, y: Sequence[str]) -> "GestureClassifier":
        self.model.fit(np.asarray(x, np.float32), np.asarray(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(np.atleast_2d(np.asarray(x, np.float32)))

    def predict_one(self, feature: np.ndarray) -> str:
        return str(self.predict(feature)[0])

    def save(self, path: Union[str, Path]) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GestureClassifier":
        import joblib

        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


class ConsecutiveFilter:
    """Emit a command after ``xi`` equal consecutive outputs (once per streak).

    ``OTHER`` and "no hand" break the streak and are never emitted.
    """

    def __init__(self, xi: int = 5) -> None:
        self.xi = xi
        self.label: Optional[str] = None
        self.count = 0
        self.fired = False

    def reset(self) -> None:
        self.label, self.count, self.fired = None, 0, False

    def update(self, label: Optional[str]) -> Optional[str]:
        if label is None or label == OTHER:
            self.reset()
            return None
        if label == self.label:
            self.count += 1
        else:
            self.label, self.count, self.fired = label, 1, False
        if self.count >= self.xi and not self.fired:
            self.fired = True
            return label
        return None


@dataclass
class GestureResult:
    raw: Optional[str]              # per-frame SVM output (None = no hand)
    command: Optional[str]          # filtered command (WAIT / FOLLOW) or None
    hand: Optional[HandObservation]
    crop_origin: tuple[int, int] = (0, 0)


def _hand_process_main(conn, cfg: GestureConfig, static_images: bool, factory: str) -> None:
    """Child process: owns the MediaPipe graph; answers detect requests over a pipe."""
    import os
    try:                                        # keep MediaPipe off the main loop's cores
        n = os.cpu_count() or 1
        if n >= 4:
            os.sched_setaffinity(0, {n - 2, n - 1})
        os.nice(5)
    except (AttributeError, OSError):
        pass
    mod, name = factory.split(":")
    det = getattr(importlib.import_module(mod), name)(cfg, static_images=static_images)
    while True:
        try:
            img = conn.recv()
        except (EOFError, OSError):
            break
        if img is None:
            break
        try:
            conn.send(det.detect(img))
        except Exception:  # noqa: BLE001 - never kill the worker on one bad frame
            conn.send(None)


class ProcessHandDetector:
    """Same interface as HandLandmarkDetector, but MediaPipe runs in a child process
    ('spawn': a clean interpreter without CUDA state). detect() blocks only the calling
    thread, and waiting on the pipe releases the GIL, so the main loop keeps running."""

    def __init__(self, cfg: GestureConfig, static_images: bool = True,
                 factory: str = "followme.gestures:HandLandmarkDetector") -> None:
        ctx = mp_proc.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_hand_process_main, args=(child, cfg, static_images, factory),
                                 daemon=True)
        self._proc.start()
        self._lock = threading.Lock()
        LOGGER.info("MediaPipe hands running in a separate process (pid %d)", self._proc.pid)

    def detect(self, bgr: np.ndarray) -> Optional[HandObservation]:
        if bgr is None or bgr.size == 0:
            return None
        with self._lock:
            if not self._proc.is_alive():
                self._warn_dead()
                return None
            try:
                self._conn.send(np.ascontiguousarray(bgr))
                return self._conn.recv()
            except (EOFError, OSError):
                self._warn_dead()
                return None

    def _warn_dead(self) -> None:
        if not getattr(self, "_dead_warned", False):
            self._dead_warned = True
            LOGGER.warning("MediaPipe helper process is not running - gestures disabled until restart")

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (OSError, ValueError):
            pass
        self._proc.join(timeout=2)


class _FakeDetector:
    """Test helper for ProcessHandDetector (no MediaPipe needed): a 'hand' whose wrist is
    at the mean pixel colour, so results are easy to check."""

    def __init__(self, cfg, static_images=True):
        pass

    def detect(self, bgr):
        v = float(bgr.mean())
        return HandObservation(np.full((21, 3), v), "Right", 1.0)


class _GilHogDetector(_FakeDetector):
    """Test helper: pure-Python busy work (~0.3 s) that holds the GIL like a blocking extension."""

    def detect(self, bgr):
        x = 0
        for i in range(3_000_000):
            x += i
        return super().detect(bgr)


class AsyncGestureRecognizer:
    """Runs a GestureRecognizer in a background thread. update() never blocks: it hands the
    newest frame to the worker (if idle) and returns the latest finished result. A command
    (WAIT/FOLLOW) is returned exactly once. Other attributes delegate to the wrapped recognizer."""

    def __init__(self, inner: "GestureRecognizer") -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self._busy = False
        self._latest = GestureResult(None, None, None)
        self._pending_cmd: Optional[str] = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _work(self, frame, bbox) -> None:
        try:
            res = self.inner.update(frame, bbox)
            with self._lock:
                self._latest = res
                if res.command is not None:
                    self._pending_cmd = res.command
        finally:
            self._busy = False

    def update(self, frame: np.ndarray, target_bbox: Optional[np.ndarray]) -> GestureResult:
        if not self._busy:
            self._busy = True
            threading.Thread(target=self._work, args=(frame.copy(),
                             None if target_bbox is None else np.array(target_bbox, copy=True)),
                             daemon=True).start()
        with self._lock:
            cmd, self._pending_cmd = self._pending_cmd, None
            r = self._latest
        return GestureResult(r.raw, cmd, r.hand, r.crop_origin)


class GestureRecognizer:
    """Landmarks -> SVM -> xi filter, on the target person's crop."""

    def __init__(self, cfg: GestureConfig, classifier: GestureClassifier,
                 detector: Optional[HandLandmarkDetector] = None) -> None:
        self.cfg = cfg
        self.classifier = classifier
        self.detector = detector or HandLandmarkDetector(cfg, static_images=cfg.only_target_person)
        self.filter = ConsecutiveFilter(cfg.consecutive_required)

    def target_crop(self, frame: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pw, ph = (x2 - x1) * self.cfg.crop_padding, (y2 - y1) * self.cfg.crop_padding
        h, w = frame.shape[:2]
        xa, ya = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        xb, yb = min(w, int(x2 + pw)), min(h, int(y2 + ph))
        return frame[ya:yb, xa:xb], (xa, ya)

    def update(self, frame: np.ndarray, target_bbox: Optional[np.ndarray]) -> GestureResult:
        if self.cfg.only_target_person:
            if target_bbox is None:
                self.filter.reset()
                return GestureResult(None, None, None)
            image, origin = self.target_crop(frame, target_bbox)
        else:
            image, origin = frame, (0, 0)
        hand = self.detector.detect(image)
        raw = None
        if hand is not None:
            raw = self.classifier.predict_one(
                landmark_features(hand.landmarks, hand.handedness, self.cfg.mirror_left_hands))
        return GestureResult(raw, self.filter.update(raw), hand, origin)

