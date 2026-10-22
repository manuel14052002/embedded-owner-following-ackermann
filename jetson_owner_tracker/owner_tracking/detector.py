"""Person detectors. Any class implementing :class:`PersonDetector` can be used."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from owner_tracking.config import DetectorConfig
from owner_tracking.datatypes import Detection

LOGGER = logging.getLogger(__name__)


class PersonDetector(ABC):
    """Interface: return person detections for a BGR frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in ``frame`` (H x W x 3, BGR, uint8)."""


class UltralyticsPersonDetector(PersonDetector):
    """YOLO (v8/11/...) person detector via the ``ultralytics`` package.

    ``model_path`` may point to ``.pt`` weights, an ONNX file or a TensorRT
    ``.engine`` exported with ``yolo export model=... format=engine half=True``
    (recommended on NVIDIA Jetson).
    """

    def __init__(self, config: DetectorConfig, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("ultralytics is not installed: pip install ultralytics") from exc
        if not config.model_path:
            raise ValueError("detector.model_path must be set")
        self.config = config
        self.device = device
        self.half = bool(config.half_precision and device.startswith("cuda"))
        self.model = YOLO(config.model_path)
        LOGGER.info("Loaded detector %s on %s (half=%s)", config.model_path, device, self.half)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.config
        results = self.model.predict(
            frame,
            classes=[cfg.person_class_id],
            conf=cfg.confidence_threshold,
            iou=cfg.nms_iou_threshold,
            imgsz=cfg.image_size,
            device=self.device,
            half=self.half,
            max_det=cfg.max_detections,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        heights = xyxy[:, 3] - xyxy[:, 1]
        keep = heights >= cfg.min_box_height
        return [Detection(bbox=b, confidence=float(c)) for b, c in zip(xyxy[keep], conf[keep])]
