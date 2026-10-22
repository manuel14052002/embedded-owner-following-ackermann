"""Person detection by instance segmentation (paper Sect. III-A).

Each detected person yields a box and a binary mask. The mask serves twice:
the Re-ID crop is the image element-wise multiplied by the mask (background
removed, Fig. 1b-c) and the mask selects the target's depth pixels for 3-D
localisation (Sect. III-C, Fig. 1d).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from followme.config import SegmentationConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class PersonInstance:
    """One segmented person.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in pixels (float32).
        mask: full-frame boolean mask ``(H, W)``.
        score: detector confidence.
    """

    bbox: np.ndarray
    mask: np.ndarray
    score: float

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.score = float(self.score)

    def int_box(self, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self.bbox
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return x1, y1, x2, y2


def masked_crop(frame: np.ndarray, person: PersonInstance,
                remove_background: bool = True) -> Optional[np.ndarray]:
    """Box crop of ``frame`` with background pixels set to 0 (Sect. III-A).

    Returns a new array (the frame is never modified) or None for a
    degenerate box.
    """
    box = person.int_box(frame.shape[1], frame.shape[0])
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if not remove_background:
        return crop.copy()
    mask = person.mask[y1:y2, x1:x2]
    return crop * mask[..., None].astype(crop.dtype)


class PersonSegmenter(ABC):
    """Interface: BGR frame -> list of :class:`PersonInstance`."""

    @abstractmethod
    def segment(self, frame: np.ndarray) -> list[PersonInstance]:
        """Segment people in an ``H x W x 3`` BGR uint8 frame."""


def _filter(instances: list[PersonInstance], cfg: SegmentationConfig) -> list[PersonInstance]:
    kept = []
    for inst in instances:
        h = inst.bbox[3] - inst.bbox[1]
        w = inst.bbox[2] - inst.bbox[0]
        if h < cfg.min_box_height or w <= 0:
            continue
        if inst.mask.sum() < cfg.min_mask_fraction * w * h:
            continue
        kept.append(inst)
    return kept


class UltralyticsSegmenter(PersonSegmenter):
    """YOLOv8/YOLO11 ``-seg`` models (``.pt``, ``.onnx`` or TensorRT ``.engine``).

    Drop-in replacement for YOLACT++: COCO-pretrained, class 0 = person,
    returns boxes and instance masks at the original image resolution
    (``retina_masks=True``).
    """

    def __init__(self, config: SegmentationConfig, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on host
            raise RuntimeError("ultralytics is not installed: pip install ultralytics") from exc
        self.config = config
        self.device = device
        self.half = bool(config.half_precision and device.startswith("cuda"))
        self.model = YOLO(config.model_path, task="segment")
        LOGGER.info("Loaded segmenter %s on %s (half=%s)", config.model_path, device, self.half)

    def segment(self, frame: np.ndarray) -> list[PersonInstance]:  # pragma: no cover - needs model
        cfg = self.config
        self.last_obstacles = []
        results = self.model.predict(frame, classes=[cfg.person_class_id] + [int(c) for c in cfg.obstacle_class_ids],
                                     conf=min([cfg.confidence_threshold] + ([cfg.obstacle_confidence] if cfg.obstacle_class_ids else [])),
                                     iou=cfg.nms_iou_threshold, imgsz=cfg.image_size, device=self.device,
                                     half=self.half, max_det=cfg.max_detections, retina_masks=True,
                                     verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0 or results[0].masks is None:
            return []
        res = results[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        scores = res.boxes.conf.detach().cpu().numpy()
        clss = res.boxes.cls.detach().cpu().numpy().astype(int)
        masks = res.masks.data.detach().cpu().numpy() > 0.5
        h, w = frame.shape[:2]
        out = []
        obst = []
        for box, score, mask, cls in zip(boxes, scores, masks, clss):
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if cls != cfg.person_class_id:
                if score >= cfg.obstacle_confidence:
                    obst.append(ObstacleInstance(int(cls), box, mask, float(score)))
                continue
            if score < cfg.confidence_threshold:
                continue
            out.append(PersonInstance(box, mask, float(score)))
        self.last_obstacles = obst
        return _filter(out, cfg)


def build_segmenter(config: SegmentationConfig, device: str) -> PersonSegmenter:
    if config.backend == "ultralytics":
        return UltralyticsSegmenter(config, device)
    raise ValueError(f"Unknown segmentation backend '{config.backend}'")


class StaticSegmenter(PersonSegmenter):
    """Returns pre-computed instances (tests, offline datasets with GT masks)."""

    def __init__(self) -> None:
        self.next: list[PersonInstance] = []

    def segment(self, frame: np.ndarray) -> list[PersonInstance]:
        return list(self.next)


COCO_OBSTACLE_NAMES = {13: "bench", 56: "chair", 57: "couch", 58: "plant", 59: "bed", 60: "table"}


@dataclass
class ObstacleInstance:
    """A non-person object from the same YOLO run (chair, table ...) used as an obstacle."""

    class_id: int
    bbox: np.ndarray
    mask: np.ndarray
    score: float

    @property
    def name(self) -> str:
        return COCO_OBSTACLE_NAMES.get(self.class_id, f"class {self.class_id}")
