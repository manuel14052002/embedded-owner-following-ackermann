"""Appearance embedding (person Re-ID) backends.

All backends return L2-normalised float32 embeddings, so cosine similarity is a
dot product. An all-zero row marks an invalid crop (e.g. out of frame).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from owner_tracking.config import ReIDConfig
from owner_tracking.geometry import clip_to_frame

LOGGER = logging.getLogger(__name__)


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, eps)).astype(np.float32)


class PersonReID(ABC):
    """Interface for appearance embedding extractors."""

    embedding_dim: int = 0

    def extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Embedding for a single box."""
        return self.extract_embeddings(frame, np.asarray(bbox, np.float32).reshape(1, 4))[0]

    @abstractmethod
    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Return ``(N, D)`` L2-normalised embeddings for ``(N, 4)`` boxes."""

    @staticmethod
    def crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Zero-copy view of the box region, or None if it is degenerate."""
        coords = clip_to_frame(bbox, frame.shape[1], frame.shape[0], min_size=4)
        if coords is None:
            return None
        x1, y1, x2, y2 = coords
        return frame[y1:y2, x1:x2]


class HistogramReID(PersonReID):
    """Cheap, deterministic colour descriptor (fallback and test backend).

    Horizontal stripes (upper / lower body) each contribute a masked
    hue-saturation histogram plus a value histogram; square-rooting the
    normalised histograms makes cosine similarity equal to the Bhattacharyya
    coefficient. This is *not* a learned Re-ID model and is far less
    discriminative than OSNet; use it only when no network is available.
    """

    def __init__(self, stripes: int = 2, hue_bins: int = 16, sat_bins: int = 4,
                 val_bins: int = 8, sat_min: int = 30, val_min: int = 30,
                 value_weight: float = 0.4, inner_width: float = 0.8) -> None:
        self.stripes = stripes
        self.hue_bins, self.sat_bins, self.val_bins = hue_bins, sat_bins, val_bins
        self.sat_min, self.val_min = sat_min, val_min
        self.value_weight = value_weight
        self.inner_width = inner_width
        self.embedding_dim = stripes * (hue_bins * sat_bins + val_bins)

    def _describe(self, crop: np.ndarray) -> np.ndarray:
        width = crop.shape[1]
        margin = int(width * (1.0 - self.inner_width) * 0.5)
        if width - 2 * margin >= 4:
            crop = crop[:, margin:width - margin]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        height = hsv.shape[0]
        parts: list[np.ndarray] = []
        for s in range(self.stripes):
            stripe = hsv[s * height // self.stripes:(s + 1) * height // self.stripes]
            mask = ((stripe[..., 1] >= self.sat_min) & (stripe[..., 2] >= self.val_min)).astype(np.uint8)
            hs = cv2.calcHist([stripe], [0, 1], mask, [self.hue_bins, self.sat_bins],
                              [0, 180, 0, 256]).ravel()
            val = cv2.calcHist([stripe], [2], None, [self.val_bins], [0, 256]).ravel()
            hs = np.sqrt(hs / hs.sum()) if hs.sum() > 0 else hs
            val = np.sqrt(val / val.sum()) if val.sum() > 0 else val
            parts.extend([hs, self.value_weight * val])
        return np.concatenate(parts).astype(np.float32)

    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(bboxes), self.embedding_dim), dtype=np.float32)
        for i, box in enumerate(bboxes):
            crop = self.crop(frame, box)
            if crop is not None:
                out[i] = self._describe(crop)
        return l2_normalize(out)


class _TorchBatchMixin:
    """Shared crop preprocessing for torch backends."""

    input_height: int
    input_width: int

    def _prepare(self, frame: np.ndarray, bboxes: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        crops, valid = [], []
        for i, box in enumerate(bboxes):
            crop = PersonReID.crop(frame, box)
            if crop is None:
                continue
            resized = cv2.resize(crop, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
            crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            valid.append(i)
        return crops, valid


class TorchreidReID(_TorchBatchMixin, PersonReID):
    """OSNet (or any torchreid model) via ``torchreid``'s FeatureExtractor.

    For real Re-ID performance ``model_path`` must point to Re-ID-trained
    weights (e.g. ``osnet_x0_25_msmt17.pth`` or ``osnet_ain_x1_0_msmt17.pth``
    from the torchreid model zoo). Without it only ImageNet weights are used.
    """

    def __init__(self, config: ReIDConfig, device: str) -> None:
        try:
            from torchreid.utils import FeatureExtractor  # torchreid >= 1.0
        except ImportError:
            try:
                from torchreid.reid.utils import FeatureExtractor  # newer layout
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("torchreid is not installed") from exc
        if not config.model_path:
            LOGGER.warning("reid.model_path is empty: torchreid will use ImageNet weights, "
                           "which are NOT Re-ID trained. Set a Re-ID checkpoint for research use.")
        self.config = config
        self.input_height, self.input_width = config.input_height, config.input_width
        self.extractor = FeatureExtractor(
            model_name=config.model_name,
            model_path=config.model_path,
            image_size=(config.input_height, config.input_width),
            device=device,
        )
        probe = self.extractor([np.zeros((config.input_height, config.input_width, 3), np.uint8)])
        self.embedding_dim = int(probe.shape[1])
        LOGGER.info("Loaded torchreid %s (dim=%d) on %s", config.model_name, self.embedding_dim, device)

    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(bboxes), self.embedding_dim), dtype=np.float32)
        crops, valid = self._prepare(frame, bboxes)
        bs = max(1, self.config.batch_size)
        for start in range(0, len(crops), bs):
            feats = self.extractor(crops[start:start + bs]).detach().cpu().numpy()
            out[valid[start:start + bs]] = feats
        return l2_normalize(out)


class TorchvisionReID(_TorchBatchMixin, PersonReID):
    """Generic ImageNet MobileNetV3 features (weak fallback, not Re-ID trained)."""

    def __init__(self, config: ReIDConfig, device: str) -> None:
        import torch
        import torchvision

        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.half = bool(config.half_precision and device.startswith("cuda"))
        self.input_height, self.input_width = config.input_height, config.input_width
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_large(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval().to(self.device)
        if self.half:
            model.half()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.embedding_dim = 960
        LOGGER.warning("Using generic ImageNet features for Re-ID; identity discrimination is limited")

    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(bboxes), self.embedding_dim), dtype=np.float32)
        crops, valid = self._prepare(frame, bboxes)
        bs = max(1, self.config.batch_size)
        with torch.inference_mode():
            for start in range(0, len(crops), bs):
                batch = np.stack(crops[start:start + bs])
                tensor = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
                tensor = (tensor - self.mean) / self.std
                if self.half:
                    tensor = tensor.half()
                feats = self.model(tensor).float().cpu().numpy()
                out[valid[start:start + bs]] = feats
        return l2_normalize(out)
