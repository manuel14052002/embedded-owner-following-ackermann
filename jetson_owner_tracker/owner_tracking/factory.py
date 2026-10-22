"""Model loading with explicit, logged fallbacks."""

from __future__ import annotations

import logging

from owner_tracking.config import DetectorConfig, ReIDConfig
from owner_tracking.detector import PersonDetector, UltralyticsPersonDetector
from owner_tracking.reid import HistogramReID, PersonReID, TorchreidReID, TorchvisionReID

LOGGER = logging.getLogger(__name__)

_REID_BACKENDS = {
    "torchreid": lambda cfg, dev: TorchreidReID(cfg, dev),
    "torchvision": lambda cfg, dev: TorchvisionReID(cfg, dev),
    "histogram": lambda cfg, dev: HistogramReID(),
}


def build_detector(config: DetectorConfig, device: str) -> PersonDetector:
    """Instantiate the configured detector."""
    if config.backend == "ultralytics":
        return UltralyticsPersonDetector(config, device)
    raise ValueError(f"Unknown detector backend '{config.backend}'")


def build_reid(config: ReIDConfig, device: str) -> PersonReID:
    """Instantiate the configured Re-ID backend, trying fallbacks in order."""
    order = [config.backend] + [b for b in config.fallback_backends if b != config.backend]
    errors: list[str] = []
    for name in order:
        factory = _REID_BACKENDS.get(name)
        if factory is None:
            errors.append(f"{name}: unknown backend")
            continue
        try:
            model = factory(config, device)
        except Exception as exc:  # noqa: BLE001 - we want to report and fall back
            LOGGER.warning("Re-ID backend '%s' failed to load: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue
        if name != config.backend:
            LOGGER.warning("Using fallback Re-ID backend '%s'", name)
        return model
    raise RuntimeError("No Re-ID backend could be loaded: " + "; ".join(errors))
