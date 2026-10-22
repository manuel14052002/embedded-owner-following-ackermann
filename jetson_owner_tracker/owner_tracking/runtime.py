"""Device selection and reproducibility helpers (torch is optional)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Return ``cuda:N`` if requested/available, otherwise ``cpu``."""
    try:
        import torch  # noqa: WPS433 (optional dependency)
    except ImportError:
        if requested.startswith("cuda"):
            LOGGER.warning("PyTorch not installed; falling back to CPU")
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return "cuda:0" if requested == "cuda" else requested


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if present) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # faster, non-deterministic kernels
