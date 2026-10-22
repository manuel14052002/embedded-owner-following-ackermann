"""TensorRT Re-ID backend for NVIDIA Jetson.

Wraps a pre-built ``.engine`` (OSNet exported to TensorRT FP16) behind the
:class:`PersonReID` interface. Two reasons this exists rather than using
``TorchreidReID``:

* **Correct weights.** Without ``reid.model_path`` torchreid falls back to
  ImageNet classification weights, which are not Re-ID trained and carry
  almost no identity information. The engine was built from a proper Re-ID
  checkpoint.
* **Latency.** Measured 6.2 ms per crop on a Jetson Orin Nano at 15 W,
  against 9-42 ms (and unstable) for the PyTorch path.

I/O buffers are allocated as torch CUDA tensors and execution is driven on
``torch.cuda.current_stream()``. Using pycuda instead creates a second CUDA
context that conflicts with the one PyTorch already owns, which produced
``Cask convolution execution`` errors and multi-second stalls.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from owner_tracking.config import ReIDConfig
from owner_tracking.reid import PersonReID, l2_normalize

LOGGER = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TensorRTReID(PersonReID):
    def __init__(self, config: ReIDConfig, device: str) -> None:
        import tensorrt as trt
        import torch

        if not config.model_path:
            raise ValueError("reid.model_path must point to a .engine file")
        self._torch = torch
        self.config = config
        self.h, self.w = config.input_height, config.input_width

        logger = trt.Logger(trt.Logger.WARNING)
        with open(config.model_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Could not deserialize engine {config.model_path}")
        self._engine = engine
        self._ctx = engine.create_execution_context()

        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self._in_name = next(n for n in names
                             if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self._out_name = next(n for n in names
                              if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

        out_shape = tuple(engine.get_tensor_shape(self._out_name))
        self.embedding_dim = int(out_shape[-1])

        self._in_t = torch.zeros((1, 3, self.h, self.w), dtype=torch.float32, device="cuda")
        self._out_t = torch.zeros((1, self.embedding_dim), dtype=torch.float32, device="cuda")
        self._ctx.set_input_shape(self._in_name, (1, 3, self.h, self.w))
        self._ctx.set_tensor_address(self._in_name, self._in_t.data_ptr())
        self._ctx.set_tensor_address(self._out_name, self._out_t.data_ptr())

        # Warm up: the first enqueue selects kernels and can take seconds,
        # which would otherwise land in the middle of owner selection.
        self._ctx.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        LOGGER.info("Loaded TensorRT Re-ID %s (dim=%d) %dx%d",
                    config.model_path, self.embedding_dim, self.w, self.h)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        torch = self._torch
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        out = np.zeros((len(bboxes), self.embedding_dim), dtype=np.float32)
        stream = torch.cuda.current_stream()

        for i, bbox in enumerate(bboxes):
            crop = self.crop(frame, bbox)
            if crop is None or crop.size == 0:
                continue          # zero row marks an invalid crop, per interface
            self._in_t.copy_(torch.from_numpy(self._preprocess(crop)).cuda())
            self._ctx.execute_async_v3(stream_handle=stream.cuda_stream)
            stream.synchronize()
            out[i] = self._out_t[0].cpu().numpy()

        return l2_normalize(out)
