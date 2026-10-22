"""Re-ID feature extraction on background-removed crops (paper Sect. III-B).

The paper uses the *pre-training* stage of Mutual Mean-Teaching (MMT,
Ge et al., ICLR 2020): an IBN-ResNet-50 trained on MSMT17 with a 256-d
embedding. :class:`MMTIBNResNet50` reproduces the MMT ``ResNetIBN('50a')``
module *key for key*, so a checkpoint produced by the MMT repository
(``examples/pretrain.py ... -a resnet_ibn50a --features 256``, saved as
``model_best.pth.tar``) loads directly. The embedding size is read from the
checkpoint (``feat.weight``), so 2048-d checkpoints trained with
``--features 0`` also work.

Backends:
    * ``mmt``        - PyTorch, the paper's network.
    * ``onnx``       - the same network exported with :func:`export_mmt_onnx`,
                       run with onnxruntime (TensorRT EP on Jetson).
    * ``torchreid``  - any torchreid model (OSNet...), for comparison.
    * ``histogram``  - colour histograms, dependency-free; tests only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from followme.config import ReIDConfig

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(crops: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    """BGR uint8 crops -> ``(N, 3, H, W)`` float32, RGB, ImageNet-normalised
    (MMT test transform: Resize((256, 128)), ToTensor, Normalize)."""
    batch = np.empty((len(crops), 3, height, width), dtype=np.float32)
    for i, crop in enumerate(crops):
        rgb = cv2.cvtColor(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch[i] = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return batch


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)).astype(np.float32)


class FeatureExtractor(ABC):
    """Crops (background already removed) -> ``(N, D)`` float32 features."""

    feature_dim: int = 0

    @abstractmethod
    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return one feature row per crop."""


# --------------------------------------------------------------------------
# MMT IBN-ResNet-50 (PyTorch)
# --------------------------------------------------------------------------

def build_mmt_ibn_resnet50(num_features: int = 256, last_stride: int = 1) -> Any:
    """Build the MMT ``ResNetIBN('50a', num_features=D)`` network.

    Structure and parameter names follow ``mmt/models/resnet_ibn.py`` and
    ``mmt/models/resnet_ibn_a.py`` of github.com/yxgeee/MMT:
    ``base`` = Sequential(conv1, bn1, relu, maxpool, layer1..layer4) with
    IBN-a blocks (InstanceNorm on half the channels of ``bn1`` in layers
    1-3), last stride 1, global average pooling, then ``feat`` (Linear
    2048->D) + ``feat_bn`` (BatchNorm1d) when D > 0, else ``feat_bn`` over the
    2048-d pooled vector. In eval mode the output is L2-normalised.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    class IBN(nn.Module):
        def __init__(self, planes: int) -> None:
            super().__init__()
            half1 = int(planes / 2)
            self.half = half1
            self.IN = nn.InstanceNorm2d(half1, affine=True)  # noqa: N815 - checkpoint key
            self.BN = nn.BatchNorm2d(planes - half1)          # noqa: N815 - checkpoint key

        def forward(self, x):
            split = torch.split(x, self.half, 1)
            return torch.cat((self.IN(split[0].contiguous()), self.BN(split[1].contiguous())), 1)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return self.relu(out + residual)

    def make_layer(inplanes, planes, blocks, stride):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))
        ibn = planes != 512
        layers = [Bottleneck(inplanes, planes, ibn, stride, downsample)]
        layers += [Bottleneck(planes * 4, planes, ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    class ResNetIBN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                make_layer(64, 64, 3, 1),
                make_layer(256, 128, 4, 2),
                make_layer(512, 256, 6, 2),
                make_layer(1024, 512, 3, last_stride),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.has_embedding = num_features > 0
            if self.has_embedding:
                self.feat = nn.Linear(2048, num_features)
                self.feat_bn = nn.BatchNorm1d(num_features)
            else:
                self.feat_bn = nn.BatchNorm1d(2048)
            self.num_features = num_features if self.has_embedding else 2048

        def forward(self, x):
            x = self.gap(self.base(x)).flatten(1)
            bn_x = self.feat_bn(self.feat(x)) if self.has_embedding else self.feat_bn(x)
            return F.normalize(bn_x) if not self.training else bn_x

    return ResNetIBN()


def load_mmt_checkpoint(path: str) -> tuple[dict, int]:
    """Load an MMT checkpoint; return ``(state_dict, num_features)``.

    Accepts ``{'state_dict': ...}`` files and bare state dicts, strips the
    ``module.`` prefix left by DataParallel and drops the training-only
    ``classifier`` weights.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for key, value in state.items():
        key = key[len("module."):] if key.startswith("module.") else key
        if key.startswith("classifier"):
            continue
        cleaned[key] = value
    num_features = int(cleaned["feat.weight"].shape[0]) if "feat.weight" in cleaned else 0
    return cleaned, num_features


class MMTReID(FeatureExtractor):
    """The paper's Re-ID network (PyTorch)."""

    def __init__(self, config: ReIDConfig, device: str) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        num_features = config.feature_dim
        state: Optional[dict] = None
        if config.model_path:
            state, num_features = load_mmt_checkpoint(config.model_path)
        else:
            LOGGER.warning("reid.model_path is empty: MMT network has RANDOM weights. "
                           "Train/download an MSMT17 MMT pre-training checkpoint.")
        model = build_mmt_ibn_resnet50(num_features)
        if state is not None:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"MMT checkpoint is missing {len(missing)} keys, e.g. {missing[:5]}")
            if unexpected:
                LOGGER.warning("Ignoring %d unexpected checkpoint keys, e.g. %s", len(unexpected), unexpected[:5])
        self.half = bool(config.half_precision and device.startswith("cuda"))
        model.eval().to(self.device)
        if self.half:
            model.half()
        self.model = model
        self.feature_dim = model.num_features
        if self.feature_dim != config.feature_dim:
            LOGGER.warning("Checkpoint feature dim %d != configured %d; using %d",
                           self.feature_dim, config.feature_dim, self.feature_dim)
        LOGGER.info("MMT IBN-ResNet-50 loaded (D=%d) on %s", self.feature_dim, device)

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        bs = max(1, self.config.batch_size)
        with torch.inference_mode():
            for s in range(0, len(crops), bs):
                x = torch.from_numpy(preprocess(crops[s:s + bs], self.config.input_height,
                                                self.config.input_width)).to(self.device)
                if self.half:
                    x = x.half()
                out[s:s + bs] = self.model(x).float().cpu().numpy()
        return _l2(out) if self.config.l2_normalize else out


def export_mmt_onnx(checkpoint: str, output: str, height: int = 256, width: int = 128,
                    opset: int = 17) -> int:
    """Export the MMT network to ONNX (dynamic batch). Returns D."""
    import torch

    state, num_features = load_mmt_checkpoint(checkpoint)
    model = build_mmt_ibn_resnet50(num_features)
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.zeros(1, 3, height, width)
    torch.onnx.export(model, dummy, output, input_names=["images"], output_names=["features"],
                      dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}}, opset_version=opset)
    return model.num_features


# --------------------------------------------------------------------------
# ONNX Runtime backend (Jetson / CPU)
# --------------------------------------------------------------------------

class OnnxReID(FeatureExtractor):
    """Runs an exported Re-ID network with onnxruntime.

    Input: ``(N, 3, H, W)`` float32 ImageNet-normalised RGB; output ``(N, D)``.
    Models with a fixed batch dimension are run one crop at a time.
    """

    def __init__(self, config: ReIDConfig, device: str) -> None:
        import onnxruntime as ort

        if not config.model_path:
            raise ValueError("reid.model_path must point to an .onnx file for the onnx backend")
        available = ort.get_available_providers()
        wanted = (["TensorrtExecutionProvider", "CUDAExecutionProvider"] if device.startswith("cuda") else [])
        providers = [p for p in wanted if p in available] + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(config.model_path, providers=providers)
        self.config = config
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.input_height = inp.shape[2] if isinstance(inp.shape[2], int) else config.input_height
        self.input_width = inp.shape[3] if isinstance(inp.shape[3], int) else config.input_width
        out_shape = self.session.get_outputs()[0].shape
        self.feature_dim = int(out_shape[1]) if isinstance(out_shape[1], int) else config.feature_dim
        LOGGER.info("ONNX Re-ID %s (D=%d) providers=%s", config.model_path, self.feature_dim,
                    self.session.get_providers())

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.feature_dim), np.float32)
        batch = preprocess(crops, self.input_height, self.input_width)
        step = self.fixed_batch or max(1, self.config.batch_size)
        outs = []
        for s in range(0, len(batch), step):
            chunk = batch[s:s + step]
            if self.fixed_batch and len(chunk) < self.fixed_batch:
                pad = np.zeros((self.fixed_batch - len(chunk),) + chunk.shape[1:], np.float32)
                res = self.session.run(None, {self.input_name: np.concatenate([chunk, pad])})[0][:len(chunk)]
            else:
                res = self.session.run(None, {self.input_name: chunk})[0]
            outs.append(np.asarray(res, np.float32).reshape(len(chunk), -1))
        out = np.concatenate(outs)
        return _l2(out) if self.config.l2_normalize else out


# --------------------------------------------------------------------------
# TensorRT backend (Jetson): engine built by scripts/followme_export_osnet_trt.py
# --------------------------------------------------------------------------

class TensorRTReID(FeatureExtractor):
    """Runs a TensorRT engine (input ``images`` NCHW float32, output ``features``).

    GPU buffers are PyTorch CUDA tensors, so no pycuda is needed; TensorRT 10 API
    (``execute_async_v3``). Falls back to batch-by-batch execution within the engine's
    optimisation profile (1..max batch).
    """

    def __init__(self, config: ReIDConfig, device: str) -> None:
        import tensorrt as trt
        import torch

        if not config.model_path:
            raise ValueError("reid.model_path must point to a .engine file for the tensorrt backend")
        self.torch, self.config = torch, config
        self.device = torch.device(device if device.startswith("cuda") else "cuda:0")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(config.model_path, "rb") as fh, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"could not load TensorRT engine {config.model_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        _, self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[0][0], \
            self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        shape = self.engine.get_tensor_shape(self.inp)
        self.input_height = int(shape[2]) if shape[2] > 0 else config.input_height
        self.input_width = int(shape[3]) if shape[3] > 0 else config.input_width
        self.context.set_input_shape(self.inp, (1, 3, self.input_height, self.input_width))
        self.feature_dim = int(self.context.get_tensor_shape(self.out)[1])
        self.stream = torch.cuda.Stream(device=self.device)
        LOGGER.info("TensorRT Re-ID %s (D=%d, max batch %d)", config.model_path, self.feature_dim, self.max_batch)

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        step = max(1, min(self.max_batch, self.config.batch_size))
        for s in range(0, len(crops), step):
            batch = preprocess(crops[s:s + step], self.input_height, self.input_width)
            n = len(batch)
            with torch.cuda.stream(self.stream):
                x = torch.from_numpy(batch).to(self.device, non_blocking=True).contiguous()
                y = torch.empty((n, self.feature_dim), dtype=torch.float32, device=self.device)
                self.context.set_input_shape(self.inp, (n, 3, self.input_height, self.input_width))
                self.context.set_tensor_address(self.inp, x.data_ptr())
                self.context.set_tensor_address(self.out, y.data_ptr())
                ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                if not ok:
                    raise RuntimeError("TensorRT execution failed")
            self.stream.synchronize()
            out[s:s + n] = y.cpu().numpy()
        return _l2(out) if self.config.l2_normalize else out


# --------------------------------------------------------------------------
# Adapters to the existing owner_tracking backends
# --------------------------------------------------------------------------

class _PersonReIDAdapter(FeatureExtractor):
    """Wraps an ``owner_tracking.reid.PersonReID`` so it runs on masked crops."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.feature_dim = backend.embedding_dim

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            box = np.array([[0, 0, crop.shape[1], crop.shape[0]]], np.float32)
            out[i] = self.backend.extract_embeddings(crop, box)[0]
        return out


class TorchreidNative(FeatureExtractor):
    """torchreid model (OSNet...) with fast NumPy preprocessing (no PIL on the CPU)."""

    def __init__(self, config: ReIDConfig, device: str) -> None:
        import torch
        import torchreid
        from torchreid.utils import load_pretrained_weights

        # cudnn.benchmark stays off: its workspace probing runs the Jetson out of memory
        torch.backends.cudnn.benchmark = False
        self.torch, self.config, self.device = torch, config, torch.device(device)
        model = torchreid.models.build_model(config.torchreid_model_name, num_classes=1, pretrained=False)
        if config.model_path:
            load_pretrained_weights(model, config.model_path)
        else:
            LOGGER.warning("reid.model_path empty: torchreid model has NO Re-ID weights")
        model.eval().to(self.device)
        self.half = bool(config.half_precision and device.startswith("cuda"))
        if self.half:
            model.half()
        self.model = model
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, config.input_height, config.input_width, device=self.device)
            self.feature_dim = int(model(dummy.half() if self.half else dummy).shape[1])
        LOGGER.info("torchreid %s (D=%d) on %s", config.torchreid_model_name, self.feature_dim, device)

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        bs = max(1, self.config.batch_size)
        with torch.inference_mode():
            for s in range(0, len(crops), bs):
                x = torch.from_numpy(preprocess(crops[s:s + bs], self.config.input_height,
                                                self.config.input_width)).to(self.device)
                if self.half:
                    x = x.half()
                out[s:s + bs] = self.model(x).float().cpu().numpy()
        return _l2(out) if self.config.l2_normalize else out


def build_feature_extractor(config: ReIDConfig, device: str) -> FeatureExtractor:
    """Instantiate the configured Re-ID backend."""
    if config.backend == "mmt":
        return MMTReID(config, device)
    if config.backend == "onnx":
        return OnnxReID(config, device)
    if config.backend == "tensorrt":
        return TensorRTReID(config, device)
    if config.backend == "torchreid":
        return TorchreidNative(config, device)
    if config.backend == "torchreid_extractor":
        from owner_tracking.config import ReIDConfig as OTReIDConfig
        from owner_tracking.reid import TorchreidReID
        ot = OTReIDConfig(backend="torchreid", model_name=config.torchreid_model_name,
                          model_path=config.model_path, input_height=config.input_height,
                          input_width=config.input_width, batch_size=config.batch_size)
        return _PersonReIDAdapter(TorchreidReID(ot, device))
    if config.backend == "histogram":
        from owner_tracking.reid import HistogramReID
        return _PersonReIDAdapter(HistogramReID())
    raise ValueError(f"Unknown Re-ID backend '{config.backend}'")
