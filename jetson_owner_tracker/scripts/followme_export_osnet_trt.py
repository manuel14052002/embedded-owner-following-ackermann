#!/usr/bin/env python3
"""Export a torchreid model (OSNet...) to ONNX and build a TensorRT FP16 engine (Jetson).

    python3 scripts/followme_export_osnet_trt.py --model osnet_x1_0 \
        --weights models/osnet_x1_0_msmt17.pth --out models/osnet_x1_0.engine

Then in the config:
    reid:
      backend: tensorrt
      model_path: models/osnet_x1_0.engine

The script ends with a speed comparison (PyTorch vs TensorRT) and a check that both
give the same features. Building the engine takes 2-10 minutes on an Orin Nano.
If the TensorRT Python module is missing, the ONNX file is still written and the
command for trtexec is printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export_onnx(model_name: str, weights: str, onnx_path: str, height: int, width: int):
    import torch
    import torchreid
    from torchreid.utils import load_pretrained_weights

    model = torchreid.models.build_model(model_name, num_classes=1, pretrained=False)
    load_pretrained_weights(model, weights)
    model.eval()
    dummy = torch.zeros(1, 3, height, width)
    torch.onnx.export(model, dummy, onnx_path, input_names=["images"], output_names=["features"],
                      dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}}, opset_version=17)
    print(f"ONNX written: {onnx_path}")
    return model


def build_engine(onnx_path: str, engine_path: str, height: int, width: int, max_batch: int,
                 fp16: bool = True) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    try:
        network = builder.create_network(0)                       # TensorRT >= 10
    except TypeError:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise SystemExit(1)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, height, width), (min(4, max_batch), 3, height, width),
                      (max_batch, 3, height, width))
    config.add_optimization_profile(profile)
    print("building TensorRT engine (this takes a few minutes)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    Path(engine_path).write_bytes(bytes(serialized))
    print(f"engine written: {engine_path} ({Path(engine_path).stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


def benchmark(torch_model, engine_path: str, height: int, width: int) -> None:
    import torch

    from followme.config import ReIDConfig
    from followme.reid import TensorRTReID, preprocess

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (300, 120, 3), dtype=np.uint8) for _ in range(4)]
    trt_ext = TensorRTReID(ReIDConfig(backend="tensorrt", model_path=engine_path, l2_normalize=False), "cuda:0")
    torch_model = torch_model.cuda().eval()
    x = torch.from_numpy(preprocess(crops, height, width)).cuda()
    with torch.inference_mode():
        ref = torch_model(x).float().cpu().numpy()
    got = trt_ext.extract(crops)
    cos = np.sum(ref * got, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    print(f"feature agreement PyTorch vs TensorRT (cosine): {cos.min():.4f} .. {cos.max():.4f}  (>= 0.99 is fine)")

    def bench(fn, n=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    with torch.inference_mode():
        print(f"PyTorch  : {bench(lambda: torch_model(x)):.1f} ms for 4 people")
    print(f"TensorRT : {bench(lambda: trt_ext.extract(crops)):.1f} ms for 4 people (incl. preprocessing)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="osnet_x1_0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="models/osnet_x1_0.engine")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    onnx_path = str(Path(args.out).with_suffix(".onnx"))
    model = export_onnx(args.model, args.weights, onnx_path, args.height, args.width)
    try:
        import tensorrt
        _ = tensorrt
    except ImportError:
        print("\nTensorRT Python module not found. Build the engine with trtexec instead:\n"
              f"  /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --fp16 "
              f"--minShapes=images:1x3x{args.height}x{args.width} "
              f"--optShapes=images:4x3x{args.height}x{args.width} "
              f"--maxShapes=images:{args.max_batch}x3x{args.height}x{args.width} --saveEngine={args.out}")
        return 0
    build_engine(onnx_path, args.out, args.height, args.width, args.max_batch, fp16=not args.no_fp16)
    benchmark(model, args.out, args.height, args.width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
