#!/usr/bin/env python3
"""Export an MMT IBN-ResNet-50 checkpoint to ONNX (for onnxruntime / TensorRT on Jetson).

    python scripts/followme_export_mmt_onnx.py model_best.pth.tar models/mmt_ibn50a.onnx
    # optional, on the Jetson:
    /usr/src/tensorrt/bin/trtexec --onnx=models/mmt_ibn50a.onnx --fp16 \
        --minShapes=images:1x3x256x128 --optShapes=images:4x3x256x128 --maxShapes=images:16x3x256x128 \
        --saveEngine=models/mmt_ibn50a.engine

Then set ``reid.backend: onnx`` and ``reid.model_path: models/mmt_ibn50a.onnx``.
The exported network already L2-normalises its output (MMT eval mode).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from followme.reid import export_mmt_onnx  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    print(f"exported D={export_mmt_onnx(sys.argv[1], sys.argv[2])} -> {sys.argv[2]}")
