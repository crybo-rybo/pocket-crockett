#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX for downstream Jetson conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.evaluate import load_model_from_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _label_map, config = load_model_from_run(args.run_dir, device)
    image_size = int(config.get("data", {}).get("image_size", 224))
    output_path = args.output or (args.run_dir / "model.onnx")

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    meta = {
        "onnx_path": str(output_path),
        "image_size": image_size,
        "run_dir": str(args.run_dir),
    }
    (args.run_dir / "export_onnx.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
