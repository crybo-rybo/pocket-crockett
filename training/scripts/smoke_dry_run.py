#!/usr/bin/env python3
"""Validate training plumbing without unpacked images (config, label map, model forward)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.config import load_config
from training.common.label_map import load_label_map, scientific_name_to_index
from training.datasets.folder_dataset import DEFAULT_SPLITS_DIR, load_split_records
from training.models.classifier import build_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/smoke_test.yaml")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Allow pretrained weight downloads; default dry-run validates architecture only.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    build_config = dict(config)
    model_cfg = dict(build_config.get("model", {}))
    if not args.pretrained:
        model_cfg["pretrained"] = False
    build_config["model"] = model_cfg
    label_map = load_label_map()
    name_to_index = scientific_name_to_index(label_map)

    # Parse split CSVs without requiring image files on disk.
    train_records = load_split_records(
        "train",
        ROOT / "vision",
        DEFAULT_SPLITS_DIR,
        name_to_index,
        require_existing=False,
    )
    val_records = load_split_records(
        "val",
        ROOT / "vision",
        DEFAULT_SPLITS_DIR,
        name_to_index,
        require_existing=False,
    )

    model = build_classifier(build_config, label_map["num_classes"])
    model.eval()
    image_size = int(build_config.get("data", {}).get("image_size", 224))
    dummy = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        logits = model(dummy)
        features = model.features(dummy)

    if features.ndim != 2 or features.shape[0] != dummy.shape[0]:
        raise RuntimeError(f"features() returned unexpected shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise RuntimeError("features() returned non-finite values")

    report = {
        "status": "ok",
        "config": str(args.config),
        "model_name": model_cfg.get("name"),
        "pretrained": bool(model_cfg.get("pretrained", False)),
        "num_classes": label_map["num_classes"],
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "logits_shape": list(logits.shape),
        "features_shape": list(features.shape),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
