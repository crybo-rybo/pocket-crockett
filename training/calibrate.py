#!/usr/bin/env python3
"""Temperature scaling on the calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.label_map import load_label_map, scientific_name_to_index
from training.common.metrics import collect_logits, fit_temperature, recommend_confidence_floor, topk_accuracy
from training.datasets.folder_dataset import DEFAULT_SPLITS_DIR, build_dataloader, load_split_records
from training.evaluate import load_model_from_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, config = load_model_from_run(args.run_dir, device)
    name_to_index = scientific_name_to_index(label_map)

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))

    records = load_split_records("calibration", args.data_root, args.splits_dir, name_to_index)
    if not records:
        raise SystemExit(f"No calibration records found under {args.data_root}")

    loader = build_dataloader(
        records,
        args.data_root,
        batch_size=batch_size,
        image_size=image_size,
        augment=False,
        num_workers=num_workers,
        shuffle=False,
        weighted_sampler=False,
    )
    logits, labels = collect_logits(model, loader, device)
    temperature = fit_temperature(logits, labels)
    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=1)
    confidence_floor = recommend_confidence_floor(probs, labels)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature,
        "confidence_floor_recommendation": confidence_floor,
        "calibration_split": "calibration",
        "num_samples": len(labels),
        "top1_before": topk_accuracy(logits, labels, k=1),
        "top1_after": topk_accuracy(scaled, labels, k=1),
        "notes": (
            "Use temperature with evaluate.py and runtime inference. "
            "If max calibrated probability is below confidence_floor_recommendation, return unknown."
        ),
    }
    out_path = args.run_dir / "calibration.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"calibration": str(out_path), **payload}, indent=2))


if __name__ == "__main__":
    main()
