#!/usr/bin/env python3
"""Evaluate a trained classifier on val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.label_map import load_label_map, scientific_name_to_index
from training.common.metrics import (
    collect_logits,
    confusion_pairs,
    macro_recall,
    per_class_recall,
    topk_accuracy,
    topn_candidates,
)
from training.datasets.folder_dataset import DEFAULT_SPLITS_DIR, build_dataloader, load_split_records
from training.models.bioclip_classifier import build_bioclip_classifier
from training.models.classifier import build_classifier


def disable_pretrained_downloads(config: dict) -> dict:
    """Return a build config that reconstructs architecture without fetching weights."""
    build_config = dict(config)
    model_cfg = dict(build_config.get("model", {}))
    model_cfg["pretrained"] = False
    build_config["model"] = model_cfg
    return build_config


def load_model_from_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    checkpoint_path = run_dir / "checkpoint-best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint-last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    label_map_path = run_dir / "label_map.json"
    label_map = load_label_map(label_map_path if label_map_path.exists() else None)
    num_classes = int(label_map["num_classes"])
    model_type = config.get("model", {}).get("type", "timm")
    build_config = disable_pretrained_downloads(config)
    if model_type == "bioclip":
        model = build_bioclip_classifier(build_config, num_classes)
    else:
        model = build_classifier(build_config, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, label_map, config


def load_temperature(run_dir: Path) -> float:
    calibration_path = run_dir / "calibration.json"
    if not calibration_path.exists():
        return 1.0
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return float(data.get("temperature", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, config = load_model_from_run(args.run_dir, device)
    temperature = load_temperature(args.run_dir)
    name_to_index = scientific_name_to_index(label_map)
    num_classes = int(label_map["num_classes"])

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "temperature": temperature,
        "splits": {},
    }

    for split in args.splits:
        records = load_split_records(split, args.data_root, args.splits_dir, name_to_index)
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
        scaled = logits / max(temperature, 1e-6)
        class_recall = per_class_recall(scaled, labels, num_classes)
        index_to_name = {c["class_index"]: c["scientific_name"] for c in label_map["classes"]}
        per_class_named = {
            index_to_name[idx]: value for idx, value in class_recall.items() if idx in index_to_name
        }
        report["splits"][split] = {
            "num_samples": len(labels),
            "top1": topk_accuracy(scaled, labels, k=1),
            "top5": topk_accuracy(scaled, labels, k=min(5, num_classes)),
            "macro_recall": macro_recall(scaled, labels, num_classes),
            "per_class_recall": per_class_named,
            "top_confusions": [
                {
                    "true": index_to_name.get(true_idx, str(true_idx)),
                    "pred": index_to_name.get(pred_idx, str(pred_idx)),
                    "count": count,
                }
                for (true_idx, pred_idx), count in sorted(
                    confusion_pairs(scaled, labels).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
            "topn_example": topn_candidates(scaled[: min(3, len(scaled))], label_map, n=args.top_n, temperature=1.0),
        }

    out_json = args.run_dir / "eval_report.json"
    out_md = args.run_dir / "eval_report.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Run dir: `{args.run_dir}`",
        f"Temperature: {temperature}",
        "",
    ]
    for split, metrics in report["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Samples: {metrics['num_samples']}",
                f"- Top-1: {metrics['top1']:.4f}",
                f"- Top-5: {metrics['top5']:.4f}",
                f"- Macro recall: {metrics['macro_recall']:.4f}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"eval_report": str(out_json), "eval_markdown": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
