#!/usr/bin/env python3
"""Evaluate a trained classifier on val/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.label_map import load_label_map, scientific_name_to_index
from training.common.metrics import (
    collect_logits,
    confusion_pairs,
    lookalike_confusion,
    macro_recall,
    per_class_recall,
    topk_accuracy,
    topn_candidates,
)
from training.datasets.folder_dataset import DEFAULT_SPLITS_DIR, build_dataloader, load_split_records
from training.models.bioclip_classifier import build_bioclip_classifier
from training.models.classifier import build_classifier

DEFAULT_EDIBILITY_CSV = ROOT / "vision" / "edibility" / "edibility_skeleton.csv"
DEFAULT_DANGER_EDGES_CSV = ROOT / "vision" / "safety" / "danger_edges.csv"


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
    return load_calibration(run_dir)["temperature"]


def load_calibration(run_dir: Path) -> dict[str, float]:
    calibration_path = run_dir / "calibration.json"
    if not calibration_path.exists():
        return {"temperature": 1.0, "confidence_floor": 0.0}
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return {
        "temperature": float(data.get("temperature", 1.0)),
        "confidence_floor": float(data.get("confidence_floor_recommendation", 0.0)),
    }


def load_danger_edges(path: Path, *, default_budget: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    edges: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            edge = dict(row)
            if edge.get("gate_budget") in {None, ""}:
                edge["gate_budget"] = default_budget
            else:
                edge["gate_budget"] = int(edge["gate_budget"])
            edges.append(edge)
    return edges


def build_danger_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_map: dict,
    *,
    edibility_csv: Path,
    danger_edges_csv: Path,
    confidence_floor: float,
    default_budget: int,
) -> dict[str, Any]:
    from safety.edibility import EdibilityTable

    table = EdibilityTable.load(edibility_csv)
    index_names = {str(c["scientific_name"]) for c in label_map.get("classes", [])}
    graph = table.lookalike_graph(index_names)
    edges = load_danger_edges(danger_edges_csv, default_budget=default_budget)
    report = lookalike_confusion(logits, labels, label_map, edges, floor=confidence_floor) if edges else {
        "confidence_floor": confidence_floor,
        "edges": [],
        "directed": {"count": 0, "confident_wrong": 0, "true_total": 0, "rate": 0.0, "confident_wrong_rate": 0.0},
        "dangerous_to_foraging": {
            "count": 0,
            "confident_wrong": 0,
            "true_total": 0,
            "rate": 0.0,
            "confident_wrong_rate": 0.0,
            "release_gate": "not_applicable",
            "failed_edges": [],
        },
        "release_gate": "not_applicable",
    }
    report["graph_edge_count"] = sum(len(targets) for targets in graph.values())
    report["danger_edge_count"] = len(edges)
    report["dangerous_to_foraging_misID_rate"] = report["dangerous_to_foraging"]["confident_wrong_rate"]
    return report


def ood_artifact_status(run_dir: Path) -> dict[str, Any]:
    ood_json = run_dir / "ood.json"
    ood_stats = run_dir / "ood_stats.pt"
    if not ood_json.exists() or not ood_stats.exists():
        return {
            "status": "missing",
            "release_gate": "not_applicable",
            "missing_artifacts": [
                name
                for name, path in (("ood.json", ood_json), ("ood_stats.pt", ood_stats))
                if not path.exists()
            ],
        }
    data = json.loads(ood_json.read_text(encoding="utf-8"))
    validation_status = data.get("ood_negative_validation", {}).get("validation_status")
    return {
        "status": "available",
        "release_gate": "fail" if validation_status == "insufficient_ood_negatives" else "pass",
        "validation_status": validation_status,
        "method": data.get("method"),
        "mahalanobis": data.get("mahalanobis", {}),
        "max_softmax_baseline": data.get("max_softmax_baseline", {}),
        "ood_negative_validation": data.get("ood_negative_validation", {}),
    }


def has_release_gate_failure(report: dict[str, Any]) -> bool:
    for metrics in report.get("splits", {}).values():
        if metrics.get("danger", {}).get("release_gate") == "fail":
            return True
        if metrics.get("ood", {}).get("release_gate") == "fail":
            return True
    return False


@torch.no_grad()
def split_ood_report(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    run_dir: Path,
    artifact_status: dict[str, Any],
) -> dict[str, Any]:
    if artifact_status.get("status") != "available":
        return dict(artifact_status)
    if not hasattr(model, "features"):
        report = dict(artifact_status)
        report["status"] = "model_missing_features"
        report["release_gate"] = "fail"
        return report

    from training.ood import load_ood_stats, rejection_rate, score_mahalanobis, score_summary

    threshold = artifact_status.get("mahalanobis", {}).get("threshold")
    if threshold is None:
        report = dict(artifact_status)
        report["status"] = "missing_threshold"
        report["release_gate"] = "fail"
        return report

    stats = load_ood_stats(run_dir / "ood_stats.pt", map_location=device)
    model.eval()
    features: list[torch.Tensor] = []
    for images, _labels, _ids in loader:
        images = images.to(device, non_blocking=True)
        features.append(model.features(images).detach().cpu())  # type: ignore[attr-defined]
    if not features:
        report = dict(artifact_status)
        report["split_num_samples"] = 0
        report["split_id_reject_rate"] = 0.0
        return report
    scores = score_mahalanobis(torch.cat(features, dim=0), stats).cpu()
    report = dict(artifact_status)
    report["split_scores"] = score_summary(scores)
    report["split_id_reject_rate"] = rejection_rate(scores, float(threshold), higher_is_ood=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--edibility-csv", type=Path, default=DEFAULT_EDIBILITY_CSV)
    parser.add_argument("--danger-edges-csv", type=Path, default=DEFAULT_DANGER_EDGES_CSV)
    parser.add_argument("--danger-budget", type=int, default=0)
    parser.add_argument("--fail-on-release-gate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, config = load_model_from_run(args.run_dir, device)
    calibration = load_calibration(args.run_dir)
    temperature = calibration["temperature"]
    confidence_floor = calibration["confidence_floor"]
    name_to_index = scientific_name_to_index(label_map)
    num_classes = int(label_map["num_classes"])
    ood_status = ood_artifact_status(args.run_dir)

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "temperature": temperature,
        "confidence_floor": confidence_floor,
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
        danger = build_danger_report(
            scaled,
            labels,
            label_map,
            edibility_csv=args.edibility_csv,
            danger_edges_csv=args.danger_edges_csv,
            confidence_floor=confidence_floor,
            default_budget=args.danger_budget,
        )
        ood = split_ood_report(model, loader, device, args.run_dir, ood_status)
        report["splits"][split]["danger"] = danger
        report["splits"][split]["ood"] = ood

    out_json = args.run_dir / "eval_report.json"
    out_md = args.run_dir / "eval_report.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Run dir: `{args.run_dir}`",
        f"Temperature: {temperature}",
        f"Confidence floor: {confidence_floor}",
        "",
    ]
    for split, metrics in report["splits"].items():
        danger = metrics["danger"]
        ood = metrics["ood"]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Samples: {metrics['num_samples']}",
                f"- Top-1: {metrics['top1']:.4f}",
                f"- Top-5: {metrics['top5']:.4f}",
                f"- Macro recall: {metrics['macro_recall']:.4f}",
                f"- Dangerous-to-foraging mis-ID rate: {danger['dangerous_to_foraging_misID_rate']:.4f}",
                f"- Danger release gate: {danger['release_gate']}",
                f"- OOD status: {ood.get('status')}; release gate: {ood.get('release_gate')}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"eval_report": str(out_json), "eval_markdown": str(out_md)}, indent=2))
    if args.fail_on_release_gate and has_release_gate_failure(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
