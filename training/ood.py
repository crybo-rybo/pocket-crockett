#!/usr/bin/env python3
"""Fit and score an embedding-distance OOD guard for vision classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ID_PERCENTILE = 95.0
DEFAULT_COVARIANCE_REGULARIZATION = 1e-4


def _as_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D feature tensor, got shape {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("cannot fit or score OOD stats with zero feature rows")
    return features.detach()


def fit_mahalanobis(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int | None = None,
    covariance_regularization: float = DEFAULT_COVARIANCE_REGULARIZATION,
) -> dict[str, Any]:
    """Fit class means and a regularized shared covariance matrix.

    The returned stats are deliberately plain tensors/dicts so they can be saved
    with torch.save and consumed by training/runtime code without another class.
    """
    feature_matrix = _as_feature_matrix(features).to(dtype=torch.float64, device="cpu")
    labels = labels.detach().to(dtype=torch.long, device="cpu").flatten()
    if feature_matrix.shape[0] != labels.numel():
        raise ValueError(
            f"feature row count {feature_matrix.shape[0]} does not match labels {labels.numel()}"
        )
    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")

    observed = torch.unique(labels[labels >= 0]).sort().values
    if num_classes is not None:
        observed = observed[observed < int(num_classes)]
    if observed.numel() == 0:
        raise ValueError("no non-negative class labels available for Mahalanobis fit")

    class_means: list[torch.Tensor] = []
    class_indices: list[int] = []
    class_counts: list[int] = []
    centered_chunks: list[torch.Tensor] = []
    for class_index in observed.tolist():
        mask = labels == class_index
        class_features = feature_matrix[mask]
        if class_features.numel() == 0:
            continue
        mean = class_features.mean(dim=0)
        class_means.append(mean)
        class_indices.append(int(class_index))
        class_counts.append(int(class_features.shape[0]))
        centered_chunks.append(class_features - mean)

    if not class_means:
        raise ValueError("no class means were fitted")

    centered = torch.cat(centered_chunks, dim=0)
    feature_dim = int(feature_matrix.shape[1])
    dof = max(int(centered.shape[0] - len(class_means)), 1)
    covariance = centered.T @ centered / dof
    scale = torch.trace(covariance) / max(feature_dim, 1)
    if not torch.isfinite(scale) or scale <= 0:
        scale = torch.tensor(1.0, dtype=covariance.dtype)
    eye = torch.eye(feature_dim, dtype=covariance.dtype)
    regularized_covariance = covariance + float(covariance_regularization) * scale * eye
    try:
        precision = torch.linalg.inv(regularized_covariance)
    except RuntimeError:
        precision = torch.linalg.pinv(regularized_covariance)

    return {
        "method": "mahalanobis_shared_covariance",
        "class_means": torch.stack(class_means, dim=0),
        "class_indices": torch.tensor(class_indices, dtype=torch.long),
        "class_counts": torch.tensor(class_counts, dtype=torch.long),
        "precision": precision,
        "feature_dim": feature_dim,
        "num_fit_samples": int(feature_matrix.shape[0]),
        "covariance_regularization": float(covariance_regularization),
    }


def score_mahalanobis(features: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    """Return the minimum class Mahalanobis distance for each feature row."""
    feature_matrix = _as_feature_matrix(features)
    dtype = stats["class_means"].dtype
    device = feature_matrix.device
    feature_matrix = feature_matrix.to(dtype=dtype)
    class_means = stats["class_means"].to(device=device, dtype=dtype)
    precision = stats["precision"].to(device=device, dtype=dtype)
    diffs = feature_matrix.unsqueeze(1) - class_means.unsqueeze(0)
    left = torch.matmul(diffs, precision)
    distances = (left * diffs).sum(dim=2)
    return distances.min(dim=1).values


def score(features: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    return score_mahalanobis(features, stats)


mahalanobis_scores = score_mahalanobis


def save_ood_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)


def load_ood_stats(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def max_softmax_confidence(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    scaled = logits.detach().to(dtype=torch.float32) / max(float(temperature), 1e-6)
    return F.softmax(scaled, dim=1).max(dim=1).values


def percentile(values: torch.Tensor, percentile_value: float) -> float:
    values = values.detach().to(dtype=torch.float32, device="cpu").flatten()
    if values.numel() == 0:
        raise ValueError("cannot compute percentile of an empty tensor")
    q = min(max(float(percentile_value), 0.0), 100.0) / 100.0
    return float(torch.quantile(values, q).item())


def score_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().to(dtype=torch.float32, device="cpu").flatten()
    if values.numel() == 0:
        return {"num_samples": 0}
    return {
        "num_samples": int(values.numel()),
        "min": float(values.min().item()),
        "p05": percentile(values, 5.0),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def rejection_rate(scores: torch.Tensor, threshold: float, *, higher_is_ood: bool) -> float:
    scores = scores.detach().to(dtype=torch.float32, device="cpu").flatten()
    if scores.numel() == 0:
        return 0.0
    if higher_is_ood:
        rejected = scores > float(threshold)
    else:
        rejected = scores < float(threshold)
    return float(rejected.to(dtype=torch.float32).mean().item())


@torch.no_grad()
def collect_features_logits_labels(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(model, "features"):
        raise AttributeError("model must expose features(x) for OOD fitting")
    model.eval()
    all_features: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels, _ids in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = model.features(images)  # type: ignore[attr-defined]
        logits = model(images)
        all_features.append(_as_feature_matrix(features).cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_features:
        raise ValueError("dataloader produced no batches")
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
    )


def load_ood_negative_records(
    *,
    data_root: Path,
    output_class_names: set[str],
    csv_path: Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    from training.datasets.folder_dataset import SampleRecord, load_pretraining_records

    records: list[Any] = []
    metadata: dict[str, Any] = {
        "sources": {},
        "notes": [],
    }

    try:
        pretraining_records, _species_map = load_pretraining_records(data_root)
    except FileNotFoundError as exc:
        pretraining_records = []
        metadata["notes"].append(f"pretraining_only unavailable: {exc}")
    except OSError as exc:
        pretraining_records = []
        metadata["notes"].append(f"pretraining_only unavailable: {exc}")

    pretraining_records = [
        record for record in pretraining_records if record.scientific_name not in output_class_names
    ]
    metadata["sources"]["pretraining_only"] = len(pretraining_records)
    records.extend(pretraining_records)

    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"OOD negatives CSV not found: {csv_path}")
        csv_records: list[Any] = []
        missing = 0
        skipped = 0
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader, start=2):
                rel_path = row.get("path") or row.get("image_path") or row.get("file")
                if not rel_path:
                    skipped += 1
                    metadata["notes"].append(f"{csv_path}:{row_number} skipped: missing path column")
                    continue
                image_path = Path(rel_path)
                abs_path = image_path if image_path.is_absolute() else data_root / rel_path
                if not abs_path.exists():
                    missing += 1
                    continue
                csv_records.append(
                    SampleRecord(
                        image_id=row.get("image_id") or image_path.stem,
                        path=str(image_path if image_path.is_absolute() else rel_path),
                        scientific_name=row.get("scientific_name") or "ood_negative",
                        taxon_id=row.get("taxon_id", ""),
                        label=-1,
                    )
                )
        metadata["sources"]["csv"] = {
            "path": str(csv_path),
            "records": len(csv_records),
            "missing_files": missing,
            "skipped_rows": skipped,
        }
        records.extend(csv_records)

    return records, metadata


def _build_loader(records: list[Any], data_root: Path, data_cfg: dict[str, Any]):
    from training.datasets.folder_dataset import build_dataloader

    return build_dataloader(
        records,
        data_root,
        batch_size=int(data_cfg.get("batch_size", 32)),
        image_size=int(data_cfg.get("image_size", 224)),
        augment=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        shuffle=False,
        weighted_sampler=False,
    )


def _load_calibration_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "calibration.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fit_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    from training.common.label_map import scientific_name_to_index
    from training.datasets.folder_dataset import load_split_records
    from training.evaluate import load_model_from_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, config = load_model_from_run(args.run_dir, device)
    name_to_index = scientific_name_to_index(label_map)
    data_cfg = config.get("data", {})

    fit_records = load_split_records(args.fit_split, args.data_root, args.splits_dir, name_to_index)
    if args.fit_split == "train" and data_cfg.get("max_train_samples"):
        fit_records = fit_records[: int(data_cfg["max_train_samples"])]
    if not fit_records:
        raise SystemExit(f"No {args.fit_split} records found under {args.data_root}")

    threshold_records = load_split_records(
        args.threshold_split,
        args.data_root,
        args.splits_dir,
        name_to_index,
    )
    if not threshold_records:
        raise SystemExit(f"No {args.threshold_split} records found under {args.data_root}")

    fit_loader = _build_loader(fit_records, args.data_root, data_cfg)
    threshold_loader = _build_loader(threshold_records, args.data_root, data_cfg)

    fit_features, _fit_logits, fit_labels = collect_features_logits_labels(model, fit_loader, device)
    stats = fit_mahalanobis(
        fit_features,
        fit_labels,
        num_classes=int(label_map["num_classes"]),
        covariance_regularization=args.covariance_regularization,
    )
    stats["fit_split"] = args.fit_split
    stats["label_map_num_classes"] = int(label_map["num_classes"])

    threshold_features, threshold_logits, _threshold_labels = collect_features_logits_labels(
        model,
        threshold_loader,
        device,
    )
    id_scores = score_mahalanobis(threshold_features, stats).cpu()
    mahalanobis_threshold = percentile(id_scores, args.id_percentile)

    calibration_metadata = _load_calibration_metadata(args.run_dir)
    temperature = float(calibration_metadata.get("temperature", 1.0))
    id_confidences = max_softmax_confidence(threshold_logits, temperature=temperature)
    softmax_threshold_percentile = max(0.0, 100.0 - float(args.id_percentile))
    max_softmax_threshold = percentile(id_confidences, softmax_threshold_percentile)

    negative_records, negative_metadata = load_ood_negative_records(
        data_root=args.data_root,
        output_class_names=set(name_to_index),
        csv_path=args.ood_negatives_csv,
    )

    validation_status = "insufficient_ood_negatives"
    negative_report: dict[str, Any] = {
        "num_samples": 0,
        "mahalanobis_reject_rate": None,
        "max_softmax_reject_rate": None,
    }
    if negative_records:
        negative_loader = _build_loader(negative_records, args.data_root, data_cfg)
        negative_features, negative_logits, _negative_labels = collect_features_logits_labels(
            model,
            negative_loader,
            device,
        )
        negative_scores = score_mahalanobis(negative_features, stats).cpu()
        negative_confidences = max_softmax_confidence(negative_logits, temperature=temperature)
        validation_status = "validated"
        negative_report = {
            "num_samples": int(negative_scores.numel()),
            "mahalanobis_scores": score_summary(negative_scores),
            "mahalanobis_reject_rate": rejection_rate(
                negative_scores,
                mahalanobis_threshold,
                higher_is_ood=True,
            ),
            "max_softmax_confidence": score_summary(negative_confidences),
            "max_softmax_reject_rate": rejection_rate(
                negative_confidences,
                max_softmax_threshold,
                higher_is_ood=False,
            ),
        }

    stats_path = args.run_dir / "ood_stats.pt"
    save_ood_stats(stats_path, stats)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "method": "mahalanobis_shared_covariance",
        "stats_artifact": "ood_stats.pt",
        "fit_split": args.fit_split,
        "threshold_split": args.threshold_split,
        "threshold": mahalanobis_threshold,
        "threshold_rule": "reject_as_unknown_when_mahalanobis_score_exceeds_threshold",
        "validation_status": validation_status,
        "num_fit_samples": int(fit_features.shape[0]),
        "feature_dim": int(fit_features.shape[1]),
        "num_classes_with_means": int(stats["class_means"].shape[0]),
        "covariance_regularization": float(args.covariance_regularization),
        "mahalanobis": {
            "higher_is_ood": True,
            "threshold": mahalanobis_threshold,
            "id_percentile": float(args.id_percentile),
            "id_scores": score_summary(id_scores),
            "id_reject_rate_at_threshold": rejection_rate(
                id_scores,
                mahalanobis_threshold,
                higher_is_ood=True,
            ),
        },
        "max_softmax_baseline": {
            "higher_is_ood": False,
            "temperature": temperature,
            "threshold": max_softmax_threshold,
            "id_percentile": softmax_threshold_percentile,
            "confidence_floor_recommendation": calibration_metadata.get(
                "confidence_floor_recommendation"
            ),
            "id_confidence": score_summary(id_confidences),
            "id_reject_rate_at_threshold": rejection_rate(
                id_confidences,
                max_softmax_threshold,
                higher_is_ood=False,
            ),
        },
        "ood_negative_validation": {
            "validation_status": validation_status,
            "sources": negative_metadata["sources"],
            "notes": negative_metadata["notes"],
            **negative_report,
        },
        "notes": (
            "Use Mahalanobis score > threshold as an OOD unknown gate. "
            "Max-softmax is recorded as a baseline only; runtime should still "
            "gate edibility separately and fail safe to unknown/do_not_eat."
        ),
    }
    out_path = args.run_dir / "ood.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ood": str(out_path), "ood_stats": str(stats_path), **report}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit OOD stats for a completed run")
    fit_parser.add_argument("--run-dir", type=Path, required=True)
    fit_parser.add_argument("--data-root", type=Path, required=True)
    fit_parser.add_argument(
        "--splits-dir",
        type=Path,
        default=ROOT / "vision" / "splits",
    )
    fit_parser.add_argument("--fit-split", default="train")
    fit_parser.add_argument("--threshold-split", default="calibration")
    fit_parser.add_argument("--id-percentile", type=float, default=DEFAULT_ID_PERCENTILE)
    fit_parser.add_argument(
        "--covariance-regularization",
        type=float,
        default=DEFAULT_COVARIANCE_REGULARIZATION,
    )
    fit_parser.add_argument(
        "--ood-negatives-csv",
        type=Path,
        default=None,
        help="Optional CSV with path/image_id/scientific_name/taxon_id for future fungi or non-plant negatives.",
    )
    fit_parser.set_defaults(func=fit_from_cli)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
