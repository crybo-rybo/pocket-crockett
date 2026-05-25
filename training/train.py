#!/usr/bin/env python3
"""Fine-tune the Pocket Crockett closed-set species classifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.config import load_config
from training.common.label_map import load_label_map, scientific_name_to_index
from training.common.metrics import collect_logits, macro_recall, topk_accuracy
from training.datasets.folder_dataset import (
    DEFAULT_SPLITS_DIR,
    build_dataloader,
    compute_class_weights,
    load_pretraining_records,
    load_split_records,
)
from training.models.bioclip_classifier import build_bioclip_classifier
from training.models.classifier import build_classifier


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    label_map: dict,
    best_metric: float,
    stage: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "label_map_source": label_map.get("source"),
        "num_classes": label_map.get("num_classes"),
        "best_metric": best_metric,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def evaluate_split(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    logits, labels = collect_logits(model, dataloader, device)
    return {
        "top1": topk_accuracy(logits, labels, k=1),
        "top5": topk_accuracy(logits, labels, k=min(5, num_classes)),
        "macro_recall": macro_recall(logits, labels, num_classes),
        "num_samples": float(len(labels)),
    }


def resolve_training_config(config: dict, stage: str) -> dict:
    training = dict(config.get("training", {}))
    nested = training.pop("pretrain", None) if stage == "pretrain" else training.pop("finetune", None)
    if isinstance(nested, dict):
        training.update(nested)
    return training


def train_one_stage(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict,
    label_map: dict,
    output_dir: Path,
    stage: str,
) -> Path:
    train_cfg = resolve_training_config(config, stage)
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 1))
    num_classes = int(label_map["num_classes"])

    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    model.to(device)

    class_weights = None
    if train_cfg.get("weighted_loss", True):
        records = train_loader.dataset.records  # type: ignore[attr-defined]
        class_weights = compute_class_weights(records, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def make_optimizer() -> torch.optim.Optimizer:
        return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    optimizer = make_optimizer()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    backbone_unfrozen = freeze_epochs <= 0 or not hasattr(model, "unfreeze_all")

    best_metric = -1.0
    best_path = output_dir / "checkpoint-best.pt"
    last_path = output_dir / "checkpoint-last.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        if not backbone_unfrozen and epoch > freeze_epochs:
            model.unfreeze_all()
            optimizer = make_optimizer()
            scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - epoch + 1, 1))
            backbone_unfrozen = True

        model.train()
        running_loss = 0.0
        seen = 0
        start = time.time()
        for images, labels, _ids in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
        scheduler.step()

        val_metrics = evaluate_split(model, val_loader, device, num_classes)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_macro_recall": val_metrics["macro_recall"],
            "seconds": time.time() - start,
        }
        history.append(epoch_record)
        print(json.dumps({"stage": stage, **epoch_record}, indent=2))

        metric = val_metrics["macro_recall"]
        if metric >= best_metric:
            best_metric = metric
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                label_map=label_map,
                best_metric=best_metric,
                stage=stage,
            )
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            label_map=label_map,
            best_metric=best_metric,
            stage=stage,
        )

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--label-map", type=Path, default=None)
    parser.add_argument("--stage", choices=("finetune", "pretrain"), default="finetune")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint for BioCLIP finetune stage")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.resolved.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))

    model_type = config.get("model", {}).get("type", "timm")

    if args.stage == "pretrain":
        records, species_map = load_pretraining_records(args.data_root)
        num_classes = len(species_map)
        val_records = records[: max(len(records) // 100, 1)]
        train_records = records[len(val_records) :]
        label_map = {
            "num_classes": num_classes,
            "source": "pretraining_only manifest",
            "classes": [
                {"class_index": idx, "scientific_name": name}
                for name, idx in sorted(species_map.items(), key=lambda item: item[1])
            ],
        }
        if model_type == "bioclip":
            model = build_bioclip_classifier(config, num_classes)
        else:
            model = build_classifier(config, num_classes)
        train_loader = build_dataloader(
            train_records,
            args.data_root,
            batch_size=batch_size,
            image_size=image_size,
            augment=True,
            num_workers=num_workers,
            shuffle=True,
            weighted_sampler=bool(data_cfg.get("weighted_sampler", True)),
        )
        val_loader = build_dataloader(
            val_records,
            args.data_root,
            batch_size=batch_size,
            image_size=image_size,
            augment=False,
            num_workers=num_workers,
            shuffle=False,
            weighted_sampler=False,
        )
        best = train_one_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            label_map=label_map,
            output_dir=args.output_dir,
            stage="pretrain",
        )
        (args.output_dir / "pretrain_species_map.json").write_text(
            json.dumps(species_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"best_checkpoint": str(best)}, indent=2))
        return

    label_map = load_label_map(args.label_map)
    name_to_index = scientific_name_to_index(label_map)
    train_records = load_split_records("train", args.data_root, args.splits_dir, name_to_index)
    val_records = load_split_records("val", args.data_root, args.splits_dir, name_to_index)
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    if max_train:
        train_records = train_records[: int(max_train)]
    if max_val:
        val_records = val_records[: int(max_val)]
    if not train_records:
        raise SystemExit(f"No train records found under {args.data_root}. Did bootstrap unpack shards?")
    if not val_records:
        raise SystemExit(f"No val records found under {args.data_root}.")

    if model_type == "bioclip":
        model = build_bioclip_classifier(config, label_map["num_classes"])
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu")
            current = model.state_dict()
            loaded = state["model_state_dict"]
            filtered = {k: v for k, v in loaded.items() if k in current and current[k].shape == v.shape}
            current.update(filtered)
            model.load_state_dict(current)
            model.replace_head(label_map["num_classes"])
    else:
        model = build_classifier(config, label_map["num_classes"])

    train_loader = build_dataloader(
        train_records,
        args.data_root,
        batch_size=batch_size,
        image_size=image_size,
        augment=True,
        num_workers=num_workers,
        shuffle=True,
        weighted_sampler=bool(data_cfg.get("weighted_sampler", True)),
    )
    val_loader = build_dataloader(
        val_records,
        args.data_root,
        batch_size=batch_size,
        image_size=image_size,
        augment=False,
        num_workers=num_workers,
        shuffle=False,
        weighted_sampler=False,
    )

    best = train_one_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        label_map=label_map,
        output_dir=args.output_dir,
        stage="finetune",
    )
    (args.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": str(best)}, indent=2))


if __name__ == "__main__":
    main()
