"""Metric helpers shared by evaluate.py and calibrate.py."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in dataloader:
        images, labels, _ids = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-6)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    if logits.numel() == 0:
        return 0.0
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


def macro_recall(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    recalls: list[float] = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum().item() == 0:
            continue
        recalls.append((preds[mask] == cls).float().mean().item())
    return sum(recalls) / len(recalls) if recalls else 0.0


def per_class_recall(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[int, float]:
    preds = logits.argmax(dim=1)
    out: dict[int, float] = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum().item() == 0:
            out[cls] = float("nan")
        else:
            out[cls] = (preds[mask] == cls).float().mean().item()
    return out


def confusion_pairs(logits: torch.Tensor, labels: torch.Tensor) -> dict[tuple[int, int], int]:
    preds = logits.argmax(dim=1)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for pred, label in zip(preds.tolist(), labels.tolist(), strict=True):
        if pred != label:
            counts[(label, pred)] += 1
    return dict(counts)


def topn_candidates(
    logits: torch.Tensor,
    label_map: dict,
    *,
    n: int = 5,
    temperature: float = 1.0,
) -> list[list[dict]]:
    probs = F.softmax(apply_temperature(logits, temperature), dim=1)
    values, indices = probs.topk(min(n, probs.shape[1]), dim=1)
    index_to_name = {c["class_index"]: c["scientific_name"] for c in label_map["classes"]}
    batches: list[list[dict]] = []
    for row_vals, row_idx in zip(values.tolist(), indices.tolist(), strict=True):
        batches.append(
            [
                {
                    "class_index": idx,
                    "scientific_name": index_to_name.get(idx, str(idx)),
                    "confidence": val,
                }
                for idx, val in zip(row_idx, row_vals, strict=True)
            ]
        )
    return batches


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 50) -> float:
    """Single-parameter temperature scaling on held-out logits."""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp_min(1e-6), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-6).item())


def recommend_confidence_floor(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Conservative floor: 5th percentile confidence on correct predictions, capped at 0.9."""
    preds = probs.argmax(dim=1)
    correct_mask = preds == labels
    if correct_mask.sum().item() == 0:
        return 0.5
    correct_conf = probs.max(dim=1).values[correct_mask]
    floor = torch.quantile(correct_conf, 0.05).item()
    return float(min(max(floor, 0.1), 0.9))
