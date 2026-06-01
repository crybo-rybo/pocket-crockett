"""Metric helpers shared by evaluate.py and calibrate.py."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

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


def _scientific_name_to_index(label_map: dict) -> dict[str, int]:
    mapping = label_map.get("scientific_name_to_index")
    if mapping:
        return {str(name): int(idx) for name, idx in mapping.items()}
    return {str(c["scientific_name"]): int(c["class_index"]) for c in label_map.get("classes", [])}


def _parse_gate_budget(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _edge_from_mapping(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_species": str(edge["source_species"]),
        "target_species": str(edge["target_species"]),
        "source_role": str(edge.get("source_role", "")),
        "target_role": str(edge.get("target_role", "")),
        "gate_budget": _parse_gate_budget(edge.get("gate_budget")),
    }


def _normalize_lookalike_edges(graph: Mapping[str, Any] | Iterable[Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if isinstance(graph, Mapping):
        if "source_species" in graph and "target_species" in graph:
            return [_edge_from_mapping(graph)]
        for source_species, targets in graph.items():
            target_values = [targets] if isinstance(targets, str) else targets
            for target_species in target_values:
                if isinstance(target_species, Mapping):
                    edge = dict(target_species)
                    edge.setdefault("source_species", source_species)
                    edges.append(_edge_from_mapping(edge))
                else:
                    edges.append(
                        {
                            "source_species": str(source_species),
                            "target_species": str(target_species),
                            "source_role": "",
                            "target_role": "",
                            "gate_budget": None,
                        }
                    )
        return edges

    for edge in graph:
        if isinstance(edge, Mapping):
            edges.append(_edge_from_mapping(edge))
            continue
        if isinstance(edge, Sequence) and not isinstance(edge, str) and len(edge) >= 2:
            edges.append(
                {
                    "source_species": str(edge[0]),
                    "target_species": str(edge[1]),
                    "source_role": str(edge[2]) if len(edge) >= 3 else "",
                    "target_role": str(edge[3]) if len(edge) >= 4 else "",
                    "gate_budget": _parse_gate_budget(edge[4]) if len(edge) >= 5 else None,
                }
            )
            continue
        raise TypeError(f"Unsupported lookalike edge: {edge!r}")
    return edges


def _rate(count: int, true_total: int) -> float:
    return float(count / true_total) if true_total else 0.0


@torch.no_grad()
def lookalike_confusion(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_map: dict,
    graph: Mapping[str, Any] | Iterable[Any],
    *,
    floor: float,
) -> dict:
    """Report calibrated top-1 confusion across explicit look-alike edges.

    ``logits`` must already have temperature scaling applied. For each configured
    directed edge, the release gate is evaluated only in the source -> target
    direction when source_role is ``dangerous`` and target_role is
    ``foraging_twin``. Reverse confusions are reported for review, but they do
    not affect the dangerous -> foraging release gate.
    """
    if logits.ndim != 2:
        raise ValueError("lookalike_confusion expects a 2D logits tensor")
    if labels.ndim != 1:
        raise ValueError("lookalike_confusion expects a 1D labels tensor")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must contain the same number of rows")

    labels_for_calc = labels.to(device=logits.device)
    edges = _normalize_lookalike_edges(graph)
    name_to_index = _scientific_name_to_index(label_map)
    missing_species = sorted(
        {
            species
            for edge in edges
            for species in (edge["source_species"], edge["target_species"])
            if species not in name_to_index
        }
    )
    if missing_species:
        raise ValueError(f"Lookalike species missing from label map: {missing_species}")

    if logits.shape[0] == 0:
        preds = torch.empty((0,), dtype=torch.long, device=logits.device)
        max_probs = torch.empty((0,), dtype=logits.dtype, device=logits.device)
    else:
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

    confidence_floor = float(floor)
    edge_results: list[dict[str, Any]] = []
    directed_count = 0
    directed_confident_wrong = 0
    directed_true_total = 0
    danger_count = 0
    danger_confident_wrong = 0
    danger_true_total = 0
    failed_edges: list[dict[str, Any]] = []

    for edge in edges:
        source_species = edge["source_species"]
        target_species = edge["target_species"]
        source_idx = name_to_index[source_species]
        target_idx = name_to_index[target_species]

        source_mask = labels_for_calc == source_idx
        target_mask = labels_for_calc == target_idx
        source_true_total = int(source_mask.sum().item())
        target_true_total = int(target_mask.sum().item())

        source_to_target_mask = source_mask & (preds == target_idx)
        target_to_source_mask = target_mask & (preds == source_idx)
        source_to_target_count = int(source_to_target_mask.sum().item())
        target_to_source_count = int(target_to_source_mask.sum().item())
        source_to_target_confident = int((source_to_target_mask & (max_probs >= confidence_floor)).sum().item())
        target_to_source_confident = int((target_to_source_mask & (max_probs >= confidence_floor)).sum().item())

        source_to_target = {
            "count": source_to_target_count,
            "confident_wrong": source_to_target_confident,
            "true_total": source_true_total,
            "rate": _rate(source_to_target_count, source_true_total),
            "confident_wrong_rate": _rate(source_to_target_confident, source_true_total),
        }
        target_to_source = {
            "count": target_to_source_count,
            "confident_wrong": target_to_source_confident,
            "true_total": target_true_total,
            "rate": _rate(target_to_source_count, target_true_total),
            "confident_wrong_rate": _rate(target_to_source_confident, target_true_total),
        }

        directed_count += source_to_target_count
        directed_confident_wrong += source_to_target_confident
        directed_true_total += source_true_total

        gate_budget = edge["gate_budget"]
        is_danger_gate = edge["source_role"] == "dangerous" and edge["target_role"] == "foraging_twin"
        gate_count = source_to_target_confident
        release_gate = "not_applicable"
        if is_danger_gate and gate_budget is not None:
            release_gate = "pass" if gate_count <= gate_budget else "fail"
            danger_count += source_to_target_count
            danger_confident_wrong += source_to_target_confident
            danger_true_total += source_true_total
            if release_gate == "fail":
                failed_edges.append(
                    {
                        "source_species": source_species,
                        "target_species": target_species,
                        "gate_budget": gate_budget,
                        "gate_count": gate_count,
                    }
                )

        edge_results.append(
            {
                "source_species": source_species,
                "target_species": target_species,
                "source_role": edge["source_role"],
                "target_role": edge["target_role"],
                "gate_budget": gate_budget,
                "count": source_to_target_count,
                "confident_wrong": source_to_target_confident,
                "true_total": source_true_total,
                "rate": source_to_target["rate"],
                "confident_wrong_rate": source_to_target["confident_wrong_rate"],
                "reverse_count": target_to_source_count,
                "reverse_confident_wrong": target_to_source_confident,
                "reverse_true_total": target_true_total,
                "reverse_rate": target_to_source["rate"],
                "reverse_confident_wrong_rate": target_to_source["confident_wrong_rate"],
                "source_to_target": source_to_target,
                "target_to_source": target_to_source,
                "release_gate": release_gate,
            }
        )

    return {
        "confidence_floor": confidence_floor,
        "edges": edge_results,
        "directed": {
            "count": directed_count,
            "confident_wrong": directed_confident_wrong,
            "true_total": directed_true_total,
            "rate": _rate(directed_count, directed_true_total),
            "confident_wrong_rate": _rate(directed_confident_wrong, directed_true_total),
        },
        "dangerous_to_foraging": {
            "count": danger_count,
            "confident_wrong": danger_confident_wrong,
            "true_total": danger_true_total,
            "rate": _rate(danger_count, danger_true_total),
            "confident_wrong_rate": _rate(danger_confident_wrong, danger_true_total),
            "release_gate": "fail" if failed_edges else "pass",
            "failed_edges": failed_edges,
        },
        "release_gate": "fail" if failed_edges else "pass",
    }


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
