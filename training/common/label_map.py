"""Label map helpers for the 47-class closed-set classifier."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MAP = ROOT / "training" / "artifacts" / "label_map.json"


def load_label_map(path: Path | None = None) -> dict:
    label_map_path = path or DEFAULT_LABEL_MAP
    with label_map_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("num_classes") != len(data.get("classes", [])):
        raise ValueError(f"num_classes mismatch in {label_map_path}")
    return data


def scientific_name_to_index(label_map: dict) -> dict[str, int]:
    mapping = label_map.get("scientific_name_to_index")
    if mapping:
        return {str(k): int(v) for k, v in mapping.items()}
    return {c["scientific_name"]: int(c["class_index"]) for c in label_map["classes"]}
