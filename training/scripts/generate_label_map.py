#!/usr/bin/env python3
"""Generate training/artifacts/label_map.json from vision/splits/output_classes.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CLASSES = ROOT / "vision" / "splits" / "output_classes.csv"
DEFAULT_OUT = ROOT / "training" / "artifacts" / "label_map.json"


def generate_label_map(output_classes_path: Path) -> dict:
    with output_classes_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # Stable ordering: alphabetical by scientific_name (matches output_classes.csv sort).
    rows.sort(key=lambda r: r["scientific_name"])
    classes = []
    for index, row in enumerate(rows):
        classes.append(
            {
                "class_index": index,
                "scientific_name": row["scientific_name"],
                "requested_scientific_name": row.get("requested_scientific_name", ""),
                "taxon_id": row.get("taxon_id", ""),
                "priority_group": row.get("priority_group", ""),
                "matched_status": row.get("matched_status", ""),
                "role": row.get("role", ""),
            }
        )
    return {
        "version": 1,
        "num_classes": len(classes),
        "source": str(output_classes_path.relative_to(ROOT)),
        "classes": classes,
        "scientific_name_to_index": {c["scientific_name"]: c["class_index"] for c in classes},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_CLASSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = generate_label_map(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_classes": payload["num_classes"]}, indent=2))


if __name__ == "__main__":
    main()
