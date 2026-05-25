"""Unit tests for training helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.label_map import load_label_map, scientific_name_to_index
from training.scripts.generate_label_map import generate_label_map


class LabelMapTests(unittest.TestCase):
    def test_generate_label_map_has_47_classes(self) -> None:
        payload = generate_label_map(ROOT / "vision" / "splits" / "output_classes.csv")
        self.assertEqual(payload["num_classes"], 47)
        self.assertEqual(len(payload["classes"]), 47)
        indices = [c["class_index"] for c in payload["classes"]]
        self.assertEqual(indices, list(range(47)))

    def test_committed_label_map_loads(self) -> None:
        label_map = load_label_map()
        self.assertEqual(label_map["num_classes"], 47)
        mapping = scientific_name_to_index(label_map)
        self.assertEqual(mapping["Acer rubrum"], 0)
        self.assertIn("Atropa belladonna", mapping)

    def test_label_map_json_matches_generator(self) -> None:
        generated = generate_label_map(ROOT / "vision" / "splits" / "output_classes.csv")
        committed = json.loads((ROOT / "training" / "artifacts" / "label_map.json").read_text(encoding="utf-8"))
        self.assertEqual(
            generated["scientific_name_to_index"],
            committed["scientific_name_to_index"],
        )


if __name__ == "__main__":
    unittest.main()
