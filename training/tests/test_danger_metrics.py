"""Unit tests for danger-aware vision metrics."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common.metrics import lookalike_confusion
from tools.vision_pipeline import LOOKALIKE_EDGE_SEEDS


def logits_from_probs(rows: list[list[float]]) -> torch.Tensor:
    return torch.log(torch.tensor(rows, dtype=torch.float32))


class DangerMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.label_map = {
            "num_classes": 3,
            "classes": [
                {"class_index": 0, "scientific_name": "Danger plant"},
                {"class_index": 1, "scientific_name": "Foraging twin"},
                {"class_index": 2, "scientific_name": "Other plant"},
            ],
        }
        self.edges = [
            {
                "source_species": "Danger plant",
                "target_species": "Foraging twin",
                "source_role": "dangerous",
                "target_role": "foraging_twin",
                "gate_budget": 0,
            }
        ]

    def test_dangerous_to_foraging_mistakes_count_against_gate(self) -> None:
        logits = logits_from_probs(
            [
                [0.05, 0.90, 0.05],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
            ]
        )
        labels = torch.tensor([0, 0, 1])

        result = lookalike_confusion(logits, labels, self.label_map, self.edges, floor=0.75)

        edge = result["edges"][0]
        self.assertEqual(edge["count"], 1)
        self.assertEqual(edge["confident_wrong"], 1)
        self.assertEqual(edge["true_total"], 2)
        self.assertAlmostEqual(edge["rate"], 0.5)
        self.assertEqual(edge["release_gate"], "fail")
        self.assertEqual(result["dangerous_to_foraging"]["release_gate"], "fail")

    def test_reverse_mistakes_are_reported_without_tripping_danger_gate(self) -> None:
        logits = logits_from_probs(
            [
                [0.90, 0.05, 0.05],
                [0.90, 0.05, 0.05],
            ]
        )
        labels = torch.tensor([1, 0])

        result = lookalike_confusion(logits, labels, self.label_map, self.edges, floor=0.75)

        edge = result["edges"][0]
        self.assertEqual(edge["count"], 0)
        self.assertEqual(edge["confident_wrong"], 0)
        self.assertEqual(edge["reverse_count"], 1)
        self.assertEqual(edge["reverse_confident_wrong"], 1)
        self.assertEqual(edge["release_gate"], "pass")
        self.assertEqual(result["dangerous_to_foraging"]["release_gate"], "pass")

    def test_confidence_floor_only_increments_confident_wrong_above_floor(self) -> None:
        logits = logits_from_probs(
            [
                [0.05, 0.90, 0.05],
                [0.25, 0.50, 0.25],
            ]
        )
        labels = torch.tensor([0, 0])

        result = lookalike_confusion(logits, labels, self.label_map, self.edges, floor=0.80)

        edge = result["edges"][0]
        self.assertEqual(edge["count"], 2)
        self.assertEqual(edge["confident_wrong"], 1)
        self.assertEqual(edge["true_total"], 2)
        self.assertAlmostEqual(edge["confident_wrong_rate"], 0.5)

    def test_danger_edges_csv_lists_expected_release_edges(self) -> None:
        path = ROOT / "vision" / "safety" / "danger_edges.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with (ROOT / "vision" / "splits" / "output_classes.csv").open(encoding="utf-8-sig", newline="") as f:
            output_classes = {row["scientific_name"] for row in csv.DictReader(f)}

        self.assertEqual(
            set(rows[0]),
            {"source_species", "target_species", "source_role", "target_role", "gate_budget"},
        )
        self.assertEqual({row["source_role"] for row in rows}, {"dangerous"})
        self.assertEqual({row["target_role"] for row in rows}, {"foraging_twin"})
        self.assertEqual({row["gate_budget"] for row in rows}, {"0"})
        expected_edges = {
            (
                edge["source_species"],
                edge["target_species"],
                edge["source_role"],
                edge["target_role"],
                "0",
            )
            for edge in LOOKALIKE_EDGE_SEEDS
            if edge["source_role"] == "dangerous"
            and edge["target_role"] == "foraging_twin"
            and edge["source_species"] in output_classes
            and edge["target_species"] in output_classes
        }
        actual_edges = {
            (
                row["source_species"],
                row["target_species"],
                row["source_role"],
                row["target_role"],
                row["gate_budget"],
            )
            for row in rows
        }
        self.assertEqual(actual_edges, expected_edges)


if __name__ == "__main__":
    unittest.main()
