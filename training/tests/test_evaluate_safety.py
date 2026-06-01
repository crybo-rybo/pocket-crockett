"""Unit tests for safety reporting helpers in evaluate.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.evaluate import (
    DEFAULT_DANGER_EDGES_CSV,
    DEFAULT_EDIBILITY_CSV,
    build_danger_report,
    has_release_gate_failure,
    ood_artifact_status,
)
from training.common.label_map import load_label_map


class EvaluateSafetyTests(unittest.TestCase):
    def test_danger_report_fails_release_gate_for_dangerous_to_foraging_confusion(self) -> None:
        label_map = load_label_map()
        name_to_index = label_map["scientific_name_to_index"]
        logits = torch.full((1, 47), -10.0)
        logits[0, name_to_index["Daucus carota"]] = 10.0
        labels = torch.tensor([name_to_index["Cicuta maculata"]])

        report = build_danger_report(
            logits,
            labels,
            label_map,
            edibility_csv=DEFAULT_EDIBILITY_CSV,
            danger_edges_csv=DEFAULT_DANGER_EDGES_CSV,
            confidence_floor=0.7,
            default_budget=0,
        )

        self.assertEqual(report["release_gate"], "fail")
        self.assertEqual(report["dangerous_to_foraging"]["confident_wrong"], 1)
        self.assertEqual(report["dangerous_to_foraging_misID_rate"], 1.0)

    def test_missing_ood_artifacts_are_reported_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = ood_artifact_status(Path(tmp))

        self.assertEqual(report["status"], "missing")
        self.assertEqual(report["release_gate"], "not_applicable")
        self.assertEqual(set(report["missing_artifacts"]), {"ood.json", "ood_stats.pt"})

    def test_release_gate_failure_helper_only_fails_on_explicit_fail(self) -> None:
        passing = {
            "splits": {
                "val": {
                    "danger": {"release_gate": "pass"},
                    "ood": {"release_gate": "not_applicable"},
                }
            }
        }
        failing = {
            "splits": {
                "val": {
                    "danger": {"release_gate": "pass"},
                    "ood": {"release_gate": "fail"},
                }
            }
        }

        self.assertFalse(has_release_gate_failure(passing))
        self.assertTrue(has_release_gate_failure(failing))


if __name__ == "__main__":
    unittest.main()
