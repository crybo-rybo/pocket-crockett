"""Unit tests for OOD fitting and packaging helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.ood import fit_mahalanobis, load_ood_stats, save_ood_stats, score_mahalanobis


class OODTests(unittest.TestCase):
    def test_synthetic_gaussian_clusters_score_far_points_high(self) -> None:
        generator = torch.Generator().manual_seed(123)
        cluster_a = torch.randn(128, 3, generator=generator) * 0.12 + torch.tensor([0.0, 0.0, 0.0])
        cluster_b = torch.randn(128, 3, generator=generator) * 0.12 + torch.tensor([4.0, 4.0, 4.0])
        features = torch.cat([cluster_a, cluster_b], dim=0)
        labels = torch.cat(
            [
                torch.zeros(cluster_a.shape[0], dtype=torch.long),
                torch.ones(cluster_b.shape[0], dtype=torch.long),
            ],
            dim=0,
        )

        stats = fit_mahalanobis(features, labels, num_classes=2, covariance_regularization=1e-3)
        id_points = torch.tensor([[0.05, -0.02, 0.03], [4.08, 3.95, 4.02]])
        far_points = torch.tensor([[10.0, 10.0, 10.0], [-7.0, 2.0, 8.0]])

        id_scores = score_mahalanobis(id_points, stats)
        far_scores = score_mahalanobis(far_points, stats)

        self.assertLess(float(id_scores.max().item()), float(far_scores.min().item()))

    def test_saved_stats_reload_with_identical_scores(self) -> None:
        features = torch.tensor(
            [
                [0.0, 0.1],
                [0.2, -0.1],
                [3.8, 4.0],
                [4.2, 3.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        query = torch.tensor([[0.1, 0.0], [4.0, 4.1], [8.0, 8.0]])
        stats = fit_mahalanobis(features, labels, num_classes=2, covariance_regularization=1e-2)
        original_scores = score_mahalanobis(query, stats)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ood_stats.pt"
            save_ood_stats(path, stats)
            loaded = load_ood_stats(path)

        reloaded_scores = score_mahalanobis(query, loaded)
        torch.testing.assert_close(reloaded_scores, original_scores, rtol=0.0, atol=0.0)

    def test_runpod_train_script_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "runpod" / "train.sh")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runpod_eval_enforces_release_gate_by_default(self) -> None:
        script = (ROOT / "runpod" / "train.sh").read_text(encoding="utf-8")
        self.assertIn('FAIL_ON_RELEASE_GATE="${FAIL_ON_RELEASE_GATE:-1}"', script)
        self.assertIn("--fail-on-release-gate", script)

    def test_uploader_allowlist_includes_ood_artifacts(self) -> None:
        path = ROOT / "tools" / "upload_checkpoints_to_r2.py"
        spec = importlib.util.spec_from_file_location("upload_checkpoints_to_r2", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIn("ood.json", module.UPLOAD_NAMES)
        self.assertIn("ood_stats.pt", module.UPLOAD_NAMES)


if __name__ == "__main__":
    unittest.main()
