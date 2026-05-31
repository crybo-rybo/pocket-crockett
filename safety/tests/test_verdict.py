"""Golden tests for safety verdict policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.edibility import EdibilityRecord, EdibilityTable
from safety.verdict import (
    FUNGUS_REFUSAL_TEXT,
    Candidate,
    Policy,
    VisionSignal,
    decide,
)


class VerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = EdibilityTable.load(ROOT / "vision" / "edibility" / "edibility_skeleton.csv")
        self.policy = Policy(confidence_floor=0.7, ood_threshold=10.0)

    def signal(
        self,
        name: str = "Daucus carota",
        confidence: float = 0.95,
        *,
        ood_score: float = 1.0,
        subject_status: str = "plant_candidate",
    ) -> VisionSignal:
        return VisionSignal(
            top_n=[
                Candidate(name, confidence),
                Candidate("Cicuta maculata", 0.03),
            ],
            ood_score=ood_score,
            subject_status=subject_status,
        )

    def test_fungus_refusal_uses_exact_text(self) -> None:
        verdict = decide(self.signal(subject_status="fungus_suspected"), self.table, self.policy)
        self.assertEqual(verdict.status, "refused")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")
        self.assertEqual(verdict.refusal_text, FUNGUS_REFUSAL_TEXT)

    def test_in_scope_unknown_preserves_refusal_text_and_safe_guidance(self) -> None:
        verdict = decide(self.signal(subject_status="in_scope_unknown"), self.table, self.policy)
        self.assertEqual(verdict.status, "unknown")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")
        self.assertEqual(verdict.refusal_text, FUNGUS_REFUSAL_TEXT)

    def test_below_confidence_floor_returns_unknown_with_candidates(self) -> None:
        signal = self.signal(confidence=0.2)
        verdict = decide(signal, self.table, self.policy)
        self.assertEqual(verdict.status, "unknown")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")
        self.assertEqual(verdict.candidates, signal.top_n)

    def test_ood_score_past_threshold_returns_unknown(self) -> None:
        verdict = decide(self.signal(ood_score=11.0), self.table, self.policy)
        self.assertEqual(verdict.status, "unknown")
        self.assertIn("ood_score_above_threshold", verdict.reasons)

    def test_high_confidence_cicuta_still_do_not_eat_and_surfaces_lookalike(self) -> None:
        verdict = decide(self.signal("Cicuta maculata"), self.table, self.policy)
        self.assertEqual(verdict.status, "identified")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")
        self.assertEqual(verdict.surfaced_lookalikes, ["Daucus carota"])

    def test_high_confidence_daucus_surfaces_deadly_lookalikes(self) -> None:
        verdict = decide(self.signal("Daucus carota"), self.table, self.policy)
        self.assertEqual(verdict.status, "identified")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")
        self.assertEqual(
            verdict.surfaced_lookalikes,
            ["Cicuta maculata", "Cicuta douglasii", "Conium maculatum"],
        )

    def test_missing_edibility_record_returns_unknown(self) -> None:
        verdict = decide(self.signal("Not A Species"), self.table, self.policy)
        self.assertEqual(verdict.status, "unknown")
        self.assertEqual(verdict.consumption_guidance, "do_not_eat")

    def test_reviewed_edible_record_can_promote_consumption_guidance(self) -> None:
        table = EdibilityTable(
            [
                EdibilityRecord(
                    taxon_id="test:EDIBLE",
                    scientific_name="Test edible",
                    common_names=[],
                    edibility="edible",
                    consumption_guidance="eat",
                    preparation_required="",
                    toxic_lookalikes=[],
                    hazard_notes="",
                    confidence_of_record="high",
                    sources=[],
                    needs_human_review=False,
                )
            ]
        )
        verdict = decide(self.signal("Test edible"), table, self.policy)
        self.assertEqual(verdict.status, "identified")
        self.assertEqual(verdict.consumption_guidance, "eat")
        self.assertEqual(verdict.candidates[0].scientific_name, "Test edible")


if __name__ == "__main__":
    unittest.main()
