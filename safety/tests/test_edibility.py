"""Unit tests for edibility table loading."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.edibility import EdibilityTable


class EdibilityTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = EdibilityTable.load(ROOT / "vision" / "edibility" / "edibility_skeleton.csv")

    def test_loads_current_skeleton(self) -> None:
        self.assertEqual(len(self.table.records), 1068)
        daucus = self.table.record_for("Daucus carota")
        self.assertIsNotNone(daucus)
        self.assertEqual(daucus.edibility, "unknown")
        self.assertEqual(daucus.consumption_guidance, "do_not_eat")
        self.assertTrue(daucus.needs_human_review)

    def test_lookalike_graph_contains_carrot_and_allium_edges(self) -> None:
        graph = self.table.lookalike_graph(
            {
                "Daucus carota",
                "Cicuta maculata",
                "Cicuta douglasii",
                "Conium maculatum",
                "Allium canadense",
                "Allium tricoccum",
                "Zigadenus venenosus",
                "Zigadenus paniculatus",
                "Zigadenus glaberrimus",
            }
        )
        self.assertEqual(
            graph["Daucus carota"],
            ["Cicuta maculata", "Cicuta douglasii", "Conium maculatum"],
        )
        self.assertEqual(graph["Cicuta maculata"], ["Daucus carota"])
        self.assertEqual(
            graph["Allium canadense"],
            ["Zigadenus venenosus", "Zigadenus paniculatus", "Zigadenus glaberrimus"],
        )
        self.assertEqual(
            graph["Zigadenus venenosus"],
            ["Allium canadense", "Allium tricoccum"],
        )


if __name__ == "__main__":
    unittest.main()
