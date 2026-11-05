"""Publication-facing OJCAS claims and report-scope regression.

The source manuscript and reviewer quotes are archival evidence, while these
active reports contain the current paper extension. No scientific measurements
or simulated coordinates are changed by this presentation check.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from cim_shared_prefix.ojcas_manuscript_baseline import manuscript_grid, paper_partial_energy
from cim_shared_prefix.ojcas_published_placement_delta import execute_728


ROOT = Path(__file__).resolve().parents[1]
CURRENT_REPORTS = (
    "README.md",
    "docs/OJCAS_PUBLISHED_S2_PLACEMENT_STAGE_RESULTS_20261105.md",
    "docs/OJCAS_PLACEMENT_README_20261105.md",
    "docs/OJCAS_MANUSCRIPT_BASELINE_LOCK_20261105.md",
    "docs/OJCAS_2026_0120_REVIEWER_REBUILD_CONTRACT.md",
)
PROHIBITED_PRESENTATION = re.compile(
    r"\bcontroller\b|\bcontrol[\s-]+overhead\b|"
    r"\bdigital[\s-]*mac\b|\bsram[\s+/-]*digital[\s-]*mac\b|"
    r"\bdigital implementation\b",
    re.IGNORECASE,
)


class SubmittedPaperPresentationScopeTests(unittest.TestCase):
    def test_active_reviewer_docs_are_focused_on_original_macro_model(self):
        for path in CURRENT_REPORTS:
            with self.subTest(path=path):
                text = (ROOT/path).read_text()
                self.assertIsNone(PROHIBITED_PRESENTATION.search(text))
        report = (ROOT/CURRENT_REPORTS[1]).read_text()
        self.assertIn("128 macros/engine (0.4096 mm²)", report)
        self.assertIn("117/728", report)
        self.assertIn("macro-area", report)
        self.assertIn("not reported", report)

    def test_summary_and_regenerated_source_do_not_create_new_scope_claim(self):
        archived = json.loads((ROOT/"results/reviewer/ojcas_published_S2_placement_stage_delta_20261105.json").read_text())
        self.assertIsNone(PROHIBITED_PRESENTATION.search(json.dumps(archived)))
        rows, evidence = execute_728()
        self.assertEqual(len(rows), 728)
        self.assertNotIn("additional_sram_controller_area", evidence)
        self.assertIsNone(evidence["new_s2_hardware_speedup"])
        self.assertIsNone(evidence["new_full_decoder_TPOT"])
        self.assertEqual(evidence["original_max_modeled_s2_QK_stage_speedup"], 1.97)
        self.assertTrue(all(x["original_engine_macros"] == 128 for x in rows))

    def test_submitted_paper_values_remain_invariant(self):
        rows, original = manuscript_grid()
        self.assertEqual(len(rows), 728)
        self.assertEqual(original["paper_favorable_counts_by_area"]["8"], 60)
        self.assertEqual(original["paper_favorable_counts_by_area"]["12"], 60)
        self.assertEqual(original["paper_max_modeled_QK_stage_speedup"], 1.97)
        energy = paper_partial_energy()
        self.assertAlmostEqual(energy["crossing_B_s1_hd"], 40.85689802913453)
        self.assertAlmostEqual(energy["crossing_B_s1_dp"], 25.44644387317909)


if __name__ == "__main__":
    unittest.main()
