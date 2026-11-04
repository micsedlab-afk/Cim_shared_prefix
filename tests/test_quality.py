import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix.quality import (validate_summary, quality_rows, export_quality,
                                       validate_sample_csv)
from cim_shared_prefix.profiles import PROFILES, geometry_row, visible_prefix, sweep_rows, export_sweep


class QualityTests(unittest.TestCase):
    def _summary(self):
        arms = {}
        for arm, delta in zip(("R0", "R1", "R2", "R3"), (0.0, .01, .02, .03)):
            mu = 2.0 + delta
            arms[arm] = {"mean_nll": mu, "nll_sum": mu * 4, "token_count": 4,
                         "window_count": 2, "ppl": math.exp(mu),
                         "nll_ci": [mu-.1, mu+.1],
                         "ppl_ci": [math.exp(mu-.1), math.exp(mu+.1)]}
        effects = {}
        for key, (a, b) in {"quantization": ("R0", "R1"), "macro": ("R1", "R2"),
                            "circuit": ("R2", "R3"), "hardware_only": ("R1", "R3"),
                            "total": ("R0", "R3")}.items():
            r = arms[b]["ppl"] / arms[a]["ppl"] - 1
            effects[key] = {"comparison": b+"-"+a, "relative_ppl": r,
                            "relative_ppl_ci": [r-.05, r+.05],
                            "delta_nll": arms[b]["mean_nll"]-arms[a]["mean_nll"]}
        return {"schema": "final_quality_scalar_summary/v1", "protocol_sha256": "a"*64,
                "n_samples": 2, "sample_ids": ["a", "b"], "prefix_lengths": [512],
                "suffix_length": 128,
                "quality_by_prefix_length": {"512": {"arms": arms,
                    "window_ids": ["a_p512_u128", "b_p512_u128"], **effects}}}

    def test_summary_and_export(self):
        s = self._summary()
        validate_summary(s)
        pts, fx = quality_rows(s)
        self.assertEqual((len(pts), len(fx)), (4, 5))
        with tempfile.TemporaryDirectory() as d:
            export_quality(s, d)
            data = json.loads((Path(d)/"quality.json").read_text())
            self.assertEqual(data["quality"][0]["ppl"], math.exp(2))
            self.assertTrue((Path(d)/"fig_absolute_ppl.svg").read_text().startswith("<svg"))
            with (Path(d)/"absolute_quality.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_reject_incomplete_and_mismatched(self):
        s = self._summary()
        del s["quality_by_prefix_length"]["512"]["arms"]["R3"]
        with self.assertRaises(ValueError): validate_summary(s)
        s = self._summary()
        s["quality_by_prefix_length"]["512"]["arms"]["R0"]["ppl"] = 3
        with self.assertRaises(ValueError): validate_summary(s)

    def test_raw_window_reconciliation(self):
        s = self._summary()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/"sample_arm.csv"
            with f.open("w", newline="") as out:
                writer = csv.DictWriter(out, ["sample_id", "Lp", "arm", "token_count", "nll_sum", "mean_nll"])
                writer.writeheader()
                for sample in ("a", "b"):
                    for arm in ("R0", "R1", "R2", "R3"):
                        mu = s["quality_by_prefix_length"]["512"]["arms"][arm]["mean_nll"]
                        writer.writerow({"sample_id": sample, "Lp": 512, "arm": arm,
                                         "token_count": 2, "nll_sum": mu*2, "mean_nll": mu})
            validate_sample_csv(f, s)
            with f.open("a") as out: out.write("a,512,R0,2,4,2\n")
            with self.assertRaises(ValueError): validate_sample_csv(f, s)


class ProfileTests(unittest.TestCase):
    def test_grounded_native_profiles(self):
        llama, mistral, opt = PROFILES
        self.assertEqual((llama.gqa_ratio, mistral.gqa_ratio, opt.gqa_ratio), (4, 4, 1))
        self.assertEqual((llama.head_dim, mistral.head_dim, opt.head_dim), (128, 128, 80))
        self.assertEqual(visible_prefix(mistral, 16384, 128, 0), 3967)
        self.assertEqual(visible_prefix(mistral, 16384, 128, 4000), 0)
        self.assertEqual(visible_prefix(llama, 16384, 128, 4000), 16384)

    def test_gqa_changes_shared_storage_not_scores(self):
        llama = PROFILES[0]
        a = geometry_row(llama, prefix=512, suffix=128, batch=8, output=16, ratio=4)
        b = geometry_row(llama, prefix=512, suffix=128, batch=8, output=16, ratio=1)
        self.assertEqual(a["effective_prefix_key_scores_all_outputs"], b["effective_prefix_key_scores_all_outputs"])
        self.assertEqual(a["physical_signed_prefix_k_bytes"]*4, b["physical_signed_prefix_k_bytes"])
        self.assertEqual(a["logical_shared_prefix_k_bytes"], 512*8*128*4*32//8)
        self.assertEqual(a["completed_score_transfer_bytes_all_outputs"], 8*32*512*32*16*4)
        self.assertEqual(a["evidence_class"], "geometry_only_native")
        self.assertEqual(b["evidence_class"], "geometry_only_counterfactual_gqa")
        self.assertEqual(a["numerical_quality"], "not_evaluated")

    def test_opt_context_and_sweep(self):
        opt = PROFILES[2]
        with self.assertRaises(ValueError):
            geometry_row(opt, prefix=2048, suffix=128, batch=1, output=32, ratio=1)
        rows = sweep_rows()
        self.assertTrue(rows)
        self.assertEqual({r["model_id"] for r in rows}, {p.model_id for p in PROFILES})
        self.assertTrue(all(r["prefix_tokens"]+r["suffix_tokens"]+r["output_tokens"] <= r["context_limit"] for r in rows))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"sweep.csv"
            self.assertEqual(len(export_sweep(path)), len(rows))
            with path.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(rows))


if __name__ == "__main__": unittest.main()
