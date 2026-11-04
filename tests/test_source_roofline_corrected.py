"""Check native-source FP16 byte correction against vendored LLMCompass geometry."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.source_roofline_corrected import from_native_grouped_row

SOURCE=Path(__file__).resolve().parents[1]/"results/integration/native_grouped_a100_fp16_prefix_roofline.json"

class ByteCorrectSourceTests(unittest.TestCase):
    def test_fp16_source_element_vs_byte_bug_reference(self):
        d=json.loads(SOURCE.read_text())
        self.assertEqual(d["schema"],"cim-shared-prefix-native-grouped-A100-FP16-roofline/v1")
        model=from_native_grouped_row(d["results"][0])
        self.assertEqual(model.logical_qk_io_elements,2166784)
        self.assertEqual(model.correct_fp16_qk_io_bytes,4333568)
        self.assertAlmostEqual(model.native_legacy_matmul_qk_s*1e6,
                               1.0626699362432563,places=10)
        self.assertAlmostEqual(model.corrected_hbm_qk_s*1e6,
                               2.1253398724865127,places=10)
        self.assertAlmostEqual(model.corrected_l2_qk_s*1e6,
                               .6002836879432625,places=10)
        self.assertAlmostEqual(model.hbm_unfused_full_prefix_s*1e6,
                               4.44352721922511,places=10)
        self.assertAlmostEqual(model.l2_unfused_full_prefix_s*1e6,
                               1.294709745206199,places=10)
        self.assertGreater(model.corrected_hbm_qk_s,model.native_legacy_matmul_qk_s)

    def test_all_six_source_batches_and_native_geometry(self):
        d=json.loads(SOURCE.read_text())
        x=[from_native_grouped_row(row) for row in d["results"]]
        self.assertEqual([row.batch for row in x],[1,2,4,8,16,32])
        for point in x:
            self.assertGreaterEqual(point.corrected_hbm_qk_s,point.corrected_l2_qk_s)
            self.assertGreater(point.hbm_unfused_full_prefix_s,
                               point.corrected_hbm_qk_s)
            self.assertGreater(point.l2_unfused_full_prefix_s,
                               point.corrected_l2_qk_s)
        bad=dict(d["results"][0],native_operand_and_score_bytes=123)
        with self.assertRaisesRegex(ValueError,"geometry"):
            from_native_grouped_row(bad)

if __name__=="__main__":
    unittest.main()
