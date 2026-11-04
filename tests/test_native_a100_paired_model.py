"""Source-native A100 FP16 grouped-QK model paired with 7nm CIM causal execution."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeA100PairedStudyTests(unittest.TestCase):
    def run_source_comparison(self, gpu_source=None):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"comparison.json"
            cmd=[
                sys.executable,str(ROOT/"scripts"/"compare_native_a100_model_7nm.py"),
                "--out",str(out),
            ]
            if gpu_source is not None:
                cmd+=["--gpu-source",str(gpu_source)]
            p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=45)
            return p,json.loads(out.read_text()) if out.exists() else None

    def test_all_128_matched_source_model_steps_and_pairwise_difference(self):
        p, result=self.run_source_comparison()
        self.assertEqual(p.returncode,0,(p.stdout,p.stderr))
        self.assertEqual(result["status"],"EXECUTED_SOURCE_ROOFLINE_VS_7NM_MACRO_SERIAL_MODEL")
        self.assertEqual(len(result["token_points"]),128)
        self.assertEqual([x["step"] for x in result["token_points"]],list(range(128)))
        self.assertTrue(math.isclose(result["source_roofline_saving_us_per_layer"],
                                     1.0606611083864643,rel_tol=1e-10))
        self.assertTrue(math.isclose(result["source_threshold_us_per_layer"],
                                     6.058113,rel_tol=1e-10))
        self.assertTrue(math.isclose(
            result["S2_minus_S0_attention_model_difference_s_per_generated_token"]*1e6,
            159.91846053163314,rel_tol=1e-9))
        self.assertTrue(math.isclose(result["floor_minus_roofline_saving_s_per_token"]*1e6,
                                     30.70746453163314,rel_tol=1e-9))
        self.assertIsNone(result["qualified_end_to_end_TPOT_s"])
        self.assertIsNone(result["global_GPU_vs_CIM_speedup"])
        self.assertFalse(result["GPU_source"]["measured_hardware"])
        self.assertIn("roofline",result["scope_limit"])

    def test_reject_missing_source_operator_point(self):
        source=json.loads(
            (ROOT/"results/integration/source_native_a100_fp16_gqa_qk_128steps.json").read_text()
        )
        source["steps"]=source["steps"][:-1]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"broken.json"
            path.write_text(json.dumps(source))
            p,_=self.run_source_comparison(path)
            self.assertNotEqual(p.returncode,0)
            self.assertIn("incomplete",p.stderr)


if __name__=="__main__":
    unittest.main()
