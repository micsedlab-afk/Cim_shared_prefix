"""Numerical equivalence, source geometry, and fail-closed GPU evidence gates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from cim_shared_prefix.gpu_attention import (
    check_inputs, full_kv, full_sdpa, grouped_prefix_scores,
    shared_prefix_gemm_attention,
)


class AttentionKernelsTests(unittest.TestCase):
    def test_native_attention_endpoint_mha_gqa_and_zero_segments(self):
        torch.manual_seed(137)
        for b, hq, hk, d, p, s in [
            (1, 4, 4, 16, 16, 8),
            (2, 8, 2, 16, 33, 12),
            (3, 8, 2, 80, 15, 1),
            (1, 8, 2, 128, 31, 0),
            (2, 4, 2, 16, 0, 23),
        ]:
            with self.subTest(shape=(b, hq, hk, d, p, s)):
                q = torch.randn(b, hq, 1, d) * .025
                pk = torch.randn(hk, p, d)
                pv = torch.randn(hk, p, d)
                sk = torch.randn(b, hk, s, d)
                sv = torch.randn(b, hk, s, d)
                k, v = full_kv(q, pk, pv, sk, sv)
                native = full_sdpa(q, k, v)
                candidate = shared_prefix_gemm_attention(q, pk, pv, sk, sv)
                self.assertEqual(candidate.shape, native.shape)
                self.assertTrue(torch.allclose(native, candidate, atol=1e-5, rtol=1e-5),
                                (native-candidate).abs().max().item())
                if p:
                    ss = grouped_prefix_scores(q, pk)
                    expected = torch.matmul(q, pk.repeat_interleave(hq // hk, dim=0)
                                             .unsqueeze(0).expand(b,-1,-1,-1)
                                             .transpose(-1,-2)).squeeze(-2) / (d**.5)
                    self.assertTrue(torch.allclose(ss, expected, atol=1e-5, rtol=1e-5))

    def test_attention_rejects_mismatched_head_and_precision(self):
        q = torch.randn(1, 6, 1, 16)
        pk = torch.randn(4, 8, 16)
        pv = torch.randn_like(pk)
        sk = torch.randn(1, 4, 2, 16)
        sv = torch.randn_like(sk)
        with self.assertRaises(ValueError):
            check_inputs(q, pk, pv, sk, sv)
        q = torch.randn(1, 8, 1, 16)
        with self.assertRaises(ValueError):
            check_inputs(q, pk.double(), pv, sk, sv)
        with self.assertRaises(ValueError):
            check_inputs(q, pk[:, :0], pv[:, :0], sk[:, :, :0], sv[:, :, :0])

    def test_cuda_benchmark_does_not_invent_measured_results(self):
        if torch.cuda.is_available():
            self.skipTest("CPU fail-closed test requires CPU-only CI runner")
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)/"unavailable.json"
            cli = [sys.executable, str(ROOT/"scripts"/"benchmark_gpu_attention.py"),
                   "--model", "llama", "--prefix", "2048", "--suffix", "128",
                   "--batch", "1", "--output-tokens", "128",
                   "--steps", "0", "--precision", "bf16",
                   "--out", str(dst)]
            proc = subprocess.run(cli, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, (proc.stdout, proc.stderr))
            data = json.loads(dst.read_text())
            self.assertEqual(data["status"], "UNAVAILABLE_NO_CUDA_DEVICE")
            self.assertFalse(data["measurements"])
            self.assertIsNone(data["full_decoder_tpot_s"])
            self.assertIsNone(data["CIM_vs_GPU_speedup"])


if __name__ == "__main__":
    unittest.main()
