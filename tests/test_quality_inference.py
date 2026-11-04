"""CPU-only numerical tests; never imply full-model OPT/Mistral qualification."""
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F
from cim_shared_prefix.quality_inference import (
    Intervention, active_prefix_bounds, _effective_mask, _pad_physical_head,
    _unrepeat_k, intervene_sdpa,
)


class FakeResult:
    def __init__(self, scores):
        self.raw_scores = scores
        self.diagnostics = {"source_domain_violation_count": 0}


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute_r1(self, state, **kw):
        q, k = state
        self.calls.append((q.shape[-2], k.shape[-2]))
        return FakeResult(torch.matmul(q.double(), k.repeat_interleave(q.shape[1] // k.shape[1], dim=1).double().transpose(-2, -1)))


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        self.fake_sram = types.ModuleType("sram2llm")
        self.fake_backend = types.ModuleType("sram2llm.sinangil_qk")
        self.fake_backend.make_quant_state = lambda q, k, query_bits: (q, k)
        self.modules = patch.dict(sys.modules, {"sram2llm": self.fake_sram,
                                                "sram2llm.sinangil_qk": self.fake_backend})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_opt_padding_zero_and_head_group(self):
        q, k = torch.randn(1, 32, 3, 80), torch.randn(1, 32, 100, 80)
        pq, pk = _pad_physical_head(q, k)
        self.assertEqual(pq.shape[-1], 128)
        self.assertTrue(torch.equal(pq[..., :80], q))
        self.assertTrue((pq[..., 80:] == 0).all())
        self.assertTrue((pk[..., 80:] == 0).all())
        self.assertTrue(torch.equal(_unrepeat_k(k, 1), k))
        with self.assertRaises(ValueError):
            _pad_physical_head(torch.ones(1, 1, 1, 96), torch.ones(1, 1, 2, 96))

    def test_mistral_window_exact_endpoints(self):
        self.assertEqual(active_prefix_bounds(16000, 0, 4096), (11905, 16000))
        self.assertEqual(active_prefix_bounds(16000, 127, 4096), (12032, 16000))
        self.assertEqual(active_prefix_bounds(400, 3, 4096), (0, 400))
        self.assertEqual(active_prefix_bounds(16000, 4096, 4096), (16000, 16000))
        with self.assertRaises(ValueError):
            active_prefix_bounds(5, 2, 0)

    def test_r0_delegates_kernel_exactly(self):
        q = torch.randn(1, 4, 3, 128)
        k = torch.randn(1, 4, 13, 128)
        v = torch.randn_like(k)
        mask = torch.zeros(1, 1, 3, 13)
        for i in range(3):
            mask[..., i, 11+i:] = -1e9
        c = Intervention(model="mistral", arm="R0", prefix=10, suffix=3)
        native = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        actual = intervene_sdpa(q, k, v, native_sdpa=F.scaled_dot_product_attention,
                                intervention=c, layer=0, head_ratio=2,
                                window=4096, attn_mask=mask)
        self.assertTrue(torch.equal(native, actual))
        self.assertEqual(c.source_calls[0], 1)

    def test_r0_eager_source_equivalence_and_opt_scaling(self):
        for kind, hd, heads, window, scale in [('mistral',128,4,4096,None),('opt',80,4,None,1.0)]:
            q = torch.randn(1,heads,3,hd)*.04
            k = torch.randn(1,heads,13,hd)*.05
            v = torch.randn_like(k)
            mask = torch.full((1,1,3,13),torch.finfo(q.dtype).min)
            for j in range(3): mask[...,j,:11+j] = 0
            c=Intervention(model=kind,arm='R0',reference_kernel='eager',prefix=10,suffix=3)
            y=intervene_sdpa(q,k,v,native_sdpa=F.scaled_dot_product_attention,
                             intervention=c,layer=0,head_ratio=1,window=window,
                             attn_mask=mask,scale=scale)
            src=torch.matmul(q,k.transpose(-2,-1)) * ((1/math.sqrt(hd)) if scale is None else scale)
            src=(src+mask).masked_fill(mask!=0,torch.finfo(q.dtype).min)
            expected=torch.matmul(F.softmax(src,dim=-1,dtype=torch.float32).to(q.dtype),v)
            self.assertTrue(torch.equal(y,expected),kind)

    def test_mistral_eviction_ignores_117_past_prefix_keys(self):
        prefix, suffix, window, nq, ratio = 4200, 3, 4096, 4, 2
        q = torch.randn(1, nq, suffix, 128) * 0.02
        unique = torch.randn(1, nq // ratio, prefix+suffix, 128) * 0.02
        k = unique.repeat_interleave(ratio, dim=1)
        v = torch.randn_like(k)
        mask = torch.full((1, 1, suffix, prefix+suffix), torch.finfo(q.dtype).min)
        for t in range(suffix):
            lo = prefix+t-window+1
            mask[..., t, lo:prefix+t+1] = 0
        native = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        executor = FakeExecutor()
        c = Intervention(model="mistral", arm="R1", prefix=prefix, suffix=suffix,
                         executor=executor, window_id="tiny")
        actual = intervene_sdpa(q, k, v, native_sdpa=F.scaled_dot_product_attention,
                                intervention=c, layer=0, head_ratio=ratio,
                                window=window, attn_mask=mask)
        self.assertTrue(torch.allclose(actual, native, atol=3e-6, rtol=3e-6))
        self.assertEqual(executor.calls, [(1, 4095), (1, 4094), (1, 4093)])
        self.assertEqual(c.active_scores[0], nq * (4095 + 4094 + 4093))
        self.assertFalse(bool(_effective_mask(mask, prefix=prefix, qlen=suffix,
                                              klen=prefix+suffix, window=window,
                                              device=q.device)[..., :100].any()))

    def test_opt_mha_unscaled_projection(self):
        prefix, suffix, heads = 12, 3, 4
        original_q = torch.randn(1, heads, suffix, 80) * 0.03
        scaled_q = original_q / math.sqrt(80)
        k = torch.randn(1, heads, prefix+suffix, 80) * 0.03
        v = torch.randn_like(k)
        mask = torch.full((1, 1, suffix, prefix+suffix), torch.finfo(k.dtype).min)
        for t in range(suffix):
            mask[..., t, :prefix+t+1] = 0
        c = Intervention(model="opt", arm="R1", prefix=prefix, suffix=suffix,
                         executor=FakeExecutor(), window_id="opt")
        native = F.scaled_dot_product_attention(scaled_q, k, v, attn_mask=mask, scale=1.0)
        actual = intervene_sdpa(scaled_q, k, v, native_sdpa=F.scaled_dot_product_attention,
                                intervention=c, layer=2, head_ratio=1,
                                window=None, unscaled_query=original_q,
                                attn_mask=mask, scale=1.0)
        self.assertTrue(torch.allclose(actual, native, atol=2e-6, rtol=2e-6))
        self.assertEqual(c.active_scores[2], prefix*suffix*heads)

    def test_invalid_calibration_and_masks_rejected(self):
        c = Intervention(model="opt", arm="R3", prefix=12, suffix=3)
        with self.assertRaises(ValueError): c.validate()
        with self.assertRaises(ValueError):
            _effective_mask(torch.zeros(1, 1, 2, 10), prefix=10, qlen=3,
                            klen=13, window=None, device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            _unrepeat_k(torch.randn(1, 5, 2, 80), 2)


if __name__ == "__main__": unittest.main()
