"""Hydragen-style fused-prefix/suffix exactness, including native GQA and SWA."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from cim_shared_prefix.hydragen_reference import prefix_suffix_attention


class HydragenReferenceTests(unittest.TestCase):
    def test_full_attention_equivalence_over_mha_and_gqa(self):
        torch.manual_seed(418)
        for b in (1, 3):
            for hq, hk in ((4, 4), (8, 2)):
                for p, s in ((5, 7), (0, 7), (5, 0), (16, 1)):
                    d = 16
                    q = torch.randn(b, hq, d)
                    pk = torch.randn(hk, p, d)
                    pv = torch.randn_like(pk)
                    sk = torch.randn(b, hk, s, d)
                    sv = torch.randn_like(sk)
                    actual = prefix_suffix_attention(q, pk, pv, sk, sv)
                    full_k = torch.cat((pk.unsqueeze(0).expand(b, -1, -1, -1), sk), dim=2)
                    full_v = torch.cat((pv.unsqueeze(0).expand(b, -1, -1, -1), sv), dim=2)
                    full_k = full_k.repeat_interleave(hq // hk, dim=1)
                    full_v = full_v.repeat_interleave(hq // hk, dim=1)
                    baseline = torch.softmax(torch.einsum("bhd,bhtd->bht", q, full_k) /
                                             (d ** 0.5), dim=-1)
                    expected = torch.einsum("bht,bhtd->bhd", baseline, full_v)
                    self.assertTrue(torch.allclose(actual, expected, atol=3e-6, rtol=3e-6),
                                    (b, hq, hk, p, s))

    def test_sliding_window_only_receives_visible_prefix(self):
        torch.manual_seed(217)
        b, hq, hk, d, window = 2, 4, 2, 16, 8
        prefix = 15
        suffix = 3
        visible = window - 1 - suffix
        q = torch.randn(b, hq, d)
        pk = torch.randn(hk, prefix, d)
        pv = torch.randn(hk, prefix, d)
        sk = torch.randn(b, hk, suffix, d)
        sv = torch.randn_like(sk)
        x = prefix_suffix_attention(q, pk[:, -visible:], pv[:, -visible:], sk, sv)
        pk[:, :prefix-visible] = 10000  # Evicted tokens must be causally invisible.
        y = prefix_suffix_attention(q, pk[:, -visible:], pv[:, -visible:], sk, sv)
        self.assertTrue(torch.equal(x, y))

    def test_reject_incompatible_geometry(self):
        q = torch.randn(2, 4, 16)
        pk = torch.randn(3, 5, 16)
        pv = torch.randn_like(pk)
        sk = torch.randn(2, 3, 2, 16)
        sv = torch.randn_like(sk)
        with self.assertRaises(ValueError):
            prefix_suffix_attention(q, pk, pv, sk, sv)
        with self.assertRaises(ValueError):
            prefix_suffix_attention(q, torch.randn(2, 0, 16),
                                    torch.randn(2, 0, 16),
                                    torch.randn(2, 0, 16),
                                    torch.randn(2, 0, 16))


if __name__ == "__main__":
    unittest.main()
