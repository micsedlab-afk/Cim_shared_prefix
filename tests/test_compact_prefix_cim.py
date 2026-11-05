"""Reproducible correctness and source-area tests for compressed shared prefixes."""
import itertools
import unittest
from math import ceil
import numpy as np

from cim_shared_prefix.compact_prefix_cim import (
    cluster_prefix,centroid_prefix_context,exact_prefix_context,
    output_error_metrics,macro_summary_area,select_layer_representatives,
    native_macro_tile_areas,
)


class CompactPrefixTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(22848)

    def test_identity_when_every_key_is_its_own_centroid(self):
        keys=self.rng.normal(size=(64,64)).astype(np.float32)
        values=self.rng.normal(size=(64,64)).astype(np.float32)
        q=self.rng.normal(size=(7,64)).astype(np.float32)
        s=cluster_prefix(keys,values,64)
        x,l=exact_prefix_context(q,keys,values)
        y,m=centroid_prefix_context(q,s)
        np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(l,m,atol=1e-6,rtol=1e-6)
        self.assertTrue(np.all(s.counts==1))
        self.assertTrue(np.all(s.radius==0))

    def test_duplicate_keys_are_losslessly_aggregated(self):
        base=self.rng.normal(size=(4,64)).astype(np.float32)
        k=np.repeat(base,16,axis=0)
        v=self.rng.normal(size=(64,64)).astype(np.float32)
        q=self.rng.normal(size=(12,64)).astype(np.float32)
        s=cluster_prefix(k,v,4)
        exact,el=exact_prefix_context(q,k,v)
        avg,al=centroid_prefix_context(q,s)
        np.testing.assert_allclose(exact,avg,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(el,al,atol=2e-6,rtol=2e-6)

    def test_count_log_correction_on_imbalanced_keys(self):
        k=np.zeros((16,64),dtype=np.float32)
        k[-1,:]=1
        v=np.zeros_like(k);v[-1,:]=2
        q=np.ones((3,64),dtype=np.float32)*0.1
        s=cluster_prefix(k,v,2)
        self.assertEqual(sorted(s.counts.tolist()),[1,15])
        x,l=exact_prefix_context(q,k,v)
        y,m=centroid_prefix_context(q,s)
        np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(l,m,atol=1e-6,rtol=1e-5)

    def test_signed_q4_is_separate_quantization_error(self):
        k=self.rng.normal(size=(128,128)).astype(np.float32)*.4
        v=self.rng.normal(size=(128,128)).astype(np.float32)
        q=self.rng.normal(size=(9,128)).astype(np.float32)*.4
        s=cluster_prefix(k,v,32)
        out_fp,l_fp=centroid_prefix_context(q,s,"fp32")
        out_k,l_k=centroid_prefix_context(q,s,"k4")
        out_qk,l_qk=centroid_prefix_context(q,s,"q4k4")
        for a in (out_fp,out_k,out_qk,l_fp,l_k,l_qk):
            self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(s.signed_k4<=7))
        self.assertTrue(np.all(s.signed_k4>=-7))
        d=output_error_metrics(q,k,v,s,"q4k4")
        self.assertGreaterEqual(d["relative_L2_mean"],0)
        with self.assertRaises(ValueError):
            centroid_prefix_context(q,s,"ADC4_fake")

    def test_native_llama_full_32layer_summary_fits_macro_budget(self):
        a=macro_summary_area(32,8,128,64)
        self.assertEqual(a["signed_K_CIM_macros"],4096)
        self.assertEqual(a["FP16_V_SRAM_macros"],128)
        self.assertEqual(a["metadata_SRAM_macros"],2)
        self.assertAlmostEqual(a["signed_K_V_metadata_macro_area_mm2"],
                               14.274279452054795,places=11)
        self.assertLess(a["signed_K_V_metadata_macro_area_mm2"],16)
        b=macro_summary_area(32,8,128,128)
        self.assertGreater(b["signed_K_V_metadata_macro_area_mm2"],16)
        self.assertIsNone(a["unmodeled_peripheral_area_mm2"])

    def test_integer_dp_vs_exhaustive_per_layer_search(self):
        e=np.array([[.5,.25,0.03],[.1,.05,.04],[.3,.2,.15],
                    [.7,.3,.15]],dtype=float)
        options=(16,32,48)
        tile_area,_,_=native_macro_tile_areas(8,128)
        metadata=ceil(4*8*max(options)*4/32768)*(262144/29.2e6)
        budget=8*tile_area+metadata+1e-9
        result=select_layer_representatives(e,options,8,128,budget)
        exhaustive=min((sum(e[i,options.index(m)] for i,m in enumerate(ms)),ms)
                       for ms in itertools.product(options,repeat=4)
                       if sum(m//16 for m in ms)<=8)
        self.assertAlmostEqual(result["error_proxy_sum"],exhaustive[0])
        self.assertEqual(tuple(result["per_layer_representatives"]),exhaustive[1])
        self.assertLessEqual(result["macro_area_including_max_metadata_mm2"],budget)
        with self.assertRaises(ValueError):
            select_layer_representatives(e,options,8,128,0.001)

    def test_sensitivity_not_claimed_as_hardware_or_model_PPL(self):
        e=np.full((32,4),0.1)
        r=select_layer_representatives(e,(16,32,64,128),8,128,16)
        self.assertEqual(len(r["per_layer_representatives"]),32)
        self.assertIn("not PPL",r["objective"])
        self.assertLessEqual(r["macro_area_including_max_metadata_mm2"],16)
        self.assertTrue(r["metadata_reserved_worst_case"])

if __name__=="__main__":
    unittest.main()
