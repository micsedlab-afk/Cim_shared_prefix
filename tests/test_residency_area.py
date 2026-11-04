"""Physical allocation and temporal execution independent invariant tests."""
import math
import unittest
from itertools import product

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.area_optimizer import CIM_7NM, SRAM_7NM
from cim_shared_prefix.residency_area import (
    ExternalArea, RateSet, layer_layout, residency_plan, enumerate_plans,
    accounting, cim_optimistic_compute_floor_s, cim_serial_stages,
    digital_serial_stages, boundary, best_cim, best_digital_requirement,
    min_rate_for_deadline,
)


def zero():return ExternalArea(0,0,0,0,0,0)

def r():return RateSet(1e12,1e12,1e12,1e12,1e12)


def large(out=128):return Workload(1,2048,0,32,8,128,32,out,key_bits=4)


class ResidencyTests(unittest.TestCase):
    def test_full_32layer_macro_area(self):
        l=layer_layout(large(),kind='cim')
        self.assertEqual(l.full_layer_macros,4096)
        self.assertEqual(l.all_layer_macros,131072)
        self.assertAlmostEqual(l.full_layer_macro_mm2,13.1072)
        self.assertAlmostEqual(l.all_layer_macro_mm2,419.4304)
        self.assertEqual(l.macros_per_tile_unit,2)

    def test_512_prefix_and_8192_prefix(self):
        for prefix, full in ((512,3.2768),(2048,13.1072),(8192,52.4288)):
            w=Workload(1,prefix,0,32,8,128,32,128)
            self.assertAlmostEqual(layer_layout(w,kind='cim').full_layer_macro_mm2,full)

    def test_total_16_support_threshold(self):
        a=ExternalArea(2.8928,0,0,0,0,0)
        p=residency_plan(large(),kind='cim',budget_mm2=16,support=a)
        self.assertIsNotNone(p)
        self.assertTrue(p.full_active_layer_in_scratch)
        self.assertAlmostEqual(p.complete_mm2,16)
        a=ExternalArea(2.9,0,0,0,0,0)
        p=residency_plan(large(),kind='cim',budget_mm2=16,support=a)
        self.assertFalse(p.full_active_layer_in_scratch)

    def test_one_mm2_partial_scratch(self):
        p=residency_plan(large(),kind='cim',budget_mm2=1,support=zero())
        self.assertIsNotNone(p)
        self.assertLess(p.instantiated_macros,4096)
        self.assertEqual(p.stream_waves_per_unpinned_layer,
                         math.ceil(2048/math.floor(1/(2*0.0032))))
        self.assertLessEqual(p.physical_macro_mm2,1)

    def test_sram_published_density_not_cell(self):
        l=layer_layout(large(),kind='sram')
        self.assertEqual(l.full_layer_macros,32)
        self.assertAlmostEqual(l.all_layer_macro_mm2,9.19299506849315)
        self.assertAlmostEqual(SRAM_7NM.area_mm2,262144/(29.2e6))

    def test_pinned_layer_requires_dedicated_scratch(self):
        w=Workload(1,512,0,32,8,128,32,128)
        m=layer_layout(w,kind='cim').full_layer_macro_mm2
        # Exactly one full layer: pinning it makes the remaining layers impossible.
        self.assertIsNone(residency_plan(w,kind='cim',budget_mm2=m,
                                         support=zero(),pinned_layers=1))
        self.assertIsNotNone(residency_plan(w,kind='cim',budget_mm2=m,
                                            support=zero(),pinned_layers=0))
        # Two full layers permit one pinned and one streaming layer.
        p=residency_plan(w,kind='cim',budget_mm2=2*m,support=zero(),pinned_layers=1)
        self.assertTrue(p.full_active_layer_in_scratch)
        self.assertEqual(p.instantiated_macros,2*layer_layout(w,kind='cim').full_layer_macros)

    def test_pinning_one_layer_and_partial_scratch(self):
        w=large()
        one=layer_layout(w,kind='cim').full_layer_macro_mm2
        p=residency_plan(w,kind='cim',budget_mm2=one+0.01,
                         support=zero(),pinned_layers=1)
        self.assertIsNotNone(p)
        self.assertGreater(p.stream_waves_per_unpinned_layer,1)
        self.assertEqual(p.pinned_layers,1)

    def test_all_layers_pinned_one_initial_program(self):
        w=large()
        a=layer_layout(w,kind='cim').all_layer_macro_mm2
        p=residency_plan(w,kind='cim',budget_mm2=a,support=zero(),pinned_layers=32)
        self.assertIsNotNone(p)
        self.assertEqual(p.stream_waves_per_unpinned_layer,0)
        self.assertEqual(p.programming_visits_total,32)
        self.assertEqual(p.programming_bytes_per_generated_token,
                         p.valid_key_layer_bytes*32/128)
        self.assertEqual(p.physical_macro_mm2,a)

    def test_cold_warm_and_output_length(self):
        w=large(out=16)
        p=residency_plan(w,kind='sram',budget_mm2=16,support=zero(),pinned_layers=16)
        self.assertEqual(p.programming_visits_total,32+15*16)
        q=residency_plan(w,kind='sram',budget_mm2=16,support=zero(),pinned_layers=16,
                         cold_start=False)
        self.assertEqual(q.programming_visits_total,16*16)
        self.assertLess(q.programming_bytes_total,p.programming_bytes_total)
        wlong=large(out=128)
        z=residency_plan(wlong,kind='sram',budget_mm2=16,support=zero(),pinned_layers=16)
        self.assertLess(z.programming_bytes_per_generated_token,p.programming_bytes_per_generated_token)

    def test_first_token_and_steady_programming_totals(self):
        w=large(out=16)
        p=residency_plan(w,kind='sram',budget_mm2=16,support=zero(),pinned_layers=16)
        self.assertEqual(p.first_token_programming_bytes,32*p.valid_key_layer_bytes)
        self.assertEqual(p.steady_token_programming_bytes,16*p.valid_key_layer_bytes)
        self.assertEqual(p.programming_bytes_total,
                         p.first_token_programming_bytes+15*p.steady_token_programming_bytes)
        z=residency_plan(w,kind='sram',budget_mm2=16,support=zero(),pinned_layers=32,
                         cold_start=False)
        self.assertEqual(z.first_token_programming_bytes,0)
        self.assertEqual(z.steady_token_programming_bytes,0)

    def test_no_pins_all_layers_reprogram_each_token(self):
        w=large()
        p=residency_plan(w,kind='cim',budget_mm2=16,support=zero(),pinned_layers=0)
        self.assertEqual(p.programming_visits_total,w.layers*w.output_tokens)
        self.assertEqual(p.programming_bytes_per_generated_token,64*1024**2)

    def test_no_double_count_macro_per_wave(self):
        p=residency_plan(large(),kind='cim',budget_mm2=1,support=zero())
        self.assertEqual(p.physical_macro_mm2,p.instantiated_macros*0.0032)
        self.assertLess(p.physical_macro_mm2,1.0)
        self.assertEqual(p.pinned_layers,0)
        self.assertGreater(p.stream_waves_per_unpinned_layer,1)

    def test_occupancy_and_native_unsigned_account(self):
        w=Workload(1,513,0,32,8,96,32,128)
        x=accounting(w)
        self.assertAlmostEqual(x.mapping_occupancy,(513*96)/(528*128))
        self.assertEqual(x.native_unsigned_mac_per_token,4*x.logical_signed_mac_per_token)
        self.assertGreater(x.native_issued_padded_mac_per_token,x.native_unsigned_mac_per_token)
        self.assertEqual(x.signed_partial_values_per_score,8)
        self.assertEqual(x.reduction_additions_per_token,7*w.layers*w.prefix_scores_per_layer)

    def test_complete_score_transfer_same_path(self):
        w=large()
        x=accounting(w)
        self.assertEqual(x.complete_score_bytes_per_token,8*1024**2)
        p=residency_plan(w,kind='cim',budget_mm2=16,support=zero())
        q=residency_plan(w,kind='sram',budget_mm2=16,support=zero())
        b1=boundary(w,p,r(),target_s=0.002)
        b2=boundary(w,q,r(),target_s=0.002,reuse=4)
        self.assertEqual(b1.complete_score_bytes_per_token,b2.complete_score_bytes_per_token)

    def test_sign_product_parallelism_two_vs_four(self):
        w=large()
        p=residency_plan(w,kind='cim',budget_mm2=16,support=zero())
        t1=cim_optimistic_compute_floor_s(w,p,sign_plane_parallelism=1)
        t2=cim_optimistic_compute_floor_s(w,p,sign_plane_parallelism=2)
        self.assertAlmostEqual(t1,2*t2)
        self.assertEqual(t2,32*4*2*5.5e-9)

    def test_physical_query_copies_reduce_rounds_only_if_allocated(self):
        w=Workload(8,512,0,32,8,128,32,64)
        p1=residency_plan(w,kind='cim',budget_mm2=16,support=zero(),copies=1)
        p2=residency_plan(w,kind='cim',budget_mm2=16,support=zero(),copies=2)
        self.assertGreater(p2.instantiated_macros,p1.instantiated_macros)
        self.assertLess(cim_optimistic_compute_floor_s(w,p2),
                        cim_optimistic_compute_floor_s(w,p1))
        with self.assertRaises(ValueError):
            cim_optimistic_compute_floor_s(w,p2,copies=1)

    def test_min_rate_domain(self):
        self.assertEqual(min_rate_for_deadline(0,1,0),0)
        self.assertEqual(min_rate_for_deadline(100,1,0.5),200)
        self.assertTrue(math.isinf(min_rate_for_deadline(100,1,1)))
        with self.assertRaises(ValueError):min_rate_for_deadline(10,0,0)

    def test_digital_area_not_qualified(self):
        w=large()
        p=residency_plan(w,kind='sram',budget_mm2=16,support=zero())
        b=boundary(w,p,r(),target_s=0.002,reuse=4)
        self.assertFalse(b.qualified_digital_isoarea)
        self.assertGreater(b.mac_core_area_mm2,0)
        self.assertGreater(b.required_mac_per_s_per_mm2,0)

    def test_required_density_monotonic_in_area_for_fixed_memory(self):
        w=large()
        a=zero();rates=r()
        p1=residency_plan(w,kind='sram',budget_mm2=4,support=a,pinned_layers=0)
        p2=residency_plan(w,kind='sram',budget_mm2=8,support=a,pinned_layers=0)
        b1=boundary(w,p1,rates,target_s=.002)
        b2=boundary(w,p2,rates,target_s=.002)
        self.assertGreater(b1.required_mac_per_s_per_mm2,b2.required_mac_per_s_per_mm2)

    def test_sram_rate_affects_digital_threshold(self):
        w=large()
        p=residency_plan(w,kind='sram',budget_mm2=16,support=zero())
        slow=boundary(w,p,RateSet(1e12,0.1e12,1e12,1e12,1e12),target_s=.002)
        fast=boundary(w,p,RateSet(1e12,10e12,1e12,1e12,1e12),target_s=.002)
        self.assertGreater(slow.required_mac_per_s_per_mm2,fast.required_mac_per_s_per_mm2)

    def test_programming_padding_full_macro_policy(self):
        w=Workload(1,513,0,32,8,96,32,32)
        for kind in ('cim','sram'):
            p=residency_plan(w,kind=kind,budget_mm2=16,support=zero(),
                             programming_policy='valid_bits')
            q=residency_plan(w,kind=kind,budget_mm2=16,support=zero(),
                             programming_policy='full_macros')
            self.assertLessEqual(p.programming_bytes_total,q.programming_bytes_total)
            self.assertEqual(p.instantiated_macros,q.instantiated_macros)

    def test_invalid_budget_and_mapping(self):
        with self.assertRaises(ValueError):
            residency_plan(large(),kind='cim',budget_mm2=-1,support=zero())
        with self.assertRaises(ValueError):
            residency_plan(large(),kind='cim',budget_mm2=16,support=zero(),pinned_layers=33)
        with self.assertRaises(ValueError):
            accounting(large(),digital_reuse=999)
        with self.assertRaises(ValueError):
            ExternalArea(-1,0,0,0,0,0)

    def test_exhaustive_budget_area_no_overcommit(self):
        for prefix,budget in product((512,2048,8192),(1,2,4,8,12,16)):
            w=Workload(1,prefix,0,32,8,128,32,64)
            for kind in ('cim','sram'):
                for p in enumerate_plans(w,kind=kind,budget_mm2=budget,
                                         support=zero()):
                    self.assertLessEqual(p.complete_mm2,budget+1e-10)
                    self.assertEqual(p.instantiated_macros,
                                     (p.pinned_layers*p.layout.tile_units_per_layer+
                                      p.scratch_tile_units)*p.layout.macros_per_tile_unit)
                    if p.pinned_layers<w.layers:
                        self.assertGreater(p.scratch_tile_units,0)

    def test_best_plan_is_valid_for_each_arch(self):
        w=Workload(8,512,0,32,8,128,32,64)
        c=best_cim(w,budget_mm2=16,support=zero(),rates=r())
        d=best_digital_requirement(w,budget_mm2=16,support=zero(),rates=r(),
                                   target_s=.002,reuse=32)
        self.assertIsNotNone(c)
        self.assertIsNotNone(d)
        self.assertEqual(d[1].kind,'sram')
        self.assertFalse(d[1].qualified_digital_isoarea)

if __name__=='__main__':unittest.main()