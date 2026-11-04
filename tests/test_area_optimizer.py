import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from cim_shared_prefix.contracts import Workload, SignedMapping
from cim_shared_prefix.area_optimizer import (SRAM_7NM, CIM_7NM, SupportArea,
    compare_physical_areas, allocate_cim, allocate_sram, cim_native_floor,
    break_even_at_physical_area)
from cim_shared_prefix.digital_architecture import (RateEvidence, DigitalSchedule,
    digital_traffic, architecture_boundary)
from cim_shared_prefix.digital_threshold import EvidenceValue
from cim_shared_prefix.scalesim_bridge import gemm_topology, read_scalesim_compute_report


def a(x): return EvidenceValue(x, 'mm2', 'explicit test sensitivity', 'sensitivity')
def t(x): return EvidenceValue(x, 'seconds', 'explicit test sensitivity', 'sensitivity')
def r(x, unit): return RateEvidence(x, unit, 'explicit test sensitivity', 'sensitivity')
def ext(x=0.): return SupportArea(a(x),a(x),a(x),a(x),a(x),a(x))


class PhysicalAreaTests(unittest.TestCase):
    def setUp(self):
        self.w = Workload(1, 2048, 0, 32, 8, 128, 32, 128, key_bits=4)
        self.area = compare_physical_areas(self.w, cim_external=ext(), digital_external=ext())

    def test_complete_published_macro_sources(self):
        self.assertEqual(CIM_7NM.capacity_bits, 4096)
        self.assertEqual(CIM_7NM.area_mm2, .0032)
        self.assertAlmostEqual(SRAM_7NM.area_mm2, 262144 / 29.2e6)
        self.assertIn('10.1109', SRAM_7NM.source)
        self.assertIn('10.1109', CIM_7NM.source)

    def test_head_and_layer_tiling_no_bit_cell_density(self):
        c = allocate_cim(self.w)
        s = allocate_sram(self.w)
        self.assertEqual(c.macro_count, 32*8*(2048//16)*(128//64)*2)
        self.assertEqual(s.macro_count, 32*8*(2048*128*4//262144))
        self.assertAlmostEqual(c.area_mm2, 419.4304)
        self.assertAlmostEqual(c.area_mm2/s.area_mm2, 45.625)
        self.assertEqual(c.stored_logical_key_bits,
                         2*self.w.prefix_key_bits_per_layer*self.w.layers)

    def test_shape_padding_and_replication(self):
        w = Workload(1, 17, 0, 4, 2, 70, 2, 8)
        self.assertEqual(allocate_cim(w).macro_count, 2*2*2*2*2)
        self.assertEqual(allocate_cim(w, copies=3).macro_count,
                         3*allocate_cim(w).macro_count)
        self.assertEqual(allocate_sram(w, copies=2).macro_count,
                         2*allocate_sram(w).macro_count)
        with self.assertRaises(ValueError): allocate_cim(w, copies=0)
        with self.assertRaises(ValueError): allocate_cim(Workload(1,8,0,4,2,32,2,8,key_bits=8))

    def test_native_gops_two_ops_and_signed_mapping(self):
        floor=cim_native_floor(self.w,self.area.cim)
        self.assertEqual(floor.native_unsigned_mav_per_token,4*self.w.total_prefix_qk_mac_per_generated_token)
        self.assertEqual(floor.signed_partials_per_score,8)
        self.assertGreater(floor.best_case_macro_compute_floor_s,0)
        self.assertEqual(floor.output_score_bytes_per_token,
                         32*32*2048*4)
        self.assertTrue('NOT completed-score' in floor.note)

    def test_every_outside_macro_block_is_charged(self):
        x=compare_physical_areas(self.w,cim_external=ext(1),digital_external=ext(2))
        self.assertAlmostEqual(x.cim_total_mm2,x.cim.area_mm2+6)
        self.assertAlmostEqual(x.digital_nonmac_mm2,x.digital.area_mm2+12)
        self.assertFalse(x.qualified_digital_isoarea)
        with self.assertRaises(ValueError):
            SupportArea(t(0),a(0),a(0),a(0),a(0),a(0))
        with self.assertRaises(ValueError):
            compare_physical_areas(self.w,cim_external=ext(),digital_external=ext(),digital_copies=0)

    def test_conditional_threshold_no_area_claim(self):
        req=break_even_at_physical_area(self.w,self.area,cim_completed_score=t(.001),
                                       digital_nonmac_score=t(.0002))
        self.assertEqual(req.status,'conditional_break_even')
        self.assertFalse(req.qualified_isoarea)
        self.assertAlmostEqual(req.required_mac_per_second_per_mm2,
                               self.w.total_prefix_qk_mac_per_generated_token/(.0008*self.area.available_digital_core_mm2))


class DigitalArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.w=Workload(8,2048,0,32,8,128,32,64)
        self.area=compare_physical_areas(self.w,cim_external=ext(),digital_external=ext())

    def schedule(self,bw=1e12,reuse=1,overlap='serial'):
        return DigitalSchedule(reuse,64,r(bw,'bytes/s'),r(1e12,'bytes/s'),
                               r(1e12,'adds/s'),t(0),overlap)

    def test_memory_and_score_endpoint(self):
        traffic=digital_traffic(self.w,read_reuse=1)
        self.assertEqual(traffic.key_sram_read_bytes,
                         self.w.prefix_key_bytes_all_layers*8*self.w.gqa_ratio)
        self.assertEqual(traffic.completed_score_bytes,
                         self.w.layers*self.w.batch*self.w.query_heads*self.w.prefix_tokens*4)
        self.assertEqual(traffic.local_reduction_adds,
                         self.w.layers*self.w.prefix_scores_per_layer)
        self.assertLess(digital_traffic(self.w,read_reuse=32).key_sram_read_bytes,
                        traffic.key_sram_read_bytes)

    def test_reuse_and_sram_bw_monotonic(self):
        slow=architecture_boundary(self.w,self.area,cim_completed_score=t(.05),schedule=self.schedule())
        fast=architecture_boundary(self.w,self.area,cim_completed_score=t(.05),schedule=self.schedule(8e12,32))
        self.assertLess(fast.nonmac.nonmac_total_s,slow.nonmac.nonmac_total_s)
        self.assertLess(fast.requirement.required_mac_per_second_per_mm2,
                        slow.requirement.required_mac_per_second_per_mm2)
        self.assertFalse(fast.qualified_digital_isoarea)

    def test_negative_denominator_and_bw_boundary(self):
        b=architecture_boundary(self.w,self.area,cim_completed_score=t(1e-9),schedule=self.schedule())
        self.assertEqual(b.status,'no_compute_time')
        self.assertEqual(b.minimum_sram_bandwidth_bytes_s,float('inf'))
        with self.assertRaises(ValueError):digital_traffic(self.w,read_reuse=33)

    def test_area_monotonic_and_zero_core(self):
        b0=architecture_boundary(self.w,self.area,cim_completed_score=t(.05),schedule=self.schedule())
        area2=compare_physical_areas(self.w,cim_external=ext(),digital_external=ext(1000))
        b1=architecture_boundary(self.w,area2,cim_completed_score=t(.05),schedule=self.schedule())
        self.assertEqual(b1.status,'no_core_area')
        self.assertTrue(b0.requirement.core_area_available_mm2 > b1.requirement.core_area_available_mm2)

    def test_overlap_is_explicit(self):
        a=architecture_boundary(self.w,self.area,cim_completed_score=t(.05),schedule=self.schedule(overlap='serial'))
        b=architecture_boundary(self.w,self.area,cim_completed_score=t(.05),schedule=self.schedule(overlap='optimistic_pipeline'))
        self.assertGreaterEqual(a.nonmac.nonmac_total_s,b.nonmac.nonmac_total_s)

    def test_rate_requires_evidence(self):
        with self.assertRaises(ValueError):r(0,'bytes/s')
        with self.assertRaises(ValueError):
            DigitalSchedule(1,64,r(1,'adds/s'),r(1,'bytes/s'),r(1,'adds/s'),t(0))

    def test_scalesim_mapping_and_trace_only(self):
        self.assertEqual(gemm_topology(self.w), 'Layer, M, N, K\nqk_per_kv_head, 32, 2048, 128\n')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'COMPUTE_REPORT.csv'
            p.write_text('LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n0, 1234, 12, 74.0, 80.0, 79.0,\n')
            obs=read_scalesim_compute_report(p,workload=self.w,clock_period=t(1e-9))
            self.assertEqual(obs.total_cycles_serial_all_heads_layers,1234*8*32)
            self.assertEqual(obs.serial_execution_seconds,1234*8*32*1e-9)
            self.assertEqual(obs.mapping_efficiency_pct,80.0)
            self.assertTrue(obs.report_source.endswith('.csv'))

if __name__=='__main__':unittest.main()
