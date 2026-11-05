"""Submitted OJCAS S0/S1/S2 model: immutable manuscript accounting.

This module reconstructs *reported manuscript inputs and aggregates*, not
unknown per-coordinate timing. Signed Q4/K4 decomposition and digital/other
processing are modeled outside the 128-macro core as stated in the paper;
the 256-macro signed-bank study is a separate alternative mapping scenario,
not a correction to this experimental design.

Source: OJCAS-2026-0120, Tables 2-5, Eqs. (5)-(10), 03 Nov 2026 reviews.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
from math import floor, isclose

BATCHES = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
PREFIXES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
AREAS_MM2 = (1, 2, 3, 4, 6, 8, 12, 16)
N_LAYERS=32
N_QUERY_HEADS=32
N_KV_HEADS=8
HEAD_DIM=128
KEY_BITS=4
SUFFIX_TOKENS=128
CIM_MACROS_PER_ENGINE=128
CIM_MACRO_AREA_MM2=0.0032
CIM_ENGINE_AREA_MM2=CIM_MACROS_PER_ENGINE*CIM_MACRO_AREA_MM2
S1_HD_BITS_PER_MACRO=512*512
S1_HD_DENSITY_BITS_PER_MM2=29.2e6
S1_DP_BYTES_PER_MACRO=4096
S1_DP_DENSITY_BITS_PER_MM2=6.0e6
PAPER_FAVORABLE_PER_91=(0,11,14,22,52,60,60,57)
PAPER_MAX_QK_STAGE_SPEEDUP=1.97
S1_HD_READ_PJ_PER_BIT=0.090625
S1_HD_WRITE_PJ_PER_BIT=0.1421875
S1_DP_READ_PJ_PER_BIT=0.053
S1_DP_WRITE_PJ_PER_BIT=0.092
S2_PJ_PER_PUBLISHED_MACRO_ACCESS=5.835
ORIGINAL_PARTIAL_ENERGY_OPERATIONS_AT_B1=128*(N_QUERY_HEADS//N_KV_HEADS)
S1_K_TILE_BYTES=N_KV_HEADS*128*HEAD_DIM*KEY_BITS//8
assert CIM_ENGINE_AREA_MM2==0.4096
assert S1_K_TILE_BYTES==65536
assert len(BATCHES)*len(PREFIXES)*len(AREAS_MM2)==728


@dataclass(frozen=True)
class PaperCapacity:
    area_mm2:int
    s2_original_engine_count:int
    s1_hd_macro_count:int
    s1_hd_capacity_bytes:int
    s1_dp_macro_count:int
    s1_dp_capacity_bytes:int

    def dict(self):return asdict(self)


def paper_capacity(area_mm2:int)->PaperCapacity:
    if area_mm2 not in AREAS_MM2:raise ValueError("outside paper's fixed area sweep")
    hd_area=S1_HD_BITS_PER_MACRO/S1_HD_DENSITY_BITS_PER_MM2
    dp_area=S1_DP_BYTES_PER_MACRO*8/S1_DP_DENSITY_BITS_PER_MM2
    h=floor((area_mm2+1e-12)/hd_area)
    d=floor((area_mm2+1e-12)/dp_area)
    return PaperCapacity(
        area_mm2=area_mm2,
        s2_original_engine_count=floor((area_mm2+1e-12)/CIM_ENGINE_AREA_MM2),
        s1_hd_macro_count=h,
        s1_hd_capacity_bytes=h*(S1_HD_BITS_PER_MACRO//8),
        s1_dp_macro_count=d,
        s1_dp_capacity_bytes=d*S1_DP_BYTES_PER_MACRO,
    )


def paper_partial_energy()->dict:
    """Manuscript Eq. 10 exactly; no added fourfold multiplier.

    This formula is the manuscript's model-defined partial macro-energy metric,
    whose scope excludes extra-system sign reconstruction/programming energy.
    """
    per_tile_bits=S1_K_TILE_BYTES*8
    e_hd=per_tile_bits*(S1_HD_READ_PJ_PER_BIT+S1_HD_WRITE_PJ_PER_BIT)
    e_dp=per_tile_bits*(S1_DP_READ_PJ_PER_BIT+S1_DP_WRITE_PJ_PER_BIT)
    e_s2=ORIGINAL_PARTIAL_ENERGY_OPERATIONS_AT_B1*S2_PJ_PER_PUBLISHED_MACRO_ACCESS
    return {
        "s1_hd_write_and_read_pJ":e_hd,
        "s1_dp_write_and_read_pJ":e_dp,
        "s2_macro_compute_at_B1_pJ":e_s2,
        "crossing_B_s1_hd":e_hd/e_s2,
        "crossing_B_s1_dp":e_dp/e_s2,
        "scope":"submitted Eq.10: SRAM write+read versus published S2 macro compute only",
        "external_sign_reconstruction_energy":"not included in Eq.10, separately accounted if new hardware boundary added",
    }


def manuscript_grid()->tuple[list[dict],dict]:
    """Exactly preserve all original axes; per-point historical latency is unknown.

    Aggregate Table 5 numbers are present as manuscript evidence. Individual
    winners and time values cannot be inferred from those aggregate counts.
    """
    rows=[]
    for a,p,b in product(AREAS_MM2,PREFIXES,BATCHES):
        cap=paper_capacity(a)
        p_tiles=(p+127)//128
        s1_hd_resident=min(N_LAYERS*p_tiles,cap.s1_hd_capacity_bytes//S1_K_TILE_BYTES)
        s1_dp_resident=min(N_LAYERS*p_tiles,cap.s1_dp_capacity_bytes//S1_K_TILE_BYTES)
        rows.append({
            "B":b,"Lp":p,"Lu":SUFFIX_TOKENS,"Aadd_mm2":a,
            "original_s2_engine_count":cap.s2_original_engine_count,
            "original_s2_engine_area_mm2":CIM_ENGINE_AREA_MM2,
            "s1_hd_resident_tile_fraction":s1_hd_resident/(N_LAYERS*p_tiles),
            "s1_dp_resident_tile_fraction":s1_dp_resident/(N_LAYERS*p_tiles),
            "individual_s2_wins":None,
            "individual_s2_stage_speedup":None,
            "status":"submitted-model grid/area facts; no per-coordinate timing reconstructed",
        })
    assert len(rows)==728
    e=paper_partial_energy()
    assert isclose(e["crossing_B_s1_hd"],40.85689802913453,rel_tol=1e-12)
    assert isclose(e["crossing_B_s1_dp"],25.44644387317909,rel_tol=1e-12)
    return rows,{
        "schema":"ojcas_2026_0120_submitted_baseline_lock/v1",
        "paper":"Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?",
        "source":"submitted manuscript, original Tables 2-5 and Eqs. 5-10",
        "original_grid_size":728,
        "axes":{"batch":BATCHES,"prefix":PREFIXES,"added_macro_area_mm2":AREAS_MM2,
                "suffix_tokens":SUFFIX_TOKENS},
        "S0":"A100 80GB original modeled 4-bit path; not a measured GPU kernel",
        "S1":["S1-HD","S1-DP"],
        "S2":"submitted 128-macro per engine 7nm SRAM-CIM shared-prefix QK score model",
        "signed_q4k4_boundary":"sign decomposition and reconstruction processed elsewhere/outside 128-macro core; original engine area and Eq.10 retained",
        "original_s2_macro_per_engine":CIM_MACROS_PER_ENGINE,
        "original_s2_engine_area_mm2":CIM_ENGINE_AREA_MM2,
        "paper_favorable_counts_by_area":dict(zip(map(str,AREAS_MM2),PAPER_FAVORABLE_PER_91)),
        "paper_favorable_coverage_pct_by_area":dict(zip(map(str,AREAS_MM2),[100*n/91 for n in PAPER_FAVORABLE_PER_91])),
        "paper_max_modeled_QK_stage_speedup":PAPER_MAX_QK_STAGE_SPEEDUP,
        "paper_macro_partial_energy":e,
        "total_reported_favorable_stage_points":sum(PAPER_FAVORABLE_PER_91),
        "source_derived_number_of_individually_qualified_stage_times":0,
        "external_sign_handling":"must be described/quantified separately without silently applying new storage replication to original S2",
        "scope_limits":["speedups apply to original modeled shared-prefix QK score path only",
                        "macro-only partial energy excludes overall system operations",
                        "not full decoder TPOT and not measured CUDA timing"],
        "separate_counterfactual":"256-macro signed K+/K- layout and 4x native macro access scenario belongs to earlier optional audit; not a correction to submitted data",
    }


def saved_stage_speedup_margin(speedup:float)->float:
    """Dimensionless max *incremental* overhead budget fraction of Tconv.

    For a specified historical modeled Tconv/Ts2 = speedup, the same stage
    remains favorable only if extra nonoverlapped delta < Tconv - Ts2.
    Does not say how many us were available, nor whether hardware can supply it.
    """
    if not isinstance(speedup,(int,float)) or speedup<=1:
        raise ValueError("requires already favorable modeled speedup > 1")
    return 1.0-1.0/speedup
