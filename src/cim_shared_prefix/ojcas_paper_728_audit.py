"""Exact-grid requalification of OJCAS-2026-0120 hardware assumptions.

This is a source-grounded *physical accounting audit*, not a reproduction of
paper S0/S1/S2 latency or claimed speedup: the original A100 INT4 operator
execution and original S1/S2 latency model are not available here.

Original paper: Sec. III, Tables 2-3, Eqs. 5-10, 2026 submission.
7-nm CIM source: Dong/Sinangil ISSCC 2020 / JSSC 2021.
Signed 2-bank, 4 unsigned calls mapping: accepted repository contract.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, floor
from itertools import product

from .area_optimizer import CIM_7NM, SRAM_7NM, CIM_PUBLISHED_ACCESS_S
from .contracts import Workload, SignedMapping

BATCHES = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
PREFIXES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
AREAS_MM2 = (1, 2, 3, 4, 6, 8, 12, 16)
SUFFIX = 128
OUTPUT = 128
LAYERS = 32
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SCORE_WORD_BYTES = 4
NATIVE_TILE_TOKENS = 128
SRAM_K_TILE_BYTES = KV_HEADS * NATIVE_TILE_TOKENS * HEAD_DIM * 4 // 8  # 64 KiB
ORIGINAL_PAPER_ENGINE_MACROS = 128
# Same eight KV heads, two 64-row slabs, eight 16-score groups.
# An unsigned-only 128-token tile is 128 physical macros.
SIGNED_ENGINE_MACROS = ORIGINAL_PAPER_ENGINE_MACROS * 2
assert SRAM_K_TILE_BYTES == 65536
assert SIGNED_ENGINE_MACROS == 256

# Explicit independent published 7-nm 2RW 8T S1-DP density in manuscript.
DP_SRAM_CAPACITY_BITS = 4096*8
DP_SRAM_MACRO_AREA_MM2 = DP_SRAM_CAPACITY_BITS / 6.0e6
DP_SRAM_SOURCE = "Yokoyama et al., JSSC 2023, 4KiB 2RW 8T, 6.00 Mbit/mm2: submitted OJCAS Table 2"

# From the actual submitted paper's Tab. 5. These are archival, NOT reverified.
PAPER_S2_WINNING_POINTS_PER_91 = (0, 11, 14, 22, 52, 60, 60, 57)
PAPER_MAX_MODELED_QK_SPEEDUP = 1.97
PAPER_S2_NOMINAL_PJ_PER_ACCESS = 5.835
PAPER_S2_HIGH_PJ_PER_ACCESS = 7.8
PAPER_S1_HD_PJ_PER_BIT_READ = 0.090625
PAPER_S1_HD_PJ_PER_BIT_WRITE = 0.1421875
PAPER_S1_DP_PJ_PER_BIT_READ = 0.053
PAPER_S1_DP_PJ_PER_BIT_WRITE = 0.092
PAPER_ADDITIONAL_SUPPORT_MM2 = 0.5  # explicitly unqualified sensitivity
# Missing full calibrated source of original INT4 A100 and S1/S2 latencies.
A100_INT4_NATIVE_PROVENANCE = "UNRECOVERED_ORIGINAL_INT4_LATENCY_SOURCE"


@dataclass(frozen=True)
class AreaCapacity:
    area_mm2: int
    s1_HD_macro_count: int
    s1_HD_capacity_bytes: int
    s1_DP_macro_count: int
    s1_DP_capacity_bytes: int
    paper_unsigned_tile_engine_count: int
    corrected_signed_tile_engine_count_macro_only: int
    corrected_signed_tile_engine_count_with_0p5mm2_support: int
    paper_macro_area_per_engine_mm2: float
    signed_macro_area_per_engine_mm2: float
    cim_physical_macros_max_no_support: int
    cim_physical_signed_K_storage_bytes_max_no_support: int

    def record(self):
        return asdict(self)


def area_capacity(area_mm2: int) -> AreaCapacity:
    if area_mm2 not in AREAS_MM2:
        raise ValueError("only paper's actual incremental-area budgets supported")
    hd_macros = floor((area_mm2 + 1e-12)/SRAM_7NM.area_mm2)
    dp_macros = floor((area_mm2 + 1e-12)/DP_SRAM_MACRO_AREA_MM2)
    paper_area = ORIGINAL_PAPER_ENGINE_MACROS * CIM_7NM.area_mm2
    signed_area = SIGNED_ENGINE_MACROS * CIM_7NM.area_mm2
    cmax = floor((area_mm2 + 1e-12)/CIM_7NM.area_mm2)
    return AreaCapacity(
        area_mm2=area_mm2,
        s1_HD_macro_count=hd_macros,
        s1_HD_capacity_bytes=hd_macros*(SRAM_7NM.capacity_bits//8),
        s1_DP_macro_count=dp_macros,
        s1_DP_capacity_bytes=dp_macros*(DP_SRAM_CAPACITY_BITS//8),
        paper_unsigned_tile_engine_count=floor((area_mm2+1e-12)/paper_area),
        corrected_signed_tile_engine_count_macro_only=floor((area_mm2+1e-12)/signed_area),
        corrected_signed_tile_engine_count_with_0p5mm2_support=max(
            0, floor((area_mm2-PAPER_ADDITIONAL_SUPPORT_MM2+1e-12)/signed_area)
        ),
        paper_macro_area_per_engine_mm2=paper_area,
        signed_macro_area_per_engine_mm2=signed_area,
        cim_physical_macros_max_no_support=cmax,
        cim_physical_signed_K_storage_bytes_max_no_support=cmax*(CIM_7NM.capacity_bits//8),
    )


def energy_crossovers() -> dict:
    # Original Eq.10 numerator: 128*g*B = 512B 5.835pJ
    # Actual native signed Q4xK4: four unsigned 4b products per logical MAC.
    # Two physical signed K banks/engine; four calls, 4x logical access count.
    bits=SRAM_K_TILE_BYTES*8
    ehd=bits*(PAPER_S1_HD_PJ_PER_BIT_READ+PAPER_S1_HD_PJ_PER_BIT_WRITE)
    edp=bits*(PAPER_S1_DP_PJ_PER_BIT_READ+PAPER_S1_DP_PJ_PER_BIT_WRITE)
    old_calls=128*(QUERY_HEADS//KV_HEADS)
    signed_calls=old_calls*SignedMapping().array_evaluations_per_logical_operation
    return {
        "reference":"submitted OJCAS Table 2 and Eq.10; identical one-S1-write-plus-one-read partial macro-energy definition",
        "paper_native_unsigned_calls_per_128token_tile_B1":old_calls,
        "signed_native_unsigned_calls_per_128token_tile_B1":signed_calls,
        "access_count_correction_factor":signed_calls/old_calls,
        "S1_HD_write_plus_read_pJ":ehd,
        "S1_DP_write_plus_read_pJ":edp,
        "paper_CIM_nominal_pJ_B1":old_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS,
        "corrected_CIM_nominal_pJ_B1":signed_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS,
        "original_HD_nominal_energy_crossing_B":ehd/(old_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS),
        "original_DP_nominal_energy_crossing_B":edp/(old_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS),
        "corrected_HD_nominal_energy_crossing_B":ehd/(signed_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS),
        "corrected_DP_nominal_energy_crossing_B":edp/(signed_calls*PAPER_S2_NOMINAL_PJ_PER_ACCESS),
        "corrected_HD_high_access_energy_crossing_B":ehd/(signed_calls*PAPER_S2_HIGH_PJ_PER_ACCESS),
        "corrected_DP_high_access_energy_crossing_B":edp/(signed_calls*PAPER_S2_HIGH_PJ_PER_ACCESS),
        "limitations":"MACRO-only partial energy, no programming, sign-split driver, supply, ADC extra (already on macro), GPU energy, or system energy; conversion assumes 5.835 pJ is per native unsigned access at the published operating point",
    }


def coordinate(batch: int, prefix: int, area_mm2: int) -> dict:
    if batch not in BATCHES or prefix not in PREFIXES or area_mm2 not in AREAS_MM2:
        raise ValueError("outside exactly the submitted 728-point design grid")
    w=Workload(batch,prefix,SUFFIX,QUERY_HEADS,KV_HEADS,HEAD_DIM,LAYERS,OUTPUT,key_bits=4)
    a=area_capacity(area_mm2)
    ntiles=ceil(prefix/NATIVE_TILE_TOKENS)
    old_e=a.paper_unsigned_tile_engine_count
    corrected_e=a.corrected_signed_tile_engine_count_macro_only
    # Per 128-token/one-layer signed Q4xK4: 8 KV heads ×
    # 2 64-row half groups × 8 groups of 16 scores × four +/- unsigned
    # products × 4 query-heads per KV head × B = 2048*B calls.
    native_calls_per_128token_tile_per_layer=(
        4*batch*QUERY_HEADS*ceil(HEAD_DIM/64)*ceil(NATIVE_TILE_TOKENS/16)
    )
    all_calls_layer=native_calls_per_128token_tile_per_layer*ntiles
    full_layer_signed_key_bytes=(
        prefix*KV_HEADS*HEAD_DIM*w.key_bits//8 *
        SignedMapping().key_storage_factor
    )
    physical_signed_K_bytes_all_layers=LAYERS*full_layer_signed_key_bytes
    source_unique_K4_bytes_all_layers=w.prefix_key_bytes_all_layers
    resident_capacity_upper=a.cim_physical_signed_K_storage_bytes_max_no_support
    min_reprogram=max(0,physical_signed_K_bytes_all_layers-resident_capacity_upper)
    # A separate physical-write lower bound; source packed K4 could be
    # transferred once and split locally, so this MUST NOT be mistaken for HBM.
    min_cold_program=physical_signed_K_bytes_all_layers
    program_total=min_cold_program+(OUTPUT-1)*min_reprogram
    # Physical K rows are shared across B; completed scores scale with B.
    scores_per_step=LAYERS*batch*QUERY_HEADS*prefix
    score_bytes=scores_per_step*SCORE_WORD_BYTES
    # The submitted paper puts d64 partial merging onto the GPU -> two
    # word-sized d64 partials cross the link unless merged on accelerator.
    manuscript_two_half_output_bytes=2*score_bytes
    # If all four signed products of *both* d64 halves leave the
    # CIM without local reduction, eight words/score would be needed.
    raw_eight_signed_partial_output_bytes=8*score_bytes
    # Max useful full 128-token engines. This is an optimistic complete
    # engine-wave *schedule*, never a universal physical lower bound.
    waves=ceil(ntiles/corrected_e) if corrected_e else None
    old_waves=ceil(ntiles/old_e) if old_e else None
    ideal_full_engine_cycles = (waves*2*batch*(QUERY_HEADS//KV_HEADS)
                                if waves is not None else None)
    ideal_macro_time = (
        LAYERS*ideal_full_engine_cycles*CIM_PUBLISHED_ACCESS_S
        if ideal_full_engine_cycles is not None else None)
    # Actual hard necessary resource bound over all macros, with even
    # ideal use of every macro at full throughput: cannot prove attainable.
    native_calls_layer=all_calls_layer
    macro_resource_floor=LAYERS*ceil(
        native_calls_layer/a.cim_physical_macros_max_no_support
    )*CIM_PUBLISHED_ACCESS_S
    hd_valid_tiles=min(ntiles*LAYERS,a.s1_HD_capacity_bytes//SRAM_K_TILE_BYTES)
    dp_valid_tiles=min(ntiles*LAYERS,a.s1_DP_capacity_bytes//SRAM_K_TILE_BYTES)
    return {
        "B":batch,"Lp":prefix,"Lu_paper_fixed":SUFFIX,"O_for_cold_amortization":OUTPUT,
        "Aadd_mm2":area_mm2,
        "paper_engine_count_unsigned_128mac":old_e,
        "corrected_full_engine_count_signed_256mac_no_support":corrected_e,
        "paper_128token_tile_waves":old_waves,
        "corrected_signed_128token_tile_waves_macro_only":waves,
        "corrected_ideal_128token_wave_compute_us_all_layers":(
            ideal_macro_time*1e6 if ideal_macro_time is not None else None),
        "corrected_optimistic_universal_macro_RESOURCE_floor_us_all_layers":macro_resource_floor*1e6,
        "physical_signed_native_unsigned_calls_per_layer":native_calls_layer,
        "physical_signed_native_unsigned_calls_per_token_all_layers":LAYERS*native_calls_layer,
        "source_unique_K4_bytes_all_layers":source_unique_K4_bytes_all_layers,
        "physical_Kplus_Kminus_bytes_all_layers":physical_signed_K_bytes_all_layers,
        "macro_key_residency_bytes_optimistic_no_support":resident_capacity_upper,
        "minimum_warm_program_BYTES_per_token":min_reprogram,
        "minimum_cold_program_BYTES_one_prefix":min_cold_program,
        "minimum_amortized_program_BYTES_per_token_O128":program_total/OUTPUT,
        "completed_FP32_score_BYTES_per_token":score_bytes,
        "paper_two_d64_half_partial_BYTES_per_token":manuscript_two_half_output_bytes,
        "eight_raw_signed_d64_partial_BYTES_per_token":raw_eight_signed_partial_output_bytes,
        "S1_HD_resident_fraction_128token_tiles":hd_valid_tiles/(ntiles*LAYERS),
        "S1_DP_resident_fraction_128token_tiles":dp_valid_tiles/(ntiles*LAYERS),
        "paper_S2_favorable_stage_coordinate":None,
        "corrected_qualified_S2_speedup":None,
        "primary_A100_INT4_source_operator_time_s":None,
        "endpoint":"original shared-prefix QK^T; two d64 score halves merged on GPU in submitted paper",
        "classification":"source macro and mathematical resource bounds only; active port/IO/programming rates not physically verified",
    }


def full_grid() -> tuple[list[dict],dict]:
    rows=[coordinate(b,p,a) for a,p,b in product(AREAS_MM2,PREFIXES,BATCHES)]
    assert len(rows)==728
    caps=[area_capacity(a).record() for a in AREAS_MM2]
    cross=energy_crossovers()
    return rows,{
        "schema":"ojcas_original_728_signed_physical_requalification/v1",
        "submitted_manuscript":"OJCAS-2026-0120; Where Is the Sweet Spot for SRAM-CIM in Shared-Prefix LLM Attention?",
        "original_manuscript_evidence":{
            "grid_size":728,"batches":BATCHES,"prefixes":PREFIXES,"areas_mm2":AREAS_MM2,
            "suffix_tokens":SUFFIX,
            "reported_S2_win_counts_by_area_PAPER_ONLY":dict(zip(map(str,AREAS_MM2),PAPER_S2_WINNING_POINTS_PER_91)),
            "reported_max_modeled_QK_speedup_PAPER_ONLY":PAPER_MAX_MODELED_QK_SPEEDUP,
            "comparison_scope":"stage only, neither corrected nor revalidated here"},
        "source_evidence":{
            "CIM_macro_area_mm2":CIM_7NM.area_mm2,
            "CIM_native_unsigned_access_ns":CIM_PUBLISHED_ACCESS_S*1e9,
            "SRAM_HD_macro_area_mm2":SRAM_7NM.area_mm2,
            "SRAM_DP_macro_area_mm2_paper_source":DP_SRAM_MACRO_AREA_MM2,
            "signed_key_banks":SignedMapping().key_storage_factor,
            "physical_native_unsigned_evaluations_per_signed_Q4K4":SignedMapping().array_evaluations_per_logical_operation,
            "original_A100_INT4_provenance":A100_INT4_NATIVE_PROVENANCE,
            "A100_FP16_is_separate_study":True},
        "physical_capacity_by_paper_area":caps,
        "partial_macro_energy_recalculation":cross,
        "executed_coordinates":len(rows),
        "qualified_INT4_A100_GPU_stage_times":0,
        "requalified_original_S2_speedup_coordinates":0,
        "qualified_full_decoder_TPOT":0,
        "full_chip_isoarea_qualified":False,
        "critically_missing":[
            "original A100 INT4 operator implementation and exact timing",
            "original conventional S1 latency source for same score endpoint",
            "original S2 728-point numerical latency input files",
            "measured effective concurrent unsigned CIM accesses, active ports and programming interface",
            "physical control/digital peripheral, delivery bandwidth and area",
            "matched complete decoder and numerically qualified GPU path"],
        "caveat_programming":"Physical signed K+/K- programming bytes count two physical copies; HBM source can carry one packed K4 copy followed by local splitting. Never label physical programming bytes as required HBM bytes.",
        "caveat_macro_resource":"Physical macro resource time is an optimistic necessary floor; full 128-token engine-wave time assumes that complete 256-macro engines operate concurrently at ideal access time, not validated silicon system throughput.",
        "caveat_partial_output":"Original GPU d64-half merge requires two partial words per score; completed one-word score requires offloaded local merge hardware; signed raw partial output requires up to eight words/score.",
        "caveat_energy":"Only the manuscript Eq.10 partial macro-energy ratio, corrected for unsigned access count; NOT system energy or tokens/J.",
    }
