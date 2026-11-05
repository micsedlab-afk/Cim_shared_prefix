#!/usr/bin/env python3
"""Reviewer R1/R3 necessary resource-rate screen, exact original 728 coordinates.

1 TB/s is an explicitly UNQUALIFIED common upper-rate sensitivity. Program
writes and GPU partial-score link are assumed independent, enabling overlap;
max() is a lower bound for that conditional rate cap, NOT a schedule or
hardware measurement. One physical signed K+/K- image may be derived from
one packed source K4 stream; this script counts physical macro writes only.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_paper_728_audit import full_grid,AREAS_MM2,BATCHES,PREFIXES

DEFAULT_SENSITIVITY_RATE_BPS=1e12

def run(out_dir:Path, rate_Bps:float=DEFAULT_SENSITIVITY_RATE_BPS):
    if not isinstance(rate_Bps,(int,float)) or rate_Bps<=0:
        raise ValueError("positive sensitivity rate required")
    rows,main=full_grid()
    enriched=[]
    for x in rows:
        program_us=x["minimum_amortized_program_BYTES_per_token_O128"]/rate_Bps*1e6
        half_score_us=x["paper_two_d64_half_partial_BYTES_per_token"]/rate_Bps*1e6
        completed_score_us=x["completed_FP32_score_BYTES_per_token"]/rate_Bps*1e6
        macro_resource_us=x["corrected_optimistic_universal_macro_RESOURCE_floor_us_all_layers"]
        stages={"physical_signed_macro_programming":program_us,
                "paper_GPU_d64_half_partial_link":half_score_us,
                "native_unsigned_macro_resource":macro_resource_us}
        active=max(stages,key=stages.__getitem__)
        enriched.append({
            "Aadd_mm2":x["Aadd_mm2"],"Lp":x["Lp"],"B":x["B"],
            "signed_programming_mean_us_per_token_O128_at_rate_sensitivity":program_us,
            "two_d64_partial_link_us_per_token_at_rate_sensitivity":half_score_us,
            "one_completed_word_link_us_per_token_if_local_reduction_at_rate_sensitivity":completed_score_us,
            "optimistic_native_macro_resource_us_per_token":macro_resource_us,
            "independent_resource_lower_floor_us_per_token_at_rate_sensitivity":stages[active],
            "largest_necessary_resource_at_rate_sensitivity":active,
            "source_evidence":"signed native macros + exactly original 728 design grid; UNQUALIFIED 1TBps independent port caps",
            "full_execution_tpot_us":None,
            "qualified_a100_INT4_speedup":None,
        })
    names=list(enriched[0])
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/"ojcas_728_conditional_1TBps_resource_screen.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=names)
        w.writeheader();w.writerows(enriched)
    r=[
        x for x in enriched if x["Lp"]==2048 and x["Aadd_mm2"]==8
    ]
    s={
      "schema":"ojcas_728_rate_capped_necessary_resource_sensitivity/v1",
      "source_original_grid_coordinates":len(rows),
      "rate_sensitivity_Bps":rate_Bps,
      "rate_provenance":"hypothetical independent UPPER rate caps, no measured 7nm programming bus or GPU link",
      "resource_combination":"maximum of independently unavoidable times, not sum; all other costs omitted",
      "reference_P2048_A8_rows":r,
      "reference_P2048_A8_winner_counts":dict(Counter(x["largest_necessary_resource_at_rate_sensitivity"] for x in r)),
      "full_grid_winner_counts":dict(Counter(x["largest_necessary_resource_at_rate_sensitivity"] for x in enriched)),
      "conditional_full_TPOT_qualified":False,
      "hardware_speedup_qualified":False,
      "limitations":[
        "does not have validated original paper A100 INT4 GPU stage latency",
        "hardware signed K physical writes are not the same as required HBM input bytes",
        "macro resource floor omits access-port constraints, bank residency and all other system overhead",
        "partial-score transfer uses two d64 halves as in submitted paper; local merge would need hardware",
        "1TBps per-interface capability is unqualified and must not be stated as real device bandwidth",
        "selected bottleneck is only among three necessary modeled resources",
      ]
    }
    (out_dir/"ojcas_728_conditional_1TBps_resource_screen.json").write_text(
        json.dumps(s,indent=2,sort_keys=True)+"\n")
    print("OJCAS_RESOURCE_SCREEN_SUMMARY "+json.dumps({
        "count":len(enriched),"rate_Bps":rate_Bps,
        "P2048_A8_B1":next(x for x in r if x["B"]==1),
        "P2048_A8_B4":next(x for x in r if x["B"]==4),
        "P2048_A8_B128":next(x for x in r if x["B"]==128),
        "original_INT4_speedup_requalified":False,
    },sort_keys=True),flush=True)
    return enriched,s

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--rate-Bps",type=float,default=DEFAULT_SENSITIVITY_RATE_BPS)
    a=p.parse_args()
    if a.rate_Bps!=DEFAULT_SENSITIVITY_RATE_BPS:
        raise ValueError("output name carries 1TBps; use declared default only")
    run(a.out_dir,a.rate_Bps)
