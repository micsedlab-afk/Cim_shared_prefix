#!/usr/bin/env python3
"""Original OJCAS S2 temporal reuse sensitivity, no original model replacement.

O=output steps per batch; R=consecutive batches reusing the same fixed prefix.
An admitted SRAM-CIM resident layer is retained between R batches. No new
prefix, KV expiration, interleaving, other prefixes, or resident eviction.
Only QK source HBM, row program and native macro cycles are compared.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_manuscript_baseline import AREAS_MM2,BATCHES,PREFIXES
from cim_shared_prefix.ojcas_published_placement_delta import optimize_published_source

OUTPUT_LENGTHS=(1,32,128,512)
PREFIX_REUSE_BATCHES=(1,8)


def run(out_dir:Path):
    rows=[]
    sections=[]
    for o in OUTPUT_LENGTHS:
        for r in PREFIX_REUSE_BATCHES:
            N=o*r
            by_area=defaultdict(lambda:{"cases":0,"source_positive_cases":0,
                                        "total_identified_three_component_seconds_saved":0.})
            for a in AREAS_MM2:
                for p in PREFIXES:
                    for b in BATCHES:
                        ctrl,sel,_=optimize_published_source(b,p,a,
                            output_batch_steps=N)
                        saved=ctrl.three_component_serial_seconds-sel.three_component_serial_seconds
                        assert saved>=-1e-12
                        c=by_area[str(a)]
                        c["cases"]+=1
                        c["source_positive_cases"]+=int(saved>1e-18)
                        c["total_identified_three_component_seconds_saved"]+=max(0.,saved)
                        rows.append({
                            "B":b,"Lp":p,"Aadd_mm2":a,"output_tokens_per_batch":o,
                            "same_prefix_consecutive_batches":r,
                            "total_decode_batch_steps":N,
                            "selected_pinned_layers":sel.pin_layers,
                            "original_macro_engine_count":128,
                            "original_paper_728_speedup_unmodified":True,
                            "baseline_three_component_us_per_step":
                                ctrl.three_component_serial_seconds*1e6/N,
                            "selected_three_component_us_per_step":
                                sel.three_component_serial_seconds*1e6/N,
                            "three_component_source_savings_us_per_step":
                                max(0.,saved)*1e6/N,
                            "three_component_source_reduction_fraction":
                                max(0.,saved)/ctrl.three_component_serial_seconds,
                            "has_source_serial_improvement":saved>1e-18,
                            "measured_A100_INT4_TPOT":None,
                        })
            sections.append({
                "O":o,"reuse_batches":r,"total_step_count":N,
                "positive_coordinates":sum(v["source_positive_cases"] for v in by_area.values()),
                "by_area":dict(by_area)
            })
    assert len(rows)==728*len(OUTPUT_LENGTHS)*len(PREFIX_REUSE_BATCHES)
    manifest={
        "schema":"ojcas_original_S2_output_and_prefix_reuse_sensitivity/v1",
        "hardware":"submitted 128-macro SRAM-CIM engines, published HBM and timing inputs, no new sign banks",
        "scope":"only previously timed S2 K4 supply, row programming and unsigned native macro computation; unchanged remainder omitted",
        "assumption":"one fixed shared prefix persists across R sequential batches of equal B, each outputting O tokens; no intervening unrelated request or eviction",
        "O_values":OUTPUT_LENGTHS,
        "prefix_reuse_batch_values":PREFIX_REUSE_BATCHES,
        "original_728_coordinates_per_scenario":728,
        "total_scenarios":len(rows),
        "sections":sections,
        "experiment":"paper-sourced deterministic serial stage timing overlay; not physical runtime, not end-to-end performance",
        "historical_paper_results_preserved":True,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/"ojcas_original_S2_output_reuse_5824.csv").open("w",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out_dir/"ojcas_original_S2_output_reuse_5824.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("OJCAS_ORIGINAL_S2_REUSE "+json.dumps({
        "total":len(rows),
        "positive_by_O_reuse":[{"O":s["O"],"R":s["reuse_batches"],
                                 "positive":s["positive_coordinates"],
                                 "by_area":{a:v["source_positive_cases"]
                                            for a,v in s["by_area"].items()}}
                                 for s in sections],
    },sort_keys=True,allow_nan=False),flush=True)
    return rows,manifest

if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    run(p.parse_args().out_dir)
