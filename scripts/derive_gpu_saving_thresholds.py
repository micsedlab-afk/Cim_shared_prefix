#!/usr/bin/env python3
"""Translate the existing 2304 source-macro study into direct GPU QK-saving thresholds.

This uses existing executed coordinates, not a new artificial 2-ms deadline.
The boundary is measured GPU FULL attention - measured/model GPU REMAINDER.
Physical CIM interface assumptions are retained unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.gpu_saving_boundary import from_integrated_architecture_row

KEEP=("model","revision","budget_mm2","support_mm2_sensitivity","batch",
      "prefix_tokens_original","suffix_tokens","output_tokens",
      "visible_prefix_first","visible_prefix_last","source_bandwidth_Bps_sensitivity",
      "physical_macros_first","pinned_layers_first","scratch_tile_units_first",
      "sliding_reprogram_policy")


def execute(input_csv: Path, output_csv: Path, output_summary: Path):
    with input_csv.open(newline="") as fd:
        source=list(csv.DictReader(fd))
    if not source:
        raise ValueError("source integrated architecture dataset is empty")
    rows=[]
    for record in source:
        threshold=from_integrated_architecture_row(record)
        data={k:record[k] for k in KEEP}
        data.update(asdict(threshold))
        data["matched_gpu_reference_time_s"]=None
        data["gpu_minus_remainder_s"]=None
        data["competitive_speedup"]=None
        rows.append(data)
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    with output_csv.open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    def example(model,prefix,batch,output,budget,support,bw):
        xs=[x for x in rows if
            x["model"]==model and int(x["prefix_tokens_original"])==prefix and
            int(x["batch"])==batch and int(x["output_tokens"])==output and
            float(x["budget_mm2"])==budget and
            float(x["support_mm2_sensitivity"])==support and
            float(x["source_bandwidth_Bps_sensitivity"])==bw]
        if len(xs)!=1:
            raise AssertionError(f"reference coordinate missing/duplicated: {len(xs)}")
        return xs[0]
    reference=example("meta-llama/Llama-3.1-8B",2048,1,128,16,0.5,1e12)
    brief={
        "schema":"cim-shared-prefix-required-gpu-qk-saving/v1",
        "input":str(input_csv),
        "source_rows":len(source),
        "derived_rows":len(rows),
        "method":"For the executed no-overlap CIM QK schedule: GPU full-attention time minus GPU post-prefix-score remainder must exceed CIM completed-prefix-score time / decoder layer.",
        "reference_coordinate":{
            "model":reference["model"],"P":2048,"B":1,"O":128,
            "total_CIM_area_budget_mm2":16,
            "external_area_assumption_mm2":0.5,
            "source_interface_sensitivity_Bps":1e12,
            "necessary_floor_us_per_layer":reference["optimistic_floor_us_per_layer"],
            "serial_policy_break_even_saving_us_per_layer":reference["required_gpu_attention_saving_us_per_layer"],
            "score_delivery_us_per_layer":reference["score_delivery_us_per_layer"],
            "source_programming_us_per_layer":reference["source_plus_program_us_per_layer"],
        },
        "limitations":[
            "No GPU latency is synthesized; GPU/reference columns remain null.",
            "Threshold requires the same final full attention-context endpoint.",
            "The candidate serial schedule does not credit overlap between GPU suffix and CIM execution.",
            "All macro port counts, supporting area, bandwidth and external digital rates remain architectural sensitivities.",
            "Q4K4O4 CIM numerical quality differs from FP16/BF16 full-GPU attention."
        ]
    }
    output_summary.write_text(json.dumps(brief,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"rows":len(rows),"reference_saving_us_per_layer":
                      reference["required_gpu_attention_saving_us_per_layer"],
                      "reference_floor_us_per_layer":
                      reference["optimistic_floor_us_per_layer"],
                      "output":str(output_csv)},sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("results/integration/architecture_1_16mm2.csv"))
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--summary",type=Path,required=True)
    a=ap.parse_args()
    execute(a.input,a.csv,a.summary)
