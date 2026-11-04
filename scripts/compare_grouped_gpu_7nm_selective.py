#!/usr/bin/env python3
"""Compare physical signed 7nm pinned-CIM wave policy with source-native grouped GPU QK.

The source GPU path batches BxG queries against ONE K matrix per KV head,
as required for a Hydragen-style shared-prefix GPU baseline.

The result is a conditional *prefix QK stage* comparison. Neither GPU
softmax/PV nor full decoder is generated or timed here; no full TPOT claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.integrated_schedule import SerialRates
from cim_shared_prefix.selective_pipeline import resident_pipeline

SRC=Path("results/integration/native_grouped_a100_fp16_prefix_roofline.json")


def execute(path:Path, csv_out:Path, json_out:Path, *,
            budget_mm2:float=16, support_mm2:float=.5,
            operand_stage_mm2:float=.1, ports:int=256, bw:float=1e12,
            O:int=128, extra_setup_us:float=0):
    data=json.loads(path.read_text())
    if data.get("schema")!="cim-shared-prefix-native-grouped-A100-FP16-roofline/v1":
        raise ValueError("GPU prefix must originate in original grouped native source model")
    if len(data.get("results",[]))!=6:
        raise ValueError("incomplete source grouped batch series")
    rates=SerialRates(bw,bw,bw,bw,1e12,1e12,
                      source="source-native grouped GPU QK vs published signed CIM; physical rates sensitivities")
    rows=[]
    for g in data["results"]:
        B=g["batch"]
        if g["queries_per_KV_head"]!=B*4 or g["KV_heads"]!=8 or g["prefix"]!=2048 or g["head_dim"]!=128:
            raise ValueError("source grouped GQA geometry mismatch")
        source_gpu=float(g["total_source_grouped_QK_s"])
        if not source_gpu>0:
            raise ValueError("source native GPU QK invalid")
        w=Workload(B,2048,128,32,8,128,32,O)
        c=resident_pipeline(
            w,area_mm2=budget_mm2,support_mm2=support_mm2,
            active_ports=ports,rates=rates,
            operand_stage_support_mm2=operand_stage_mm2,
        )
        if c is None:
            rows.append({"batch":B,"status":"NO_RESIDENT_PIPELINE",
                         "gpu_source_prefix_us_per_layer":source_gpu*1e6,
                         "CIM_mean_us_per_layer":None,
                         "CIM_warm_us_per_layer":None,
                         "CIM_cold_us_per_layer":None,
                         "selected_layers":0,"modeled_hybrid_minus_GPU_us_per_token":0,
                         "wave_group_size":None,"waves":None,
                         "buffer_area_mm2":None,"setup_margin_us_per_token":None,
                         "evidence":"source A100 grouped roofline + explicit 7nm rates"})
            continue
        mean_cost=(c.warm_pipelined_s_per_layer +
                   c.cold_source_plus_program_s_per_layer/O+
                   extra_setup_us*1e-6)
        chosen=c.pinned_layers_max if mean_cost<source_gpu else 0
        rows.append({
            "batch":B,
            "status":"CONDITIONAL_QK_WIN" if chosen else "GPU_SOURCE_PREFIX_QK_FASTER",
            "gpu_source_prefix_us_per_layer":source_gpu*1e6,
            "CIM_mean_us_per_layer":mean_cost*1e6,
            "CIM_warm_us_per_layer":c.warm_pipelined_s_per_layer*1e6,
            "CIM_cold_us_per_layer":c.cold_source_plus_program_s_per_layer*1e6,
            "selected_layers":chosen,
            "modeled_hybrid_minus_GPU_us_per_token":chosen*(mean_cost-source_gpu)*1e6,
            "wave_group_size":c.groups_per_wave,
            "waves":len(c.waves),
            "buffer_area_mm2":c.physical_stage_buffer_area_mm2,
            "setup_margin_us_per_token":(source_gpu -
                c.warm_pipelined_s_per_layer - c.cold_source_plus_program_s_per_layer/O)*1e6,
            "evidence":"source A100 grouped roofline + explicit 7nm rates",
        })
    csv_out.parent.mkdir(parents=True,exist_ok=True)
    with csv_out.open("w",newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    report={
        "schema":"cim-shared-prefix-grouped-GPU-vs-7nm-resident-pipeline-model/v1",
        "P":2048,"O":O,"budget_mm2":budget_mm2,
        "support_mm2":support_mm2,
        "operand_staging_mm2":operand_stage_mm2,
        "port_cap_sensitivity":ports,
        "byte_interfaces_Bps_sensitivity":bw,
        "extra_setup_us_per_layer_per_token":extra_setup_us,
        "rows":rows,
        "comparison_scope":"GPU shared-K grouped prefix QK source roofline versus CIM published-macro signed Q4K4 conditional wave pipeline",
        "numerical_quality":"old Llama R0-R3 split-mapping quality remains preserved; new selective policy has no full-model evaluation",
        "unsupported":["GPU kernel measurements","true optimized Hydragen fused attention","CIM integrated 7nm peripheral service","full decoder TPOT","hardware speedup","real quantization setup latency"],
    }
    json_out.parent.mkdir(parents=True,exist_ok=True)
    json_out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"source_model_batches":len(rows),"model_winning_batches":
                      [r["batch"] for r in rows if r["selected_layers"]],
                      "rows":rows},sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=SRC)
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--json",type=Path,required=True)
    ap.add_argument("--budget-mm2",type=float,default=16)
    ap.add_argument("--support-mm2",type=float,default=.5)
    ap.add_argument("--operand-stage-mm2",type=float,default=.1)
    ap.add_argument("--ports",type=int,default=256)
    ap.add_argument("--bandwidth-Bps",type=float,default=1e12)
    ap.add_argument("--output-tokens",type=int,default=128)
    ap.add_argument("--extra-setup-us",type=float,default=0)
    a=ap.parse_args()
    execute(a.source,a.csv,a.json,budget_mm2=a.budget_mm2,
            support_mm2=a.support_mm2,operand_stage_mm2=a.operand_stage_mm2,
            ports=a.ports,bw=a.bandwidth_Bps,O=a.output_tokens,
            extra_setup_us=a.extra_setup_us)
