#!/usr/bin/env python3
"""Execute paired source-model A100 GQA QK vs 7nm CIM serial-model differential.

This compares the native LLMCompass A100 FP16 *roofline* GPU QK sources
against the macro-calibrated Q4K4 7nm CIM path. It assumes the same GPU
postscore softmax/PV operator model cancels on S0 and S2; score assembly is
omitted (favoring CIM). It does NOT claim optimized Hydragen, silicon
speedup, BF16 equivalence, or a fully simulated decoder TPOT.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.causal_timeline import simulate
from cim_shared_prefix.integrated_schedule import SerialRates,best_serial
from cim_shared_prefix.residency_area import ExternalArea
from cim_shared_prefix.gpu_saving_boundary import from_integrated_architecture_row

MODEL="meta-llama/Llama-3.1-8B"
SCHEMA="cim-shared-prefix-model-paired-a100-roofline-v1"


def validate_source(path):
    s=json.loads(path.read_text())
    if s.get("schema")!="cim-shared-prefix-source-native-a100-gqa-roofline/v1":
        raise ValueError("unexpected source GPU operator evidence")
    w=s["workload"]
    if any(w.get(k)!=v for k,v in {
        "model_id":MODEL,"precision":"FP16","prefix":2048,"suffix":128,
        "batch":1,"output_tokens":128,"query_heads":32,"kv_heads":8,
        "head_dim":128,"layers":32
    }.items()):
        raise ValueError("GPU native model/operator geometry differs")
    if len(s.get("steps",[]))!=128 or [x["token_index"] for x in s["steps"]]!=list(range(128)):
        raise ValueError("source 128-step profile incomplete")
    if s.get("provenance",{}).get("measured_hardware") is not False:
        raise ValueError("roofline model provenance may not claim hardware")
    for x in s["steps"]:
        full,suffix,delta=(x[k] for k in
            ("full_score_s","suffix_score_s","delta_s"))
        if not all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0
                   for v in (full,suffix)) or not math.isclose(
            full-suffix,delta,rel_tol=1e-11,abs_tol=1e-14
        ):
            raise ValueError("native source GPU timings inconsistent")
    return s


def main(a):
    gpu=validate_source(a.gpu_source)
    w=Workload(1,2048,128,32,8,128,32,128,key_bits=4)
    support=ExternalArea(*(0.5/6 for _ in range(6)),
                         source="unqualified equal-six-block support-area sensitivity")
    rate=SerialRates(1e12,1e12,1e12,1e12,1e12,1e12,
                     source="explicit 1 TB/s, 1 Toper/s conditional peripheral rates")
    selected=best_serial(w,budget_mm2=16,support=support,
                         rates=rate,active_macro_ports=256)
    if selected is None:
        raise ValueError("signed physical CIM layout invalid")
    plan,_=selected
    physical=simulate(w,plan,rate,active_macro_ports=256)

    # Independently check against published 2,304-case coordinate, avoiding a
    # stale matching profile or a changed physical allocation.
    with a.architectural_csv.open(newline="") as f:
        source=list(csv.DictReader(f))
    filtered=[x for x in source if x["model"]==MODEL and
              int(x["prefix_tokens_original"])==2048 and
              int(x["suffix_tokens"])==128 and
              int(x["batch"])==1 and int(x["output_tokens"])==128 and
              float(x["budget_mm2"])==16.0 and
              float(x["support_mm2_sensitivity"])==0.5 and
              float(x["source_bandwidth_Bps_sensitivity"])==1e12]
    if len(filtered)!=1:
        raise ValueError("published macro source coordinate missing")
    threshold=from_integrated_architecture_row(filtered[0])
    if not math.isclose(threshold.modeled_cim_prefix_s,
                        physical.cim_mean_prefix_score_s,
                        rel_tol=1e-10,abs_tol=1e-14):
        raise AssertionError("independent source macro chronology mismatch")

    pairs=[]
    for i,(step,g) in enumerate(zip(physical.token_steps,gpu["steps"])):
        saving=w.layers*g["delta_s"]
        excess=step.cim_prefix_complete_score_s-saving
        pairs.append({
            "step":i,"gpu_source_roofline_prefix_QK_saving_s":saving,
            "cim_7nm_serial_prefix_QK_s":step.cim_prefix_complete_score_s,
            "modeled_S2_minus_S0_attention_s":excess,
            "scope":"source roofline+serial CIM, same non-QK stages cancelled",
        })
    def mean(k):
        return sum(x[k] for x in pairs)/len(pairs)
    diff=mean("modeled_S2_minus_S0_attention_s")
    floor_excess=(threshold.source_macro_floor_s -
                  mean("gpu_source_roofline_prefix_QK_saving_s"))
    out={
        "schema":SCHEMA,
        "status":"EXECUTED_SOURCE_ROOFLINE_VS_7NM_MACRO_SERIAL_MODEL",
        "model":MODEL,
        "GPU_source":gpu["provenance"],
        "CIM_source":"published 7nm Sinangil 0.0032mm2 5.5ns unsigned 4b macro",
        "comparison":"S2-S0 = CIM prefix-QK completed-score service - (GPU full-context QK - GPU suffix QK), with identical postscore softmax/PV and other model operations cancelled",
        "score_assembly_sensitivity":"zero extra GPU score assembly; optimistic for CIM",
        "params":{"P":2048,"suffix":128,"B":1,"O":128,"layers":32,
                  "total_area_budget_mm2":16,"external_support_mm2_assumed":0.5,
                  "active_port_limit_assumed":256,
                  "source_program_query_score_bandwidth_Bps_assumed":1e12,
                  "signed_reduction_and_scaling_rate_per_s_assumed":1e12},
        "source_average_full_GPU_roofline_QK_s_per_layer":gpu["mean_full_score_s"],
        "source_average_suffix_GPU_roofline_QK_s_per_layer":gpu["mean_suffix_score_s"],
        "source_average_model_prefix_GPU_QK_saving_s_per_layer":gpu["mean_modeled_qk_saving_s"],
        "cim_necessary_resource_floor_s_per_generated_token":threshold.source_macro_floor_s,
        "cim_serial_average_s_per_generated_token":physical.cim_mean_prefix_score_s,
        "gpu_model_prefix_QK_saving_across_layers_s":mean("gpu_source_roofline_prefix_QK_saving_s"),
        "S2_minus_S0_attention_model_difference_s_per_generated_token":diff,
        "floor_minus_roofline_saving_s_per_token":floor_excess,
        "source_threshold_us_per_layer":threshold.required_gpu_attention_saving_us_per_layer,
        "source_roofline_saving_us_per_layer":gpu["mean_modeled_qk_saving_s"]*1e6,
        "relative_7nm_numerical_quality":"CIM signed Q4K4 O4 numerical quality differs from FP16 GPU; not precision matched",
        "global_GPU_vs_CIM_speedup":None,
        "qualified_end_to_end_TPOT_s":None,
        "scope_limit":"model comparison only: difference of source roofline estimates is not an upper bound on real optimized GPU savings; no physical integrated peripheral validation",
        "token_points":pairs,
    }
    # Recheck average against first principles, not a model conversion factor.
    if not math.isclose(diff,physical.cim_mean_prefix_score_s-
                        w.layers*gpu["mean_modeled_qk_saving_s"],
                        rel_tol=1e-11,abs_tol=1e-14):
        raise AssertionError("paired difference recomposition failed")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":out["status"],
                      "GPU_prefix_saving_us_per_layer":out["source_roofline_saving_us_per_layer"],
                      "CIM_serial_required_us_per_layer":out["source_threshold_us_per_layer"],
                      "S2_minus_S0_attention_us_per_token":diff*1e6,
                      "model_floor_minus_GPU_roofline_saving_us_per_token":floor_excess*1e6,
                      "steps":len(pairs),
                      "output":str(a.out)},sort_keys=True))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu-source",type=Path,default=Path("results/integration/source_native_a100_fp16_gqa_qk_128steps.json"))
    ap.add_argument("--architectural-csv",type=Path,default=Path("results/integration/architecture_1_16mm2.csv"))
    ap.add_argument("--out",type=Path,required=True)
    main(ap.parse_args())
