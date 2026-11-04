#!/usr/bin/env python3
"""Execute source-native 128-step A100 GQA roofline against selective pinned 7nm Q4K4 CIM.

Sweep physically charged complete-layer residency, finite two-wave buffers,
causal compute -> scale -> signed reduce -> completed-score output, cold key
provisioning, and no-regression GPU fallback. All links, ports and off-macro
rates are **explicit** 7nm hypothetical sensitivities, not measured silicon.
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
from cim_shared_prefix.selective_pipeline import resident_pipeline,optimize_selective

GPU_INPUT=Path("results/integration/source_native_a100_fp16_gqa_qk_128steps.json")
SCHEMA="cim-shared-prefix-selective-pinned-pipeline-simulated-study/v1"


def comma_numbers(raw,cast):
    result=tuple(cast(x) for x in raw.split(","))
    if not result:
        raise ValueError("empty axis")
    return result


def validate_gpu(data):
    if data.get("schema")!="cim-shared-prefix-source-native-a100-gqa-roofline/v1":
        raise ValueError("GPU model record must come from source-native LLMCompass")
    p=data["provenance"]
    if p["measured_hardware"] is not False or "A100" not in p["hardware_model"]:
        raise ValueError("model type/provenance is not the pinned A100 roofline")
    w=data["workload"]
    required={"model_id":"meta-llama/Llama-3.1-8B","precision":"FP16",
              "batch":1,"prefix":2048,"suffix":128,"output_tokens":128,
              "query_heads":32,"kv_heads":8,"head_dim":128,"layers":32}
    if any(w.get(k)!=v for k,v in required.items()):
        raise ValueError("native GQA/precision/model geometry mismatch")
    steps=data["steps"]
    if len(steps)!=128 or [s["token_index"] for s in steps]!=list(range(128)):
        raise ValueError("one full source-model step per output required")
    return tuple(s["full_score_s"] for s in steps),tuple(s["suffix_score_s"] for s in steps)


def study(args):
    gpu=json.loads(args.gpu_input.read_text())
    full,suffix=validate_gpu(gpu)
    rows=[]
    examples={}
    for budget in args.budgets:
        for ports in args.ports:
            for bw in args.bandwidths:
                rates=SerialRates(bw,bw,bw,bw,1e12,1e12,
                                  source="explicit non-hardware macro services for source-native policy comparison")
                for o in args.outputs:
                    w=Workload(1,2048,128,32,8,128,32,o)
                    pipe=resident_pipeline(
                        w,area_mm2=budget,support_mm2=args.support_mm2,
                        active_ports=ports,rates=rates,
                        operand_stage_support_mm2=args.operand_staging_mm2)
                    for setup_us in args.setups_us:
                        base={
                            "model":"meta-llama/Llama-3.1-8B",
                            "gpu_source_model":"native_LLMCompass_A100_FP16_roofline",
                            "physical_macro":"Sinangil_7nm_Q4K4_posneg_O4",
                            "P":2048,"suffix":128,"B":1,"O":o,
                            "budget_mm2":budget,
                            "external_area_sensitivity_mm2":args.support_mm2,
                            "operand_staging_reserved_mm2":args.operand_staging_mm2,
                            "port_cap_sensitivity":ports,
                            "all_interfaces_Bps_sensitivity":bw,
                            "extra_gpu_to_cim_setup_us_per_layer":setup_us,
                            "GPU_measured_time_s":None,
                            "full_decoder_TPOT_s":None,
                            "hardware_speedup":None,
                        }
                        if pipe is None:
                            base.update({
                                "status":"NO_COMPLETE_SIGNED_K_LAYER_FITS_WITH_STAGING",
                                "pinned_layer_capacity":0,
                                "selected_CIM_layers":0,
                                "gpu_source_qk_mean_us_per_token":None,
                                "hybrid_source_qk_mean_us_per_token":None,
                                "hybrid_minus_gpu_us_per_token":None,
                                "conditional_qk_speedup":None,
                                "warm_pipelined_us_per_layer":None,
                                "cold_provision_us_per_layer":None,
                                "pipeline_vs_serial_factor":None,
                                "max_extra_setup_before_no_win_us":None,
                                "min_outputs_to_amortize_cold":None,
                                "CIM_footprint_mm2":None,
                                "buffer_SRAM_macros":None,
                                "buffer_SRAM_area_mm2":None,
                            })
                            rows.append(base)
                            continue
                        result=optimize_selective(
                            w,pipe,
                            full_qk_s_per_layer=full[:o],
                            suffix_qk_s_per_layer=suffix[:o],
                            area_budget_mm2=budget,support_mm2=args.support_mm2,
                            source="MAGMA-gpt native A100 FP16 roofline 128 step source record",
                            extra_setup_s_per_layer_per_token=setup_us*1e-6)
                        base.update({
                            "status":("CONDITIONAL_MODEL_WIN" if result.policy_selected_layers>0
                                      else "GPU_ONLY_OPTIMAL_IN_POLICY_FAMILY"),
                            "pinned_layer_capacity":result.pinned_physical_layer_count,
                            "selected_CIM_layers":result.policy_selected_layers,
                            "gpu_source_qk_mean_us_per_token":result.baseline_mean_gpu_qk_service_s*1e6,
                            "hybrid_source_qk_mean_us_per_token":result.candidate_mean_hybrid_qk_service_s*1e6,
                            "hybrid_minus_gpu_us_per_token":result.candidate_minus_gpu_s_per_token*1e6,
                            "conditional_qk_speedup":result.conditional_qk_speedup,
                            "warm_pipelined_us_per_layer":pipe.warm_pipelined_s_per_layer*1e6,
                            "cold_provision_us_per_layer":pipe.cold_source_plus_program_s_per_layer*1e6,
                            "pipeline_vs_serial_factor":pipe.warm_nonoverlapped_s_per_layer/pipe.warm_pipelined_s_per_layer,
                            "max_extra_setup_before_no_win_us":(
                                result.maximum_extra_setup_s_to_retain_advantage*1e6
                                if result.maximum_extra_setup_s_to_retain_advantage is not None else None),
                            "min_outputs_to_amortize_cold":result.minimum_outputs_to_amortize_cold,
                            "CIM_footprint_mm2":pipe.macro_area_one_layer_mm2*result.policy_selected_layers,
                            "buffer_SRAM_macros":pipe.physical_storage_macros,
                            "buffer_SRAM_area_mm2":pipe.physical_stage_buffer_area_mm2,
                        })
                        rows.append(base)
                        if (budget==16 and ports==256 and bw==1e12
                                and o==128 and setup_us==0):
                            examples["reference"]=result.json_record()

    args.csv.parent.mkdir(parents=True,exist_ok=True)
    with args.csv.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if "reference" not in examples:
        raise ValueError("reference case not included in study axes")
    reference=examples["reference"]
    results={
        "schema":SCHEMA,"source_reference":str(args.gpu_input),
        "gpu_model":"A100 FP16 LLMCompass roofline; not measured GPU or Hydragen",
        "cim_model":"published 7nm Sinangil signed positive-negative split; 4 unsigned accesses; 2 K banks",
        "source_support":"published 7nm Yokoyama SRAM density; buffer size charged in macro granularity",
        "assumptions":{
            "area_support_mm2":args.support_mm2,
            "operand_staging_within_support_mm2":args.operand_staging_mm2,
            "rates_digital_adds_and_scales_per_s":1e12,
            "GPU_prefix_score_assembly_and_Q4_quantization_overhead":"explicit setup sensitivity; zero for reference",
            "pipeline_stage_resources":"independent unsigned array, scaling, reduction and output",
            "pipeline_buffers":"two complete intermediates each: raw O4 codes and 32b score words",
            "GPU_attention_non_QK_stages":"cancel by assumption, not profiled",
            "cold_pinned_programming":"charged once per selected layer and amortized over O tokens",
            "non_pinned_layers":"remain on source-modeled GPU; never repeatedly streamed",
        },
        "total_cases":len(rows),
        "conditional_model_wins":sum(r["status"]=="CONDITIONAL_MODEL_WIN" for r in rows),
        "GPU_fallback_cases":sum(r["status"]=="GPU_ONLY_OPTIMAL_IN_POLICY_FAMILY" for r in rows),
        "no_pinned_fit_cases":sum(r["status"].startswith("NO_COMPLETE") for r in rows),
        "reference":reference,
        "qualifications":[
            "One held source-native GPU FP16 roofline model is used; no BF16 or optimized Hydragen assertion.",
            "R0-R3 Llama numerical results use the preserved positive-negative mapping; per-layer selective execution is not directly evaluated for end-to-end PPL.",
            "Physical interface bandwidth, 256 ports and separate digital units are unqualified assumptions.",
            "The source-model QK-only latency difference is not full attention or full decoder TPOT.",
            "The max extra setup threshold defines how quantization, DMA launch and extra merging can erase the conditional modeled gain.",
            "The mapping and scalar quality gate are preserved exactly; no affine-offset remapping or ADC recalibration.",
        ],
    }
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    r=reference
    print(json.dumps({
        "coordinates":len(rows),"conditional_model_wins":results["conditional_model_wins"],
        "fallbacks":results["GPU_fallback_cases"],
        "reference_selected_layers":r["policy_selected_layers"],
        "reference_mean_difference_us":r["candidate_minus_gpu_s_per_token"]*1e6,
        "reference_warm_cim_us_per_layer":r["pipeline"]["warm_pipelined_s_per_layer"]*1e6,
        "reference_first_cim_us_per_layer":r["pipeline"]["warm_pipelined_s_per_layer"]*1e6+r["pipeline"]["cold_source_plus_program_s_per_layer"]*1e6,
        "reference_max_extra_setup_us":r["maximum_extra_setup_s_to_retain_advantage"]*1e6,
        "csv":str(args.csv),"summary":str(args.summary),
    },sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gpu-input",type=Path,default=GPU_INPUT)
    p.add_argument("--budgets",type=lambda s:comma_numbers(s,int),default=(13,14,15,16))
    p.add_argument("--ports",type=lambda s:comma_numbers(s,int),default=(64,128,256,512))
    p.add_argument("--bandwidths",type=lambda s:comma_numbers(s,float),default=(1e11,5e11,1e12))
    p.add_argument("--outputs",type=lambda s:comma_numbers(s,int),default=(1,8,16,32,128))
    p.add_argument("--setups-us",type=lambda s:comma_numbers(s,float),default=(0.0,0.1,0.3))
    p.add_argument("--support-mm2",type=float,default=0.5)
    p.add_argument("--operand-staging-mm2",type=float,default=0.1)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    study(p.parse_args())
