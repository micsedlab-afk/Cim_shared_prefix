#!/usr/bin/env python3
"""Requalify physical exclusion ONLY against actual matched GPU CUDA events.

Requires *all* per-step full Flash/Hydragen and GPU score-to-attention
remainder measurements (same device/precision and complete attention output).
No interpolation, synthetic operation counts, borrowed workload, 2ms deadline
or unmeasured GPU speedup. The CIM rate/area parameters remain explicit
unqualified sensitivity assumptions.

If no valid GPU measurement file is provided, this script refuses to produce
a competitive region. The prior fixed deadline archive remains historical.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.profiles import PROFILES, visible_prefix
from cim_shared_prefix.residency_area import ExternalArea, residency_plan
from cim_shared_prefix.integrated_schedule import SerialRates, best_serial
from cim_shared_prefix.feasibility import PhysicalBudget, lower_bound

GPU_SCHEMA = "cim-shared-prefix-real-CUDA-attention/v1"
RESULT_SCHEMA = "cim-shared-prefix-measured-attention-physical-envelope/v1"
ACCEPTED = ("full_flash_sdpa", "grouped_gemm", "compiled_grouped_gemm")


def _positive(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0


def validate_gpu(path: Path, *, mode: str):
    r=json.loads(path.read_text())
    if r.get("schema") != GPU_SCHEMA or r.get("status") != "GPU_KERNEL_MEASUREMENTS_RECORDED":
        raise ValueError("no qualified measured CUDA attention evidence")
    if not r.get("hardware",{}).get("cuda_available"):
        raise ValueError("source is not a CUDA hardware execution")
    if not r.get("hardware",{}).get("gpu_name") or not r.get("hardware",{}).get("cuda_runtime"):
        raise ValueError("missing GPU identity/runtime")
    profile=next((x for x in PROFILES if x.model_id==r["model_id"]),None)
    if profile is None or profile.revision!=r.get("model_revision"):
        raise ValueError("model/revision does not match native source profile")
    q=r.get("requested",{})
    b,p,s,o=(q.get(x) for x in ("batch","prefix","suffix","output_tokens"))
    if not all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in (b,p,o)) or not isinstance(s,int) or s<0:
        raise ValueError("invalid requested native workload")
    if p+s+o > profile.context_limit:
        raise ValueError("source model context exceeded")
    if q.get("steps")!=list(range(o)) or len(r.get("measurements",[]))!=o:
        raise ValueError("requires one GPU measurement at EVERY output step; no interpolation")
    if r.get("precision") not in ("fp16","bf16"):
        raise ValueError("unsupported kernel precision")
    series=[]
    for i,point in enumerate(r["measurements"]):
        if point.get("token_step")!=i or point.get("status")!="MEASURED_ATTENTION_KERNEL_ONLY":
            raise ValueError("missing measured CUDA step")
        if (point.get("batch")!=b or point.get("model_id")!=profile.model_id or
            point.get("model_revision")!=profile.revision or
            point.get("precision")!=r["precision"] or
            point.get("active_prefix_tokens")!=visible_prefix(profile,p,s,i) or
            point.get("completed_score_word_bits")!=32):
            raise ValueError("nonmatching per-token geometry, precision or endpoint")
        measures=point.get("measurements",[])
        valid={}
        for m in measures:
            if (m.get("execution_mode")!=mode or
                m.get("status")!="GPU_KERNEL_EVENT_MEASURED" or
                m.get("name") not in ACCEPTED+("gpu_score_to_attention_remainder",)):
                continue
            if (m.get("endpoint")!="full_prefix_suffix_attention_output"
                or not _positive(m.get("median_us")) or
                m.get("kernel_proof",{}).get("cuda_kernel_count",0)<1):
                raise ValueError("incomplete GPU output endpoint or missing CUDA kernels")
            if m["name"]=="full_flash_sdpa" and not m["kernel_proof"].get("flash_kernel_observed"):
                raise ValueError("FlashAttention backend was not observed")
            if m["name"] in valid:
                raise ValueError("duplicate measured implementation")
            valid[m["name"]]=float(m["median_us"])/1e6
        if "full_flash_sdpa" not in valid or "gpu_score_to_attention_remainder" not in valid:
            raise ValueError("missing competitive full GPU or score-remainder kernel measurement")
        preferred=min((k for k in ACCEPTED if k in valid), key=lambda k: valid[k])
        series.append({"step":i,"visible_prefix":point["active_prefix_tokens"],
                       "gpu_attention_best_s":valid[preferred],
                       "gpu_attention_implementation":preferred,
                       "gpu_remainder_s":valid["gpu_score_to_attention_remainder"],
                       "flash_s":valid["full_flash_sdpa"]})
    return r,profile,series


def compare(path: Path, *, mode: str, budgets: tuple[float,...],
            support_mm2: float, programming_Bps: float, output_Bps: float,
            key_supply_Bps: float, query_Bps: float, reduction_adds_s: float,
            partial_scales_s: float, ports: int):
    info,profile,gpu=validate_gpu(path,mode=mode)
    b=info["requested"]["batch"]
    p=info["requested"]["prefix"]
    s=info["requested"]["suffix"]
    o=info["requested"]["output_tokens"]
    macro_area=ExternalArea(*(support_mm2/6 for _ in range(6)),
        source="unqualified six-block external area sensitivity")
    rates=SerialRates(key_supply_Bps,programming_Bps,query_Bps,output_Bps,
                      reduction_adds_s,partial_scales_s,
                      source="user-given unqualified physical interface rates")
    measured_reference=statistics.mean(x["gpu_attention_best_s"] for x in gpu)
    measured_remainder=statistics.mean(x["gpu_remainder_s"] for x in gpu)
    if not _positive(measured_reference) or not _positive(measured_remainder):
        raise ValueError("unqualified or zero measured GPU reference")
    rows=[]
    for budget in budgets:
        traces=[];floors=[];bytes_out=[];bytes_program=[]
        if len(set(x["visible_prefix"] for x in gpu))==1:
            w=Workload(b,gpu[0]["visible_prefix"],s,profile.query_heads,
                       profile.kv_heads,profile.head_dim,profile.layers,o)
            opt=best_serial(w,budget_mm2=budget,support=macro_area,
                            rates=rates,active_macro_ports=ports)
            if opt is None:
                rows.append({"budget_mm2":budget,"status":"NO_SIGNED_SCRATCH_ALLOCATION",
                             "GPU_attention_s":measured_reference})
                continue
            plan,trace=opt
            serial=trace.serial_completed_score_s/profile.layers
            floor=trace.optimistic_resource_floor_s/profile.layers
            count=plan.instantiated_macros
            avg_program=plan.programming_bytes_per_generated_token/profile.layers
            avg_out=trace.completed_score_bytes_per_token/profile.layers
            pins=plan.pinned_layers
            dynamic=False
        else:
            # Mistral SWA: constructive cold-per-position schedule is conservative.
            # Necessary floor cannot assume reprogramming on every changed window.
            dynamic=True
            ticks=[]
            for item in gpu:
                w=Workload(b,item["visible_prefix"],s,profile.query_heads,
                           profile.kv_heads,profile.head_dim,profile.layers,1)
                opt=best_serial(w,budget_mm2=budget,support=macro_area,
                                rates=rates,active_macro_ports=ports)
                if opt is None:
                    raise ValueError("visible prefix unschedulable at area")
                plan,trace=opt
                bounds=lower_bound(w,PhysicalBudget(budget,support_mm2),
                                   rates.ceilings(),active_macro_port_cap=ports)
                strict=max(bounds.macro_compute_floor_s,bounds.query_floor_s,
                           bounds.score_link_floor_s)/profile.layers
                ticks.append((trace.serial_completed_score_s/profile.layers,strict,
                              trace.completed_score_bytes_per_token/profile.layers,
                              plan.programming_bytes_per_generated_token/profile.layers,
                              plan.instantiated_macros,plan.pinned_layers))
            serial,floor,avg_out,avg_program,count,pins=(
                statistics.mean(x[j] for x in ticks) for j in range(6))
        # Same attention endpoint. All CIM/transport timing is conditional
        # while GPU service is measured. GPU remainder may overlap with score
        # streaming, so exclusion uses QK-only floor by itself, not their sum.
        conditional_total=serial+measured_remainder
        exclusion=floor>measured_reference
        achievable=conditional_total<=measured_reference
        if exclusion and achievable:
            raise AssertionError("schedule beats an independently necessary floor")
        status=("CONDITIONALLY_EXCLUDED_BY_QK_FLOOR" if exclusion else
                "CONDITIONALLY_ATTAINABLE_FULL_ATTENTION" if achievable else
                "UNRESOLVED_WITH_STATED_INTERFACE_SENSITIVITY")
        rows.append({
            "model_id":profile.model_id, "revision":profile.revision,
            "GPU":info["hardware"]["gpu_name"],"precision":info["precision"],
            "batch":b,"prefix":p,"suffix":s,"output_steps":o,
            "area_budget_mm2":budget,"external_support_mm2_sensitivity":support_mm2,
            "physical_CIM_macros":count,"pinned_layers":pins,
            "port_cap_sensitivity":ports,
            "program_Bps_sensitivity":programming_Bps,
            "score_Bps_sensitivity":output_Bps,
            "source_Bps_sensitivity":key_supply_Bps,
            "SW_window_varies":dynamic,
            "GPU_best_full_attention_mean_s":measured_reference,
            "GPU_score_to_attention_remainder_mean_s":measured_remainder,
            "CIM_QK_serial_mean_per_layer_s_sensitivity":serial,
            "CIM_QK_necessary_floor_mean_per_layer_s":floor,
            "CIM_plus_GPU_remainder_serial_mean_s_sensitivity":conditional_total,
            "necessary_output_Bps_for_any_beat":avg_out/measured_reference,
            "candidate_program_Bps_for_entire_traffic":avg_program/measured_reference,
            "minimum_full_CIM_macro_area_mm2":2*profile.kv_heads*
                math.ceil(profile.head_dim/64)*math.ceil(max(x["visible_prefix"] for x in gpu)/16)*
                profile.layers*0.0032,
            "status":status,
            "measured_GPU_attention_kernel":True,
            "physically_qualified_CIM_interface":False,
            "full_decoder_TPOT_s":None,
            "full_system_CIM_speedup":None,
            "digital_isoarea_claim":False,
        })
    return {"schema":RESULT_SCHEMA, "GPU_input":str(path),
            "hardware":info["hardware"],"model_id":profile.model_id,
            "source_revision":profile.revision,
            "GPU_baseline": "min(real Flash SDPA, real grouped-GEMM implementations) at identical full attention endpoint",
            "qualifications": {
                "GPU_kernel_service":"measured_cuda_event_on_shape_matched_activation_tensors",
                "GQA_SLIDING":profile.sliding_window,
                "CIM":"published_macro_plus_unqualified_interface_sensitivities",
                "SWA_floor":"programming discarded when visibility changes",
                "GPU_remainder":"real CUDA kernel service with FP32 prefix completed scores already GPU-resident",
                "decoder":"unqualified, do not extend kernel scores to full TPOT",
            },
            "rows":rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--mode",choices=("graph","direct"),default="graph")
    ap.add_argument("--budgets-mm2",type=float,nargs="+",default=[1,4,8,16])
    ap.add_argument("--support-mm2",type=float,default=.5)
    ap.add_argument("--programming-Bps",type=float,required=True)
    ap.add_argument("--score-output-Bps",type=float,required=True)
    ap.add_argument("--key-supply-Bps",type=float,required=True)
    ap.add_argument("--query-Bps",type=float,required=True)
    ap.add_argument("--reduction-adds-per-s",type=float,required=True)
    ap.add_argument("--partial-scalings-per-s",type=float,required=True)
    ap.add_argument("--port-cap",type=int,required=True)
    a=ap.parse_args()
    if (a.support_mm2<0 or not math.isfinite(a.support_mm2) or
        a.port_cap<1 or any(not _positive(v) for v in (
            a.programming_Bps,a.score_output_Bps,a.key_supply_Bps,a.query_Bps,
            a.reduction_adds_per_s,a.partial_scalings_per_s))):
        ap.error("all hypothetical physical rates and active ports must be declared positive")
    result=compare(a.gpu_json,mode=a.mode,budgets=tuple(a.budgets_mm2),
        support_mm2=a.support_mm2,programming_Bps=a.programming_Bps,
        output_Bps=a.score_output_Bps,key_supply_Bps=a.key_supply_Bps,
        query_Bps=a.query_Bps,reduction_adds_s=a.reduction_adds_per_s,
        partial_scales_s=a.partial_scalings_per_s,ports=a.port_cap)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False,sort_keys=True)+"\n")
    csv_file=a.out.with_suffix(".csv")
    with csv_file.open("w",newline="") as f:
        fields=list(dict.fromkeys(key for row in result["rows"] for key in row))
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"status_counts":{s:sum(r["status"]==s for r in result["rows"])
                                       for s in sorted({r["status"] for r in result["rows"]})},
                      "json":str(a.out),"csv":str(csv_file)}))


if __name__=="__main__":
    main()
