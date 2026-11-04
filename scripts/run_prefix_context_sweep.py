#!/usr/bin/env python3
"""Execute conditional 7nm K-CIM/FP16V-SRAM prefix-context architecture sweep.

Comparison endpoints:
 (a) source A100 grouped QK-only roofline, a demanding *proxy* threshold;
 (b) source A100 serial, unfused grouped QK+Softmax+PV roofline, which is
     a complete prefix-context output but NOT a competitive fused GPU kernel.
No hardware measurement/TPOT/speedup is inferred. All prefix engine link,
local read and digital compute rates are visibly unqualified sensitivities.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.prefix_context_engine import ContextRates,simulate_prefix_context

GROUPED=Path("results/integration/native_grouped_a100_fp16_prefix_roofline.json")
FULL=Path("results/integration/native_full_prefix_a100_fp16_roofline.json")

def numbers(s,typ):
    return tuple(typ(v) for v in s.split(","))


def run(args):
    gpu=json.loads(args.gpu.read_text())
    if gpu.get("schema")!="cim-shared-prefix-native-grouped-A100-FP16-roofline/v1":
        raise ValueError("unrecognized source GPU grouped QK model")
    grouped={d["batch"]:d["total_source_grouped_QK_s"] for d in gpu["results"]}
    if args.full.exists():
        full=json.loads(args.full.read_text())
        if full.get("schema")!="native-llmcompass-unfused-shared-prefix-attention-FP16/v1":
            raise ValueError("unrecognized full-prefix source model")
        full_by_batch={x["batch"]:x for x in full["results"]}
        if set(grouped)!=set(full_by_batch):
            raise ValueError("native GPU comparator batch geometry differs")
    else:
        full_by_batch={}
    rows=[]
    examples={}
    for B in args.batches:
        if B not in grouped:
            raise ValueError("missing source GPU QK batch")
        for area in args.areas:
            for port in args.ports:
                for vbw in args.v_bw_TBps:
                    for mac in args.mac_Tper_s:
                        for O in args.outputs:
                            w=Workload(B,2048,128,32,8,128,32,O)
                            rates=ContextRates(
                                key_supply_Bps=args.external_TBps*1e12,
                                key_program_Bps=args.external_TBps*1e12,
                                v_supply_Bps=args.external_TBps*1e12,
                                v_program_Bps=args.external_TBps*1e12,
                                query_Bps=args.external_TBps*1e12,
                                local_v_read_Bps=vbw*1e12,
                                exp_per_s=args.exp_Tper_s*1e12,
                                scalar_ops_per_s=args.scalar_Tper_s*1e12,
                                pv_mac_per_s=mac*1e12,
                                context_output_Bps=args.external_TBps*1e12,
                                partial_scalings_per_s=args.reduction_Tper_s*1e12,
                                signed_adds_per_s=args.reduction_Tper_s*1e12,
                                source="unqualified 7nm near-memory digital and on-chip SRAM rate sensitivities",
                            )
                            p=simulate_prefix_context(
                                w,area_mm2=area,
                                support_mm2=args.support_mm2,ports=port,
                                rates=rates,gpu_prefix_qk_s=grouped[B],
                                external_setup_s_per_token=args.setup_us*1e-6)
                            gpu_full=full_by_batch.get(B,{}).get("full_prefix_unfused_operator_sum_s")
                            base=dict(B=B,P=2048,O=O,area_mm2=area,
                                      support_mm2=args.support_mm2,port_cap=port,
                                      V_local_TBps=vbw,PV_digital_TMAC_per_s=mac,
                                      exp_digital_Texp_per_s=args.exp_Tper_s,
                                      external_link_TBps=args.external_TBps,
                                      external_setup_us_per_token=args.setup_us,
                                      grouped_GPU_QK_source_us=grouped[B]*1e6,
                                      unfused_GPU_full_prefix_source_us=(
                                          gpu_full*1e6 if gpu_full else None),
                                      hardware_speedup=None,
                                      full_decoder_TPOT=None)
                            if p is None:
                                base.update(status="NO_COMPLETE_K_V_RESIDENCY",resident_layers=0,
                                            context_us=None,model_vs_GPU_QK_us=None,
                                            model_vs_unfused_GPU_us=None,physical_area_used_mm2=None,
                                            K_CIM_macros=None,V_SRAM_macros=None,buffer_SRAM_macros=None,
                                            wave_groups=None,query_microbatch=None,model_extra_setup_margin_us=None,
                                            required_PV_MAC_density_TMAC_per_s_mm2=None)
                            else:
                                delta_qk=(p.mean_context_s-grouped[B])*1e6
                                delta_full=((p.mean_context_s-gpu_full)*1e6
                                            if gpu_full is not None else None)
                                base.update(status="MODEL_BELOW_GPU_QK_PROXY" if delta_qk<0 else
                                            ("MODEL_BELOW_UNFUSED_GPU" if delta_full is not None and delta_full<0
                                             else "GPU_SOURCE_UNFUSED_LOWER_OR_UNKNOWN"),
                                            resident_layers=p.max_pinned_layers,
                                            context_us=p.mean_context_s*1e6,
                                            warm_context_us=p.warm_attention_context_s*1e6,
                                            cold_KV_us_once=p.cold_KV_supply_and_program_s*1e6,
                                            model_vs_GPU_QK_us=delta_qk,
                                            model_vs_unfused_GPU_us=delta_full,
                                            physical_area_used_mm2=p.total_physical_area_mm2,
                                            K_CIM_macros=p.cim_k_macros,
                                            V_SRAM_macros=p.sram_v_macros,
                                            buffer_SRAM_macros=p.sram_pipeline_macros,
                                            V_read_bytes_per_token=p.per_token_v_read_bytes,
                                            wave_groups=p.wave_groups,query_microbatch=p.query_microbatch,
                                            model_extra_setup_margin_us=(grouped[B] -
                                                p.mean_context_s+args.setup_us*1e-6)*1e6,
                                            required_PV_MAC_density_TMAC_per_s_mm2=(
                                                p.digital_MAC_rate_density_required_per_mm2/1e12))
                                if (B==1 and area==16 and port==256 and O==128
                                    and vbw==16 and mac==64):
                                    examples["reference_B1"]=p.json_record()
                            rows.append(base)
    if not rows or "reference_B1" not in examples:
        raise AssertionError("reference coordinate not executed")
    args.csv.parent.mkdir(parents=True,exist_ok=True)
    allkeys=list(dict.fromkeys(key for r in rows for key in r))
    with args.csv.open("w",newline="") as fd:
        writer=csv.DictWriter(fd,allkeys)
        writer.writeheader();writer.writerows(rows)
    status={}
    for row in rows:status[row["status"]]=status.get(row["status"],0)+1
    result={
        "schema":"conditional-7nm-K-CIM-FP16V-prefix-context-sweep/v1",
        "source_GPU_QK":"native LLMCompass A100 FP16 grouped K-shared source roofline",
        "source_GPU_full":"native LLMCompass A100 FP16 unfused QK+softmax+PV roofline, if present",
        "architecture":"one complete pinned 7nm Q4K4O4 K-CIM layer + FP16 V SRAM + digital stable online softmax and PV",
        "conditional_inputs":{
            "external_support_mm2":args.support_mm2,
            "external_bandwidth_TBps":args.external_TBps,
            "scalar_Tops_per_s":args.scalar_Tper_s,
            "exp_Tper_s":args.exp_Tper_s,
            "signed_reduction_Tper_s":args.reduction_Tper_s,
            "setup_us_per_token":args.setup_us,
            "physical_PV_MAC_area_and_power":"unknown; input MAC rate is a sensitivity, not a synthesized 7nm block",
        },
        "cases":len(rows),"classifications":status,
        "reference_B1":examples["reference_B1"],
        "model_comparison_scope":[
            "QK-only source GPU roofline is a proxy threshold, not an observed lower bound on optimized fused GPU attention",
            "unfused QK+softmax+PV roofline is a complete prefix endpoint but an unoptimized serial GPU model",
            "near-memory PV is digital with explicit throughput, area and power unqualified",
            "FP16 V is preserved, signed Q4K4O4 scores approximate BF16 GPU score values",
            "prefix context plus LSE passed to GPU for its suffix fusion; common suffix not profiled",
            "numerical oracle tests streaming softmax and fusion for supplied scores, but no real 7nm ADC output quality evaluation",
            "full decoder TPOT and actual speedup remain null",
        ],
    }
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    reference=next(x for x in rows if (x["B"],x["area_mm2"],x["port_cap"],x["O"],x["V_local_TBps"],x["PV_digital_TMAC_per_s"])==(1,16,256,128,16,64))
    print("PREFIX_CONTEXT_STUDY "+json.dumps({
        "rows":len(rows),"classifications":status,
        "reference_B1":reference,
        "outputs":[str(args.csv),str(args.json)]},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=Path,default=GROUPED)
    p.add_argument("--full",type=Path,default=FULL)
    p.add_argument("--batches",type=lambda s:numbers(s,int),default=(1,2,4,8))
    p.add_argument("--areas",type=lambda s:numbers(s,int),default=(15,16))
    p.add_argument("--ports",type=lambda s:numbers(s,int),default=(128,256))
    p.add_argument("--v-bw-TBps",type=lambda s:numbers(s,float),default=(1,2,4,8,16,32))
    p.add_argument("--mac-Tper-s",type=lambda s:numbers(s,float),default=(4,16,64))
    p.add_argument("--outputs",type=lambda s:numbers(s,int),default=(32,128))
    p.add_argument("--support-mm2",type=float,default=0.5)
    p.add_argument("--external-TBps",type=float,default=1)
    p.add_argument("--exp-Tper-s",type=float,default=1)
    p.add_argument("--scalar-Tper-s",type=float,default=1)
    p.add_argument("--reduction-Tper-s",type=float,default=1)
    p.add_argument("--setup-us",type=float,default=0)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--json",type=Path,required=True)
    run(p.parse_args())
