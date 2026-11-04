#!/usr/bin/env python3
"""Execute a single source-macro-calibrated causal decoder scenario on CPU.

This is real Python scheduling/simulation execution; no GPU/CIM physical
hardware run is implied. GPU full decoder times are emitted only if an exact
source operator profile is supplied; missing stages remain null.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.profiles import PROFILES, visible_prefix
from cim_shared_prefix.integrated_schedule import SerialRates, best_serial
from cim_shared_prefix.residency_area import ExternalArea, enumerate_plans
from cim_shared_prefix.causal_timeline import (
    GPUStagePoint, GPUSourceSeries, SRAMRates, simulate,
)

MODEL_NAMES={"llama":"meta-llama/Llama-3.1-8B",
             "mistral":"mistralai/Mistral-7B-v0.1",
             "opt":"facebook/opt-2.7b"}


def source_series(path: Path) -> GPUSourceSeries:
    blob=json.loads(path.read_text())
    if blob.get("schema")!="cim-shared-prefix-native-gpu-series/v1":
        raise ValueError("no source-grounded GPU operator series contract")
    values={k:v for k,v in blob.items() if k not in ("schema", "steps")}
    return GPUSourceSeries(**values,
                           steps=tuple(GPUStagePoint(**v) for v in blob["steps"]))


def run(args):
    p=next(p for p in PROFILES if p.model_id==MODEL_NAMES[args.model])
    if args.prefix+args.suffix+args.output_tokens>p.context_limit:
        raise ValueError("source context limit exceeded")
    visible=[visible_prefix(p,args.prefix,args.suffix,i)
             for i in range(args.output_tokens)]
    if min(visible)<=0 or len(set(visible))>1:
        raise ValueError("dynamic SWA visibility requires the separate token-varying residency model")
    w=Workload(args.batch,visible[0],args.suffix,p.query_heads,p.kv_heads,
               p.head_dim,p.layers,args.output_tokens)
    area=ExternalArea(*(args.support_mm2/6 for _ in range(6)),
                      source="six-part assumed peripheral area; not physical layout")
    r=SerialRates(args.prefix_supply_Bps,args.program_Bps,args.query_Bps,
                  args.score_output_Bps,args.reduction_adds_per_s,
                  args.partial_scalings_per_s,
                  source="scenario-defined unverified peripheral rates")
    selected=best_serial(w,budget_mm2=args.area_mm2,support=area,
                         rates=r,active_macro_ports=args.ports)
    if selected is None:
        raise ValueError("no physical signed macro pair fits area")
    cim,_=selected
    gpu=source_series(args.gpu_operator_json) if args.gpu_operator_json else None
    sram=None
    sr=None
    if args.sram:
        # The conventional SRAM path uses the same incremental area budget,
        # with an independent source bandwidth. For this fixed-prefix case,
        # minimizing recurring program bytes maximizes useful pinned residency.
        plans=enumerate_plans(w,kind="sram",budget_mm2=args.area_mm2,
                              support=area,cold_start=True)
        if plans:
            sram=min(plans,key=lambda q:(q.programming_bytes_total,
                                         q.instantiated_macros))
            sr=SRAMRates(args.prefix_supply_Bps,args.sram_program_Bps,
                         args.sram_read_Bps,args.sram_link_Bps,
                         "scenario-defined unverified SRAM read/write/host rates")
    result=simulate(w,cim,r,active_macro_ports=args.ports,
                    model_id=p.model_id,model_revision=p.revision,
                    gpu=gpu,sram_plan=sram,sram_rates=sr)
    payload={
        "schema":"cim-shared-prefix-executed-causal-timeline/v1",
        "model_id":p.model_id,"model_revision":p.revision,
        "attention":"MHA" if p.gqa_ratio==1 else "GQA",
        "workload":asdict(w),
        "physical_macro":{"area_per_macro_mm2":0.0032,
                          "native_access_seconds":5.5e-9,
                          "signed_key_planes":2,
                          "unsigned_evaluations_per_signed_operation":4},
        "source_types":{"CIM":"published_macro_plus_explicit_rate_sensitivity",
                        "GPU":gpu.evidence if gpu else None,
                        "GPU_profile_source":gpu.source if gpu else None,
                        "SRAM":"published_7nm_macro_plus_rate_sensitivity" if sram else None},
        "sensitivity_inputs":{
            "total_area_mm2":args.area_mm2,
            "peripheral_area_mm2":args.support_mm2,
            "active_macro_port_cap":args.ports,
            "prefix_supply_Bps":args.prefix_supply_Bps,
            "CIM_program_Bps":args.program_Bps,
            "query_Bps":args.query_Bps,
            "score_output_Bps":args.score_output_Bps,
            "reduction_adds_per_s":args.reduction_adds_per_s,
            "partial_scalings_per_s":args.partial_scalings_per_s,
        },
        "execution":result.json_record(),
        "publication_scope":"source macro timing and constructed serial model, not silicon/real GPU measurement; full decoder only with exact source operator series"
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "output":str(args.out),
        "actual_simulated_steps":len(result.token_steps),
        "instantiated_CIM_macros":result.cim_macro_count,
        "CIM_area_mm2":result.cim_macro_area_mm2,
        "CIM_prefix_mean_us":result.cim_mean_prefix_score_s*1e6,
        "CIM_prefix_first_us":result.cim_first_prefix_score_s*1e6,
        "CIM_prefix_steady_us":(None if result.cim_steady_mean_prefix_score_s is None
                                else result.cim_steady_mean_prefix_score_s*1e6),
        "full_decoder_S0_s":result.s0_model_mean_tpot_s,
        "full_decoder_S1_s":result.s1_model_mean_tpot_s,
        "full_decoder_S2_s":result.s2_model_mean_tpot_s,
        "status":result.status
    },sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=MODEL_NAMES,default="llama")
    ap.add_argument("--prefix",type=int,default=2048)
    ap.add_argument("--suffix",type=int,default=128)
    ap.add_argument("--batch",type=int,default=1)
    ap.add_argument("--output-tokens",type=int,default=128)
    ap.add_argument("--area-mm2",type=float,default=16.0)
    ap.add_argument("--support-mm2",type=float,default=0.5)
    ap.add_argument("--ports",type=int,default=256)
    # All rates are deliberately named and explicitly scenario assumptions.
    ap.add_argument("--prefix-supply-Bps",type=float,default=1e12)
    ap.add_argument("--program-Bps",type=float,default=1e12)
    ap.add_argument("--query-Bps",type=float,default=1e12)
    ap.add_argument("--score-output-Bps",type=float,default=1e12)
    ap.add_argument("--reduction-adds-per-s",type=float,default=1e12)
    ap.add_argument("--partial-scalings-per-s",type=float,default=1e12)
    ap.add_argument("--sram",action="store_true")
    ap.add_argument("--sram-program-Bps",type=float,default=1e12)
    ap.add_argument("--sram-read-Bps",type=float,default=1e12)
    ap.add_argument("--sram-link-Bps",type=float,default=1e12)
    ap.add_argument("--gpu-operator-json",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    run(ap.parse_args())
