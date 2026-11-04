#!/usr/bin/env python3
"""Execute source 7-nm KV residency and FP16 GPU memory limits, without rates.

No CIM bandwidth, MAC/s, area allowance, or latency is invented. The
only source GPU performance rate used is NVIDIA's published A100 SXM
2039 GB/s, to derive a *conditional all-HBM bytes/peak-bandwidth floor*.
The core 16mm² capacity and traffic results are independent of that rate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.verified_residency_limits import (
    capacity_limit, dense_llama_weight_parameters,
    fp16_decoder_weight_bytes, A100_80GB_SXM_HBM_BPS,
    A100_L2_NOMINAL_BYTES, A100_PUBLIC_DOC, A100_SOURCE,
    LLAMA_CONFIG_SOURCE,
)

PREF=(16,32,64,80,128,256,304,320,384,512,768,1024,1536,2048,2304,3072,4096)
BUDGETS=(1,2,4,8,16,32,64,128)


def execute(csv_path:Path,summary_path:Path,
            prefixes=PREF,area_budgets=BUDGETS):
    rows=[capacity_limit(p,area).json_record() for area in area_budgets for p in prefixes]
    if not rows:
        raise ValueError("empty source-derived capacity study")
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main=capacity_limit(2048,16)
    multi=capacity_limit(4096,16)
    short=capacity_limit(64,16)
    head=capacity_limit(320,16)
    data={
        "schema":"source-grounded-7nm-cim-fp16v-kv-capacity-no-go/v1",
        "result_type":"exact integer macro capacity and ideal GPU byte/traffic envelope; not CUDA or CIM latency",
        "sources":{
            "llama_native_model_config":LLAMA_CONFIG_SOURCE,
            "published_7nm_CIM_macro":"https://doi.org/10.1109/JSSC.2020.3031290",
            "published_7nm_SRAM_macro":"https://doi.org/10.1109/VLSICircuits18222.2020.9162985",
            "A100_source_peak":"https://www.nvidia.com/en-us/data-center/a100/",
            "A100_40MB_cache":A100_PUBLIC_DOC
        },
        "source_gpu_FP16_streaming_model":{
            "hbm_peak_Bps":A100_80GB_SXM_HBM_BPS,
            "l2_nominal_capacity_bytes":A100_L2_NOMINAL_BYTES,
            "model_dimensions":"Llama-3.1-8B Q32 KV8 D128 layers32 hidden4096 intermediate14336 vocab128256",
            "weight_matmul_parameters_by_component":dense_llama_weight_parameters(),
            "weight_bytes_FP16_per_token_if_all_projections_and_untied_LM_head_stream_once":fp16_decoder_weight_bytes(),
            "layer_attention_prefix_KV_bytes_FP16_P2048":main.gpu_fp16_prefix_KV_bytes_per_layer,
            "model_attention_prefix_KV_bytes_FP16_P2048":main.gpu_fp16_prefix_KV_bytes_all_layers,
            "ideal_HBM_time_min_per_full_model_prefix_KV_only_us_P2048":main.gpu_fp16_prefix_KV_bytes_all_layers/A100_80GB_SXM_HBM_BPS*1e6,
            "ideal_HBM_time_min_per_full_model_weights_plus_prefix_KV_us_P2048":main.ideal_weight_plus_KV_HBM_bytes_per_token/A100_80GB_SXM_HBM_BPS*1e6,
            "read_assumption":"every dense decoder/lm-head FP16 weight once per B1 token; every FP16 shared prefix KV once per token, from HBM",
            "not_a_full_decoder_latency_measurement":True
        },
        "main_16mm2_P2048":main.json_record(),
        "full_KV_P4096_16mm2":multi.json_record(),
        "entire_model_KV_cache_B1_P64":short.json_record(),
        "A100_L2_pressure_boundary_P320":head.json_record(),
        "area_budgets_mm2":list(area_budgets),
        "prefixes":list(prefixes),
        "case_count":len(rows),
        "bound_scope":[
            "fixed 7nm complete signed K4 CIM and complete 7nm FP16 V SRAM macros only",
            "16mm2 is the explicitly studied accelerator storage budget, not measured die size",
            "digital and controller areas are zero ONLY to form a permissive storage upper bound",
            "persistent tile residency means no per-token reprogramming; streamed reprogrammed configurations are outside the occupancy exclusion",
            "packing tiles full-layer-first minimizes positive per-touched-layer setup cost for equal tiles; does not prove a latency optimum under unknown heterogeneous per-layer workloads",
            "A100 L2 40MB is a nominal capacity, not proof of cache hits; complete KV footprint exceeding it only rules out simultaneous full residency",
            "the all-HBM model is a conditional ideal byte-streaming model, not a measured GPU performance result",
            "the uniform-layer attention speedup ceiling applies to equal per-layer baseline prefix attention times and zero offload cost, not every possible GPU scheduling implementation",
            "weight traffic comparison assumes B1 FP16 dense model weights and no effective weight reuse across output tokens; does not establish measured speedup or energy",
            "no KV reduction of advanced compressed/pruned GPU baseline modeled; user can compare only matching precisions"
        ]
    }
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("CAPACITY_SOURCE_STUDY "+json.dumps({
        "rows":len(rows),"main_P":2048,
        "min_KV_area_mm2":main.complete_kv_macro_area_mm2_per_layer,
        "fully_pinned_layers":main.max_complete_layers_without_peripherals,
        "max_partially_persistent_tiles":main.max_persistent_16token_layer_tiles_without_peripherals,
        "max_persistent_prefix_fraction":main.persistent_prefix_token_layer_fraction_ceiling,
        "max_GPU_streaming_HBM_byte_fraction_saved":main.optimistic_fraction_of_full_decoder_HBM_bytes_removed,
        "P4096_full_KV_layers":multi.max_complete_layers_without_peripherals,
        "P64_full_KV_layers":short.max_complete_layers_without_peripherals,
        "csv":str(csv_path),"summary":str(summary_path)
    },sort_keys=True))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--summary",type=Path,required=True)
    args=ap.parse_args()
    execute(args.csv,args.summary)
