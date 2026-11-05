#!/usr/bin/env python3
"""Source-executed A100 FP16: exact vs compressed shared-prefix GPU envelopes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.a100_compressed_comparator import point

def run(path:Path):
    rows = [point(batch=b).json_record() for b in (1,2,4,8,16,32)]
    payload={
        "schema":"a100_fp16_compressed_shared_prefix_source_comparator/v1",
        "source":"MAGMA-gpt/LLMCompass A100_80GB_fp16 HBM2e; byte-corrected matmul; complete published 7nm macro geometry",
        "workload":"Llama-3.1-8B native 32Q/8KV/D128/32 layers, P2048 M64 O128",
        "numerical_status":"GPU FP16 compressed centroid is distinct from ideal signed Q4K4 and O4 ADC; quality matching and GPU INT4 timings unavailable",
        "memory_status":"L2 capacity is nominal only; actual residency not proven; HBM and L2 treated separately",
        "performance_status":"fused results are lower FLOP/byte floors; unfused results are hypothetical source-operator sums not measured GPU",
        "area_status":"macro-only; no synthesized digital, controller or physical integration",
        "all_layers_quality":"UNMEASURED_LLAMA_8B",
        "a100_kernel_measurements":0,
        "end_to_end_tpot_us":None,
        "hardware_speedup":None,
        "rows":rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"EXECUTED_A100_SOURCE_MODELS_NO_GPU_MEASUREMENTS","B1":rows[0],
        "case_count":len(rows)},sort_keys=True))
    return payload

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    run(a.parse_args().out)
