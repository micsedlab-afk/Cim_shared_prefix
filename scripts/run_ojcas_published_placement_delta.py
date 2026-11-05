#!/usr/bin/env python3
"""Evaluate original OJCAS S2 paper-source placement Δ; no original speedup edits."""
from __future__ import annotations
import argparse,csv,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_published_placement_delta import (
    execute_728,optimize_published_source,first_pin_critical_hbm_Bps,
)

def run(out_dir:Path):
    rows,meta=execute_728(output_batch_steps=128)
    anchors=[]
    for area,p,b in [(8,2048,1),(12,2048,1),(16,2048,1),
                     (16,2048,4),(8,512,1),(16,32768,128)]:
        control,best,_=optimize_published_source(b,p,area,output_batch_steps=128)
        anchors.append({
            "Aadd_mm2":area,"P":p,"B":b,
            "original_engines":control.active_engines,
            "selected_pinned_layers":best.pin_layers,
            "original_3component_us_per_batch_step":control.three_component_serial_seconds*1e6/128,
            "optimized_3component_us_per_batch_step":best.three_component_serial_seconds*1e6/128,
            "incremental_3component_delta_us_per_batch_step":(
                best.three_component_serial_seconds-control.three_component_serial_seconds)*1e6/128,
            "original_K4_supply_us_per_step":control.k4_supply_seconds*1e6/128,
            "selected_K4_supply_us_per_step":best.k4_supply_seconds*1e6/128,
            "original_program_us_per_step":control.program_seconds*1e6/128,
            "selected_program_us_per_step":best.program_seconds*1e6/128,
            "original_macro_compute_us_per_step":control.macro_compute_seconds*1e6/128,
            "selected_macro_compute_us_per_step":best.macro_compute_seconds*1e6/128,
            "first_pin_critical_HBM_GBps":(
                crit/1e9 if (crit := first_pin_critical_hbm_Bps(
                    b,p,area,output_batch_steps=128)) is not None and math.isfinite(crit)
                else None),
            "first_pin_unconditional_at_finite_HBM":math.isinf(
                first_pin_critical_hbm_Bps(b,p,area,output_batch_steps=128)
                or 0.0),
        })
    meta["anchored_reference_points"]=anchors
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/"original_728_published_S2_placement_stage_delta.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out_dir/"original_728_published_S2_placement_stage_delta.json").write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("OJCAS_PUBLISHED_PLACEMENT_DELTA "+json.dumps({
        "count":len(rows),"by_area":meta["by_area"],
        "anchors":anchors,"source":meta["experimental_class"]
    },sort_keys=True),flush=True)
    return rows,meta

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    run(parser.parse_args().out_dir)
