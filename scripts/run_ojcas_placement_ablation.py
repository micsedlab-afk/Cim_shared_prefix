#!/usr/bin/env python3
"""Run complete original 728-coordinate reviewer placement-extension ablation."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import argparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_placement_extension import (
    run_grid,candidates,optimize,first_pin_threshold,dimensionless_cost,
)

def run(out_dir:Path):
    rows,manifest=run_grid(decode_batch_steps=128,lambda_values=(1.,10.,40.))
    def point(area,b):
        old,opt,gain=optimize(b,2048,area,decode_batch_steps=128,
                              lambda_write_per_tile=10.)
        return {"area":area,"batch":b,"E":old.total_engines,
                "old_layer_waves":old.compute_tile_waves_per_batch_step,
                "selected_k":opt.pinned_complete_layers,
                "selected_layer_waves":opt.compute_tile_waves_per_batch_step,
                "old_total_tile_writes":old.K4_tile_writes_total,
                "new_total_tile_writes":opt.K4_tile_writes_total,
                "cost_gain_at_unqualified_lambda10":gain,
                "first_pin_threshold":first_pin_threshold(
                    b,2048,area,decode_batch_steps=128)}
    manifest["reference_points"]=[
        point(8,1),point(8,4),point(8,8),
        point(12,1),point(16,1),point(16,4),
    ]
    manifest["validation"]={
        "submitted_engine_macros_unchanged":128,
        "original_stage_speedup_from_manuscript_not_modified":True,
        "real_hardware_measurement_count":0,
        "new_end_to_end_decoder_speedup":None,
        "selected_policy_only_with_dimensionless_cost_sensitivity":True,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/"original_728_placement_ablation.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out_dir/"original_728_placement_ablation.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("OJCAS_PLACEMENT_ABLATION "+json.dumps({
        "total_coordinates":len(rows),"per_area":manifest["areas"],
        "reference_points":manifest["reference_points"],
        "evidence_class":manifest["evidence_class"],
    },sort_keys=True),flush=True)
    return rows,manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    run(p.parse_args().out_dir)
