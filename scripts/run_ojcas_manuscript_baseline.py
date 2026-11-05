#!/usr/bin/env python3
"""Export unchanged submitted OJCAS baseline and optional reviewer overhead gate."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_manuscript_baseline import manuscript_grid,saved_stage_speedup_margin
def run(out_dir:Path):
    rows,s=manuscript_grid()
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/"submitted_OJCAS_728_model_grid.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    s["paper_best_point_extra_stage_overhead_margin_fraction_of_Tconv"]=saved_stage_speedup_margin(
        s["paper_max_modeled_QK_stage_speedup"])
    s["overhead_gate_caveat"]="Applies only to original single 1.97x best stage-speedup point; no absolute timing or broad claim"
    (out_dir/"submitted_OJCAS_baseline_lock.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    print("ORIGINAL_OJCAS_BASELINE_LOCK "+json.dumps({
        "coordinates":len(rows),"S2_macros_per_engine":s["original_s2_macro_per_engine"],
        "S2_engine_mm2":s["original_s2_engine_area_mm2"],
        "win_counts":s["paper_favorable_counts_by_area"],
        "max_model_QK_stage_speedup":s["paper_max_modeled_QK_stage_speedup"],
        "original_eq10_HD_crossover_B":s["paper_macro_partial_energy"]["crossing_B_s1_hd"],
        "original_eq10_DP_crossover_B":s["paper_macro_partial_energy"]["crossing_B_s1_dp"],
        "best_point_extra_overhead_margin_fraction":s["paper_best_point_extra_stage_overhead_margin_fraction_of_Tconv"],
        "individual_original_timing_records_recovered":0,
        "prior_256_macro_sensitivity_applied_to_baseline":False,
    },sort_keys=True),flush=True)
    return s
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    run(p.parse_args().out_dir)
