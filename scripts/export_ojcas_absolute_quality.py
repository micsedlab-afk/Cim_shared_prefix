#!/usr/bin/env python3
"""Export absolute manuscript R0-R3 Llama perplexity without reinterpreting data."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

REFERENCE = Path("results/quality/absolute_quality.csv")
PREFIXES = (512,2048,16384)
ARMS = ("R0","R1","R2","R3")
MODEL = "meta-llama/Llama-3.1-8B"

def export(source:Path, out_dir:Path):
    with source.open(newline="") as f:
        allrows=list(csv.DictReader(f))
    found={}
    for row in allrows:
        if (row.get("model_id")==MODEL and row.get("batch")=="1" and
            int(row["prefix_tokens"]) in PREFIXES and
            row.get("suffix_tokens")=="128" and row.get("arm") in ARMS):
            p=int(row["prefix_tokens"]); arm=row["arm"]
            if (p,arm) in found: raise ValueError("duplicate frozen quality evidence")
            nll=float(row["mean_nll"]);ppl=float(row["ppl"])
            if not math.isclose(math.exp(nll),ppl,rel_tol=1e-9):
                raise ValueError("absolute PPL is not exp pooled mean NLL")
            if int(row["sample_count"])!=8 or int(row["scored_tokens"])!=1024:
                raise ValueError("archived Llama sample count differs")
            found[p,arm]=row
    if set(found)!={(p,a) for p in PREFIXES for a in ARMS}:
        raise ValueError("incomplete frozen Llama R0-R3 evidence")
    table=[]
    delta=[]
    for p in PREFIXES:
        vals={a:float(found[p,a]["ppl"]) for a in ARMS}
        for a in ARMS:
            row=found[p,a]
            table.append({
                "prefix_tokens":p,"arm":a,
                "sample_count":8,"scored_tokens":1024,
                "mean_nll":row["mean_nll"],
                "absolute_ppl":row["ppl"],
                "ppl_ci95_low":row["ppl_ci95_low"],
                "ppl_ci95_high":row["ppl_ci95_high"],
                "source_url":row["source_url"],
            })
        delta.append({
            "prefix_tokens":p,
            "R0_BF16_ppl":vals["R0"],"R1_Q4K4_ppl":vals["R1"],
            "R2_ADC_ppl":vals["R2"],"R3_circuit_ppl":vals["R3"],
            "R1_vs_R0_pct":100*(vals["R1"]/vals["R0"]-1),
            "R2_vs_R1_pct":100*(vals["R2"]/vals["R1"]-1),
            "R3_vs_R2_pct":100*(vals["R3"]/vals["R2"]-1),
            "R3_vs_R0_pct":100*(vals["R3"]/vals["R0"]-1),
            "R3_vs_R1_pct":100*(vals["R3"]/vals["R1"]-1),
        })
    out_dir.mkdir(parents=True,exist_ok=True)
    for filename,rows in [
        ("ojcas_absolute_ppl_R0_R3.csv",table),
        ("ojcas_ppl_stage_effects.csv",delta),
    ]:
        with (out_dir/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s={
        "schema":"ojcas_2026_0120_absolute_quality_from_archived_llama8b/v1",
        "model":MODEL,"scored_targets_per_prefix_arm":1024,
        "samples_per_prefix_arm":8,
        "source_csv":str(source),
        "data_status":"ORIGINAL_REAL_PRETRAINED_LLAMA_R0_R3_ARCHIVED_SCALARS_NOT_NEW_INFERENCE",
        "rows":delta,
        "nonmonotonic_R3_lower_than_R2_at_2048":delta[1]["R3_vs_R2_pct"]<0,
        "caveats":[
            "PPL computed from same restricted held-out samples, not full benchmark corpus",
            "R2→R3 non-monotonic observation is not a statistically established benefit",
            "No confidence interval on ratio is created by combining marginal PPL CIs",
            "These quality data are original R0-R3, not new compressed-prefix inference",
        ],
    }
    (out_dir/"ojcas_absolute_ppl_summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    print("OJCAS_ABSOLUTE_PPL_SUMMARY "+json.dumps(s,sort_keys=True),flush=True)
    return s

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=REFERENCE)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    export(a.source,a.out_dir)
