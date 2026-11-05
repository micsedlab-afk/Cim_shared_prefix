#!/usr/bin/env python3
"""Requalify the original paper's 728-point grid without inventing INT4 speedups."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_paper_728_audit import full_grid

def execute(dest:Path):
    dest.mkdir(parents=True,exist_ok=True)
    rows,s=full_grid()
    with (dest/"ojcas_728_signed_macro_audit.csv").open("w",newline="") as fd:
        w=csv.DictWriter(fd,fieldnames=list(rows[0]),extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    (dest/"ojcas_728_signed_macro_audit.json").write_text(json.dumps(
        s,indent=2,sort_keys=True)+"\n")
    anchors=[x for x in rows if x["Lp"]==2048 and x["B"]==1
             and x["Aadd_mm2"] in (1,8,16)]
    print("OJCAS_728_AUDIT_SUMMARY "+json.dumps({
        "execution":"ACTUAL_PYTHON_SOURCE_PHYSICAL_ACCOUNTING",
        "coordinates":len(rows),
        "S1_original_capacity_at_8mm2":next(
            v for v in s["physical_capacity_by_paper_area"]
            if v["area_mm2"]==8),
        "energy":s["partial_macro_energy_recalculation"],
        "P2048_B1_anchors":anchors,
        "qualifications":{
            "original_728_speedups_reproduced":False,
            "A100_INT4_native_operator_evidence":False,
            "GPU_TPOT_evidence":False,
            "fully_qualified_isoarea":False
        }},sort_keys=True),flush=True)
    return rows,s

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.out_dir)
