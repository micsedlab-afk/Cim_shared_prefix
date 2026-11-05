#!/usr/bin/env python3
"""Export original S1 excluded cold-placement and S2-stage decoder scope overlay."""
from __future__ import annotations
import json,sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.ojcas_review_scope import reference_sensitivity

def run(out_dir:Path):
    doc=reference_sensitivity()
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"original_S1_placement_and_S2_scope.json").write_text(
        json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print("OJCAS_ORIGINAL_REVIEW_SCOPE "+json.dumps({
       "S1":doc["S1_placement_sensitivities"],
       "Amdahl":doc["decoder_scope_sensitivities"],
       "evidence_class":doc["evidence_class"],
    },sort_keys=True),flush=True)
    return doc

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    run(parser.parse_args().out_dir)
