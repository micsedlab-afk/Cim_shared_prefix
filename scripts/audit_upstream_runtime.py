#!/usr/bin/env python3
"""Evidence-only audit of the original four upstream checkouts; never auto-substitute."""
import argparse,json,subprocess
from pathlib import Path

PIN={'shared_prefix_cim':'a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4',
     'SRAM2LLM':None,'MAGMA-gpt':None,'HBM4PiM':None}

def audit(root:Path)->dict:
    rec={}
    for name,rev in PIN.items():
        repo=root/'_upstream'/name
        if not (repo/'.git').exists():
            rec[name]={'status':'SOURCE_CHECKOUT_NOT_AVAILABLE','source':'scripts/acquire_upstreams.sh'}
            if name=='shared_prefix_cim':
                ideal=repo/'experiments/sram2llm_shared_prefix/sinangil_ideal_macro.py'
                if ideal.is_file():
                    h=subprocess.run(['git','hash-object',str(ideal)],capture_output=True,text=True,check=True).stdout.strip()
                    mapping=repo/'experiments/sram2llm_shared_prefix/sinangil_mappings.py'
                    mh=subprocess.run(['git','hash-object',str(mapping)],capture_output=True,text=True,check=True).stdout.strip() if mapping.is_file() else None
                    rec[name]={'status':'PARTIAL_SHA_VERIFIED_SOURCE_FILES',
                               'original_repo_checkout':False,'ideal_macro_blob_sha':h,
                               'mapping_blob_sha':mh,
                               'exact_upstream_blob_match':h=='6e968ef3b4cd2b09529930ad24cce66425ff93e0' and mh=='78c30ae3996f1cd830d2a6d08ebb1685cedbe645'}
            continue
        head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        rec[name]={'status':'CHECKOUT_READY' if rev is None or rev==head else 'REVISION_MISMATCH',
                   'head_sha':head,'expected_sha':rev}
    return {'schema':'cim-shared-prefix-upstream-availability/v1','repos':rec,
            'full_native_MAGMA_HBM4PiM_available':rec['MAGMA-gpt']['status']=='CHECKOUT_READY' and rec['HBM4PiM']['status']=='CHECKOUT_READY',
            'acquisition_note':'Git connector can read repository code but local git clone requires network authentication; no secret files recorded'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2))
