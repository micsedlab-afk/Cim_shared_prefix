#!/usr/bin/env python3
"""Execute SHA-verified source-native ideal 64x16 macro on signed Q4/K4 checks.

The upstream source file must be checked out under _upstream/shared_prefix_cim.
This script does not vendor or silently substitute the source implementation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import random
import subprocess
from pathlib import Path

EXPECTED_BLOB='6e968ef3b4cd2b09529930ad24cce66425ff93e0'
EXPECTED_MAPPING_BLOB='78c30ae3996f1cd830d2a6d08ebb1685cedbe645'

def run(root:Path,seed:int=20260917) -> dict:
    f=root/'_upstream/shared_prefix_cim/experiments/sram2llm_shared_prefix/sinangil_ideal_macro.py'
    if not f.is_file(): raise FileNotFoundError(f)
    actual=subprocess.check_output(['git','hash-object',str(f)],text=True).strip()
    if actual!=EXPECTED_BLOB:raise RuntimeError(f'upstream source blob mismatch {actual}')
    spec=importlib.util.spec_from_file_location('source_native_sinangil_ideal_macro',f)
    if spec is None or spec.loader is None: raise RuntimeError('could not load upstream module')
    import sys
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    rng=random.Random(seed)
    checked=0
    native_calls=0
    max_abs_error=0
    for _ in range(12):
        q=[rng.randint(-7,7) for _ in range(128)]
        keys=[[rng.randint(-7,7) for _ in range(128)] for _ in range(16)]
        scores=[0]*16
        for tile in range(2):
            for qsign in (1,-1):
                qq=[max(0,qsign*a) for a in q[tile*64:(tile+1)*64]]
                for ksign in (1,-1):
                    kk=[[max(0,ksign*keys[j][tile*64+i]) for j in range(16)] for i in range(64)]
                    result=mod.run_ideal_macro(qq,kk)
                    native_calls+=1
                    scores=[s+qsign*ksign*r for s,r in zip(scores,result.pre_adc_aggregate)]
                    for bit in range(4):
                        for j in range(16):
                            if result.bit_plane_counts[bit][j] < 0:
                                raise RuntimeError('negative native bit-plane count')
        for j in range(16):
            expect=sum(a*b for a,b in zip(q,keys[j]))
            max_abs_error=max(max_abs_error,abs(scores[j]-expect))
            checked+=1
            assert scores[j]==expect
    mapping_path=f.with_name('sinangil_mappings.py')
    if not mapping_path.is_file():raise FileNotFoundError('source-native signed mapping file absent')
    mapping_blob=subprocess.check_output(['git','hash-object',str(mapping_path)],text=True).strip()
    if mapping_blob!=EXPECTED_MAPPING_BLOB:raise RuntimeError('source-native signed mapping hash mismatch')
    upstream=f.parents[2]
    sys.path.insert(0,str(upstream))
    try:
        from experiments.sram2llm_shared_prefix.sinangil_mappings import SignedQuantizedVector,execute_mapping
        mapping_macro_calls=0
        mapping_scores=0
        for _ in range(8):
            query=[rng.randint(-7,7) for _ in range(128)]
            keys=[[rng.randint(-7,7) for _ in range(128)] for _ in range(16)]
            qvec=SignedQuantizedVector(tuple(query),(1.,1.),4)
            keyvec=[SignedQuantizedVector(tuple(k),(1.,1.),4) for k in keys]
            mapped=execute_mapping(qvec,keyvec,'positive_negative_split')
            assert mapped.active_macro_operations==8 and mapped.key_storage_multiplier==2
            mapping_macro_calls+=mapped.active_macro_operations
            for j,key in enumerate(keys):
                assert mapped.score[j]==sum(x*y for x,y in zip(query,key))
                mapping_scores+=1
    finally:
        sys.path.remove(str(upstream))
    return {'schema':'cim-shared-prefix-upstream-execution/v1' ,'backend':'original unmodified sinangil_ideal_macro.run_ideal_macro and sinangil_mappings.execute_mapping',
            'source_repo':'micsedlab-afk/shared_prefix_cim','source_commit':'a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4',
            'source_file_blob_git_sha1':actual,'signed_mapping_blob_git_sha1':mapping_blob,'source_verified':True,
            'samples':checked,'native_macro_calls':native_calls,'max_integer_dot_error':max_abs_error,
            'mapping_samples':mapping_scores,'source_mapping_macro_calls':mapping_macro_calls,
            'checked_work':'Q4K4 positive/negative sign components, two 64-row tiles, sixteen score outputs',
            'hardware_timing_executed':False,'physical_macros_simulated':False,
            'classification':'upstream_ideal_digital_arithmetic_reference_only'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();d=run(args.root);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(json.dumps(d,indent=2))
