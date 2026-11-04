#!/usr/bin/env python3
"""Strict exact-geometry S0/S1/S2 manifest runner.

Missing native operator evidence gives null TPOT/speedup and enumerated missing
stages. Synthetic or lower-bound source class is never promoted to measurement.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.source_grounded import PhysicalCIM,RateSensitivity,GQAGPUPoint,compare


def read_manifest(path:Path)->dict:
    raw=json.loads(path.read_text())
    if raw.get('schema')!='cim-shared-prefix-matched-system-request/v1':
        raise ValueError('unexpected manifest schema')
    w=Workload(**raw['workload']);phy=PhysicalCIM(**raw['cim_physical'])
    r=RateSensitivity(**raw['periphery_sensitivity']) if 'periphery_sensitivity' in raw else None
    gpu_points=None
    if 'gpu_profile_files' in raw:
        profiles=[]
        for source in raw['gpu_profile_files']:
            file=(path.parent/source).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError('GPU profile path escapes manifest directory')
            data=json.loads(file.read_text())
            if data.get('schema')=='cim-shared-prefix-llmcompass-gqa-stage-point/v1':
                d=data['gpu_point']
            elif data.get('schema')=='cim-shared-prefix-gqa-stage-record/v1':
                d=data['gpu_point']
            else:raise ValueError('unrecognized GPU stage record schema')
            profiles.append(GQAGPUPoint(**d))
        gpu_points=tuple(profiles)
    data=compare(w,phy,gpu_points=gpu_points,rates=r,sram_budget_macros=raw.get('sram_macro_budget'))
    data['request_source']=str(path.resolve())
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=read_manifest(args.manifest);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['evidence_class'],'S0':result['S0_GPU']['tpot_s'],
                     'S1':result['S1_SRAM']['tpot_s'],'S2':result['S2_CIM']['tpot_s']}))
