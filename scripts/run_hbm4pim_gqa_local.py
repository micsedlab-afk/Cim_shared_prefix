#!/usr/bin/env python3
"""Optional H100/BF16 GQA-native source local-tile adapter.

Requires all pinned HBM4PiM+LLMCompass+SCALE-Sim providers installed in the
private workspace. Returns local compute service only (no HBM4 timing).
It cannot be promoted to full GPU operator/TPOT evidence.
"""
import argparse,json,os,sys
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.gqa_native import gqa_shapes


def run(source_root:Path,w:Workload,step:int,workdir:Path)->dict:
    root=source_root.resolve()
    if not (root/'hbm4pim/science/gpu.py').is_file():
        raise FileNotFoundError('HBM4PiM source checkout unavailable')
    sys.path.insert(0,str(root))
    try:
        from hbm4pim.science.gpu import provider,component
        if not workdir.is_absolute():workdir=workdir.resolve()
        stage=provider(workdir,'H100')
        prev=Path.cwd();os.chdir(stage)
        try:
            stages={};unavailable=[]
            for name,shape in gqa_shapes(w,step).items():
                kind=shape['kind']
                if kind=='softmax':
                    unavailable.append(name);continue
                left=shape['input_a'];right=shape['input_b']
                m=left[-2] if kind=='batched_matmul' else left[0]
                k=left[-1];n=right[-1]
                # Original component is a single GEMM; for grouped score,
                # each (batch,KV-head) group is independent, no K broadcast.
                groups=w.batch*w.kv_heads if kind=='batched_matmul' else 1
                if kind=='matmul':m=w.batch
                if n%32 or k%32:
                    unavailable.append(name+' (physical N/K alignment; no silent padding)')
                    continue
                profile=component('H100','BF16',m,k,n)
                seconds=float(profile['local_compute']['compute_seconds'])*groups
                stages[name]={'seconds_local_only':seconds,'groups':groups,
                               'source':profile['status'],'external_memory_service_included':False}
            return {'schema':'cim-shared-prefix-hbm4pim-gqa-local/v1',
                    'status':'PARTIAL_BF16_LOCAL_GPU_ONLY','source':str(root),
                    'geometry':asdict(w),'stages':stages,'unsupported':unavailable,
                    'full_tpot_s':None,'physical_score_endpoint_qualified':False}
        finally:os.chdir(prev)
    finally:sys.path.remove(str(root))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hbm4pim-root',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--prefix',type=int,default=1024)
    p.add_argument('--suffix',type=int,default=128);p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();w=Workload(a.batch,a.prefix,a.suffix,32,8,128,32,1)
    r=run(a.hbm4pim_root,w,0,a.work_dir);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
