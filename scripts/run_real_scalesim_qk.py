"""Execute real SCALE-Sim v3 GEMM if installed; no synthetic trace fallback.

Usage: PYTHONPATH=src python scripts/run_real_scalesim_qk.py --config ACTUAL.cfg
       --out /path/to/results [--batch 1 --prefix 2048 --dim 128 --heads 32 --kv 8 --layers 32]
SCALE-Sim executes the DIGITAL GEMM and provides mapping/cycles only; it cannot
provide physical MAC area, native signed INT4 verification, or macro port proof.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from cim_shared_prefix.contracts import Workload


def topology(w: Workload) -> str:
    if w.key_bits != 4:
        raise ValueError('Q4/K4 analysis only')
    return ('Layer, M, N, K\n'
            f'QK_per_KV_head, {w.batch * w.gqa_ratio}, {w.prefix_tokens}, {w.head_dim}\n')


def run(w: Workload, *, config: Path, output: Path) -> dict:
    if importlib.util.find_spec('scalesim') is None:
        raise RuntimeError('SCALE-Sim is not installed; no measured cycles or area may be reported')
    if not config.is_file():
        raise FileNotFoundError(f'actual SCALE-Sim architecture config required: {config}')
    output.mkdir(parents=True, exist_ok=True)
    topo=output/'qk_gemm_topology.csv';topo.write_text(topology(w))
    cmd=[sys.executable,'-m','scalesim.scale','-i','gemm','-c',str(config.resolve()),
         '-t',str(topo.resolve()),'-p',str((output/'trace').resolve())]
    result=subprocess.run(cmd,capture_output=True,text=True,check=False)
    receipt={
        'command':cmd, 'config':str(config.resolve()), 'returncode':result.returncode,
        'stdout':result.stdout[-3000:], 'stderr':result.stderr[-3000:],
        'workload':vars(w), 'GEMM_iterations':w.layers*w.kv_heads,
        'timestamp_UTC':datetime.now(timezone.utc).isoformat(),
        'qualification':'actual SCALE-Sim execution only; digital area, signed Q4xK4 precision, physical banking and complete score not established',
    }
    (output/'scalesim_run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'SCALE-Sim failed; see {output}/scalesim_run_receipt.json')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--prefix',type=int,default=2048)
    p.add_argument('--dim',type=int,default=128)
    p.add_argument('--heads',type=int,default=32)
    p.add_argument('--kv',type=int,default=8)
    p.add_argument('--layers',type=int,default=32)
    a=p.parse_args()
    w=Workload(a.batch,a.prefix,0,a.heads,a.kv,a.dim,a.layers,128,key_bits=4)
    print(json.dumps(run(w,config=a.config,output=a.out),indent=2))

if __name__=='__main__':main()