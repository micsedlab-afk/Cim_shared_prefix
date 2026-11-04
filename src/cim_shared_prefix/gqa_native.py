"""Native LLMCompass *operator* runner for exact Llama-3.1-8B GQA shapes.

The vendored TransformerBlockAutoRegressionTP is architecturally MHA/GeLU
and therefore must NOT be used. This adapter calls the original Matmul,
BatchedMatmul and Softmax operators directly at GQA-compatible dimensions,
leaving RMSNorm, SwiGLU, RoPE and other absent source primitives as None.
It cannot supply INT4, BF16 or complete TPOT by re-labeling FP16 results.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import importlib
import json
import math
import os
import sys

from .contracts import Workload
from .source_grounded import Llama31, REQUIRED_NON_SCORE_STAGES, GQAGPUPoint

MAGMA_NATIVE_SOURCE = ('micsedlab-afk/MAGMA-gpt/src/magma/llmcompass_native_a100.py; '
                       'vendored third_party/LLMCompass/software_model/{matmul,softmax,transformer}.py')


def gqa_shapes(w: Workload, token_index: int) -> dict[str,dict[str,object]]:
    """Exact arithmetic dimensions in source Llama geometry, no GQA K broadcast copies."""
    Llama31().validate(w)
    if not 0 <= token_index < w.output_tokens:
        raise ValueError('token_index out of range')
    b=w.batch; d=4096; kv=8; heads=32; g=heads//kv; h=128
    c=w.prefix_tokens+w.suffix_tokens+token_index+1
    u=w.suffix_tokens+token_index+1
    stages={
        'q_proj':('matmul',(b,1,d),(d,d)),
        'k_proj':('matmul',(b,1,d),(d,kv*h)),
        'v_proj':('matmul',(b,1,d),(d,kv*h)),
        'out_proj':('matmul',(b,1,d),(d,d)),
        'gate_proj':('matmul',(b,1,d),(d,14336)),
        'up_proj':('matmul',(b,1,d),(d,14336)),
        'down_proj':('matmul',(b,1,14336),(14336,d)),
        'full_score':('batched_matmul',(b,kv,g,h),(b,kv,h,c)),
        'prefix_score':('batched_matmul',(b,kv,g,h),(b,kv,h,w.prefix_tokens)),
        'suffix_score':('batched_matmul',(b,kv,g,h),(b,kv,h,u)),
        'attention_apply':('batched_matmul',(b,kv,g,c),(b,kv,c,h)),
        'softmax':('softmax',(b,heads,1,c),None),
        'lm_head':('matmul',(b,1,d),(d,128256)),
    }
    return {name:{'kind':kind,'input_a':a,'input_b':right,'context':c,'output_elements':(
        math.prod(a[:-1])*right[-1] if kind=='matmul'
        else math.prod(a[:-2])*a[-2]*right[-1] if kind=='batched_matmul'
        else math.prod(a))}
        for name,(kind,a,right) in stages.items()}


def _verify_vendored(root: Path):
    vendor=root/'third_party'/'LLMCompass'
    scalesim=root/'third_party'/'SCALE-Sim'
    if not vendor.is_dir() or not scalesim.is_dir():
        raise FileNotFoundError('MAGMA-gpt vendored LLMCompass and SCALE-Sim required; run scripts/acquire_upstreams.sh')
    paths=(vendor,scalesim)
    sys.path[:0]=[str(p) for p in paths]
    try:
        native=importlib.import_module('software_model.matmul')
        sm=importlib.import_module('software_model.softmax')
        util=importlib.import_module('software_model.utils')
        devices=importlib.import_module('hardware_model.device')
        for mod in (native,sm,util,devices):
            if not Path(mod.__file__).resolve().is_relative_to(vendor.resolve()):
                raise RuntimeError('LLMCompass import escaped pinned vendor source')
        if native._magma_latency_provider is not None:
            raise RuntimeError('MAGMA stage-timing override installed; native operator must run source model')
        return native,sm,util,devices,paths
    except BaseException:
        for p in paths:
            sys.path.remove(str(p))
        raise


def run_native_fp16(w:Workload, token_index:int, magma_root:Path) -> dict[str,object]:
    """Execute genuine native A100 FP16 operators at exact GQA geometry.

    The source A100 device is FP16/HBM2e, so the return value is a distinct
    FP16 architectural candidate, not BF16/INT4 or HBM4 calibration. The
    original Llama model is BF16; the primary BF16 system result stays missing.
    """
    native,softmax,utils,devices,added=_verify_vendored(magma_root.resolve())
    origcwd=Path.cwd()
    os.chdir(magma_root/'third_party'/'LLMCompass')
    try:
        fp16=utils.data_type_dict['fp16']
        device=devices.device_dict['A100_80GB_fp16']
        timings={}
        shapes=gqa_shapes(w,token_index)
        for name,d in shapes.items():
            kind=d['kind']
            if kind=='matmul':
                op=native.Matmul(fp16)
                op(utils.Tensor(list(d['input_a']),fp16),utils.Tensor(list(d['input_b']),fp16))
                overhead=device.compute_module.overhead.matmul
            elif kind=='batched_matmul':
                op=native.BatchedMatmul(fp16)
                op(utils.Tensor(list(d['input_a']),fp16),utils.Tensor(list(d['input_b']),fp16))
                overhead=device.compute_module.overhead.matmul
            else:
                op=softmax.Softmax(fp16)
                op(utils.Tensor(list(d['input_a']),fp16))
                overhead=device.compute_module.overhead.softmax
            t=float(op.compile_and_simulate(device,'heuristic-GPU'))+float(overhead)
            if not math.isfinite(t) or t<0:
                raise RuntimeError('invalid LLMCompass native timing '+name)
            timings[name]=t
        stages={name:timings.get(name) for name in REQUIRED_NON_SCORE_STAGES}
        # LLMCompass has no exact native signed INT4 QK, RoPE, RMSNorm, or SwiGLU.
        # No source-unsupported stage may be replaced by GelU/LayerNorm.
        point=GQAGPUPoint(context=w.prefix_tokens+w.suffix_tokens+token_index+1,
            batch=w.batch,query_heads=w.query_heads,kv_heads=w.kv_heads,head_dim=w.head_dim,
            d_model=4096,ffn_dim=14336,layers=w.layers,vocab=128256,gpu_precision='fp16',
            score_precision='fp16',stage_seconds=stages,
            score_full_s=timings['full_score'],score_suffix_s=timings['suffix_score'],
            score_prefix_int4_s=None,source=MAGMA_NATIVE_SOURCE,
            operator_evidence='native_llmcompass_A100_FP16_HBM2e_model_geometry_matched_GQA')
        point.validate(w,token_index)
        return {'schema':'cim-shared-prefix-llmcompass-gqa-stage-point/v1','status':'PARTIAL_NATIVE_FP16_GQA',
                'model_id':point.model_id,'source':MAGMA_NATIVE_SOURCE,
                'gpu_point':asdict(point),'geometry':shapes,'supported_native_stages':sorted(timings),
                'unsupported_stages':point.missing+['score_prefix_int4'],
                'limitations':['FP16/HBM2e, not source-primary BF16',
                               'operator mapping valid for GQA, but structural and INT4 operators unmodeled',
                               'BatchedMatmul uses grouped Q, shares K per KV head, no MHA K storage replication',
                               'not complete decoder TPOT']}
    finally:
        os.chdir(origcwd)
        for p in added:
            sys.path.remove(str(p))
