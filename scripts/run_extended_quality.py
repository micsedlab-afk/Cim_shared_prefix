#!/usr/bin/env python3
"""Genuine scalar OPT/Mistral R0--R3 quality, model-specific calibration.

No CPU/tiny-model stand-in is ever emitted as evaluated model quality.
Requires pinned upstream SRAM2LLM, Transformers 4.46.2, full model snapshots,
WikiText validation and a CUDA GPU. Each output is checkpointed by model,
calibration, frozen token stream and evaluator identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cim_shared_prefix.quality_evaluation import (
    SEED, RESAMPLES, aggregate_units, fixed_windows, token_digest, write_aggregate,
)
from cim_shared_prefix.quality_inference import (
    Intervention, MODEL_REVISIONS, install_model_adapter, CACHE_SOURCE_BLOB,
    MISTRAL_SOURCE_BLOB, OPT_SOURCE_BLOB, TRANSFORMERS_VERSION, _git_blob_sha,
)

PREFIXES={"mistral":(512,2048,16384), "opt":(256,512,1024)}
DATASET_ID="Salesforce/wikitext"
DATASET_REV="b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG="wikitext-2-raw-v1"
BACKEND_SHA256="11d8d252b0bbe1f3ed558609a7a0de188b317b9b4875a29cd6d34925f800ba36"
RATIOS=tuple(round(.20+.05*i,2) for i in range(17))
POLICY_ID="sinangil_static_per_layer_r3aware_nearest_o4_v2_model_specific"


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def save_json(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp')
    with temp.open('wb') as f:
        f.write(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def load_dependencies(upstream:Path):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA full-model execution unavailable; no OPT/Mistral NLL/PPL may be emitted')
    import transformers
    if transformers.__version__!=TRANSFORMERS_VERSION:
        raise RuntimeError(f'exact Transformers {TRANSFORMERS_VERSION} required')
    backend=upstream/'sources/SRAM2LLM/src/sram2llm/sinangil_qk.py'
    if not backend.is_file() or digest(backend)!=BACKEND_SHA256:
        raise RuntimeError('original pinned Sinangil Q4/K4 backend is missing or modified')
    sys.path.insert(0,str(upstream/'sources/SRAM2LLM/src'))
    sys.path.insert(0,str(upstream))
    from sram2llm.sinangil_qk import (
        ADC_QUANTIZER_NEAREST,LayerADCPolicy,SinangilQKExecutor,
    )
    from experiments.sram2llm_shared_prefix.sinangil_backend import (
        load_fig16_bitplane_response,target_descriptor_sha256,
    )
    if _git_blob_sha(Path(transformers.cache_utils.__file__))!=CACHE_SOURCE_BLOB:
        raise RuntimeError('cache-utils source blob differs from frozen 4.46.2')
    executor=SinangilQKExecutor(target_descriptor_sha256=target_descriptor_sha256(),
                                key_chunk_size=16,check_native=True)
    response=load_fig16_bitplane_response()
    return (torch,transformers,LayerADCPolicy,ADC_QUANTIZER_NEAREST,
            executor,response)


def load_model(kind:str, model_root:Path|None, allow_download:bool):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model_id,rev=MODEL_REVISIONS[kind]
    source=str(model_root) if model_root is not None else model_id
    if model_root is not None and not model_root.is_dir():
        raise FileNotFoundError(model_root)
    local=model_root is not None or not allow_download
    if model_root is not None and model_root.resolve().name != rev:
        raise RuntimeError('local model directory must be an exact-revision Hugging Face snapshot')
    tokenizer=AutoTokenizer.from_pretrained(source,revision=rev,
                                             local_files_only=local,use_fast=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('R0 BF16 protocol requires a BF16-capable CUDA GPU')
    dtype=torch.bfloat16  # OPT source declares FP16; explicit BF16 evaluation copy.
    model=AutoModelForCausalLM.from_pretrained(source,revision=rev,
            torch_dtype=dtype,local_files_only=local,attn_implementation='sdpa',
            low_cpu_mem_usage=True).to('cuda:0').eval()
    conf=model.config
    if kind=='mistral':
        assert (conf.num_hidden_layers,conf.num_attention_heads,conf.num_key_value_heads,
                conf.hidden_size,conf.sliding_window,conf.max_position_embeddings)==(32,32,8,4096,4096,32768)
    else:
        assert (conf.num_hidden_layers,conf.num_attention_heads,conf.hidden_size,
                conf.max_position_embeddings)==(32,32,2560,2048)
    if conf._attn_implementation!='sdpa':raise RuntimeError('source SDPA required')
    return model,tokenizer


def load_tokens(tokenizer, *, cache_dir:Path, allow_download:bool):
    import numpy as np
    if not allow_download:
        os.environ['HF_DATASETS_OFFLINE']='1'
    from datasets import load_dataset
    cache_dir.mkdir(parents=True,exist_ok=True)
    dset=load_dataset(DATASET_ID,DATASET_CONFIG,split='validation',revision=DATASET_REV,
                      cache_dir=str(cache_dir),download_mode='reuse_dataset_if_exists')
    joined='\n\n'.join(dset['text'])
    result=tokenizer(joined,add_special_tokens=False,return_attention_mask=False)
    ids=np.asarray(result['input_ids'],dtype='<i8')
    if ids.ndim!=1 or len(ids)==0 or (ids<0).any():raise RuntimeError('invalid source token stream')
    return ids,hashlib.sha256(joined.encode()).hexdigest()


def clone_cache(kind,master):
    if kind=='mistral':
        from transformers import DynamicCache
        if not isinstance(master,DynamicCache):raise ValueError('Mistral requires native DynamicCache')
        return DynamicCache.from_legacy_cache(master.to_legacy_cache())
    if not isinstance(master,(tuple,list)):
        raise ValueError('OPT requires legacy per-layer cache tuples')
    return tuple((k,v) for k,v in master)


def cache_signature(kind,cache):
    if kind=='mistral':
        entries=cache.to_legacy_cache()
    else: entries=cache
    return tuple((int(k.data_ptr()),int(v.data_ptr()),tuple(k.shape),tuple(v.shape))
                 for k,v in entries)


def local_tokens(tokens,window,device):
    import numpy as np
    import torch
    lp,lu=window['prefix_tokens'],window['suffix_tokens']
    x=np.array(tokens[window['start']:window['end']],dtype='<i8',copy=True)
    if token_digest(x)!=window['window_sha256'] or token_digest(x[lp+1:])!=window['targets_sha256']:
        raise ValueError('token input and paired target hash mismatch')
    return torch.as_tensor(x,device=device,dtype=torch.long).view(1,-1)


def prefix_cache(model,kind,inputs,lp,controller):
    import torch
    controller.enabled=False
    if kind=='mistral':
        from transformers import DynamicCache
        cache=DynamicCache()
        with torch.inference_mode():result=model(input_ids=inputs[:,:lp],past_key_values=cache,use_cache=True)
    else:
        with torch.inference_mode():result=model(input_ids=inputs[:,:lp],use_cache=True)
    master=result.past_key_values
    if len(cache_signature(kind,master))!=32:
        raise RuntimeError('source prefill lacks all 32 KV layers')
    return master


def nll_vector(logits, targets):
    import torch
    import torch.nn.functional as F
    if logits.shape[:2]!=targets.shape:raise ValueError('target shape differs from scored suffix')
    with torch.inference_mode():
        nll=-F.log_softmax(logits.float(),dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(nll).all():raise ValueError('nonfinite model NLL')
    return nll[0].detach().cpu().tolist()


def run_suffix(model,kind,controller,inputs,master,window,arm):
    import torch
    lp,lu=window['prefix_tokens'],window['suffix_tokens']
    controller.enabled=True
    controller.arm=arm
    controller.prefix=lp;controller.suffix=lu
    controller.window_id=f"{window['sample_id']}:p{lp}:u{lu}"
    controller.source_calls.clear();controller.active_scores.clear()
    start=time.perf_counter()
    with torch.inference_mode():
        suffix=model(input_ids=inputs[:,lp:lp+lu],
                     past_key_values=clone_cache(kind,master),use_cache=True)
    torch.cuda.synchronize()
    if len(controller.source_calls)!=32 or any(n!=1 for n in controller.source_calls.values()):
        raise RuntimeError('not all 32 attention layers used the qualified SDPA hook')
    out=nll_vector(suffix.logits,inputs[:,lp+1:lp+lu+1])
    controller.enabled=False
    return out,time.perf_counter()-start,dict(controller.active_scores)


def qualify(model,kind,controller,tokens,plan,out):
    import torch
    trials=[]
    dev=[w for w in plan['development'] if w['sample_id']=='dev_000'
         and w['prefix_tokens'] in (min(PREFIXES[kind]),max(PREFIXES[kind]))]
    for window in dev:
        controller.reference_kernel="native"
        ids=local_tokens(tokens,window,'cuda:0')
        master=prefix_cache(model,kind,ids,window['prefix_tokens'],controller)
        before=cache_signature(kind,master)
        native,_,_=run_suffix(model,kind,controller,ids,master,window,'R0')
        # The independent baseline uses the same source model with the
        # interception wrapper disabled for every layer.
        controller.enabled=False
        with torch.inference_mode():
            ref=model(input_ids=ids[:,window['prefix_tokens']:-1],
                      past_key_values=clone_cache(kind,master),use_cache=True)
        baseline=nll_vector(ref.logits,ids[:,window['prefix_tokens']+1:])
        delta=max(abs(a-b) for a,b in zip(native,baseline))
        if len(native)!=window['suffix_tokens'] or delta>2e-5:
            raise RuntimeError(f'R0 source versus intercepted numeric mismatch {delta:.8g}')
        if cache_signature(kind,master)!=before:
            raise RuntimeError('master prefix cache was mutated by suffix execution')
        trials.append({'prefix_tokens':window['prefix_tokens'],'max_abs_token_nll_delta':delta,
                       'sample_id':window['sample_id'],'status':'PASS'})
    report={'schema':'model_quality_adapter_qualification/v1','model_id':MODEL_REVISIONS[kind][0],
            'model_revision':MODEL_REVISIONS[kind][1],
            'dataset_token_stream_sha256':plan['token_stream_sha256'],
            'status':'PASS','source':f'Transformers {TRANSFORMERS_VERSION} model SDPA',
            'trials':trials,'tolerance':2e-5}
    save_json(out/'qualification.json',report)
    return report


def calibration_digest(plan,kind):
    return hashlib.sha256(canonical({'schema':'model_quality_calibration_subset/v1',
        'model':MODEL_REVISIONS[kind],
        'token_stream_sha256':plan['token_stream_sha256'],
        'development_windows':plan['development'],
        'ratio_candidates':RATIOS})).hexdigest()


def run_development(model,kind,controller,tokens,plan,callback):
    controller.capture=callback
    controller.arm='R0'
    controller.reference_kernel="eager"
    for w in plan['development']:
        ids=local_tokens(tokens,w,'cuda:0')
        cache=prefix_cache(model,kind,ids,w['prefix_tokens'],controller)
        controller.capture=callback
        run_suffix(model,kind,controller,ids,cache,w,'R0')
    controller.capture=None


def calibrate(model,kind,controller,tokens,plan,out,backend):
    (_,_,LayerADCPolicy,nearest,executor,response)=backend
    subset=calibration_digest(plan,kind)
    maxima={i:0.0 for i in range(32)}
    observations={i:0 for i in range(32)}
    def scan(layer,state):
        obs=executor.collect_r2_calibration(state)
        if obs.source_domain_violation_count:
            raise RuntimeError('source Fig.16 support exceeded in calibration')
        maxima[layer]=max(maxima[layer],obs.full_scale_aggregate)
        observations[layer]+=1
    run_development(model,kind,controller,tokens,plan,scan)
    if set(maxima)!=set(range(32)) or any(v<=0 for v in maxima.values()):
        raise RuntimeError('missing or zero per-layer ADC full scale')
    error={layer:{r:[] for r in RATIOS} for layer in range(32)}
    def sweep(layer,state):
        ref=executor.execute_r1(state,layer_id=str(layer),window_id='calibration-r1').raw_scores.double()
        denom=max(float(torch.sum(ref*ref)),1e-12)
        for ratio in RATIOS:
            policy=LayerADCPolicy(layer=layer,full_scale_aggregate=maxima[layer]*ratio,
                      calibration_subset_digest=subset,quantizer=nearest)
            raw=executor.execute_r3(state,policy=policy,response=response,
                  layer_id=str(layer),window_id='calibration-r3').raw_scores.double()
            metric=float(torch.sum((raw-ref)**2))/denom
            if not math.isfinite(metric):raise RuntimeError('nonfinite ADC development MSE')
            error[layer][ratio].append(metric)
    import torch
    run_development(model,kind,controller,tokens,plan,sweep)
    policies=[];bad=[]
    for layer in range(32):
        candidates=[{'ratio':r,'mean_normalized_mse':math.fsum(error[layer][r])/len(error[layer][r]),
                     'observation_count':len(error[layer][r])} for r in RATIOS]
        best=min(candidates,key=lambda x:(x['mean_normalized_mse'],-x['ratio']))
        if best['ratio'] in (RATIOS[0],RATIOS[-1]):bad.append(layer)
        policy=LayerADCPolicy(layer=layer,full_scale_aggregate=maxima[layer]*best['ratio'],
                              calibration_subset_digest=subset,quantizer=nearest)
        policies.append({'layer':layer,'full_scale':maxima[layer],
                         'selected_ratio':best['ratio'],
                         'selected_full_scale':maxima[layer]*best['ratio'],
                         'policy_digest':policy.digest,'candidates':candidates})
    data={'schema':'model_specific_r3aware_calibration/v1',
          'status':'QUALIFIED' if not bad else 'BLOCKED_BOUNDARY_RATIO',
          'model_id':MODEL_REVISIONS[kind][0], 'revision':MODEL_REVISIONS[kind][1],
          'policy_id':POLICY_ID,'subsample':'first eight suffix queries per development window',
          'dev_blocks':plan['development_blocks'],'token_stream_sha256':plan['token_stream_sha256'],
          'subset_digest':subset,'source_backend_sha256':BACKEND_SHA256,
          'fig16_response_digest':response.digest,
          'candidate_ratios':list(RATIOS),
          'source_method':'R3 vs R1 normalized MSE, static 4-bit nearest O4 calibration; model-specific development-only',
          'observations_by_layer':observations,'boundary_layers':bad,'layers':policies}
    save_json(out/'calibration.json',data)
    if bad:raise RuntimeError(f'R3-aware ADC ratio selected boundary on layers {bad}; review before held-out inference')
    return data


def load_calibration(path,kind,plan,backend):
    (_,_,LayerADCPolicy,nearest,_,response)=backend
    data=json.loads(path.read_text())
    if (data.get('schema')!='model_specific_r3aware_calibration/v1'
            or data.get('status')!='QUALIFIED'
            or data['model_id']!=MODEL_REVISIONS[kind][0]
            or data['revision']!=MODEL_REVISIONS[kind][1]
            or data['token_stream_sha256']!=plan['token_stream_sha256']
            or data['subset_digest']!=calibration_digest(plan,kind)
            or data['source_backend_sha256']!=BACKEND_SHA256
            or data['fig16_response_digest']!=response.digest
            or data['policy_id']!=POLICY_ID
            or data['candidate_ratios']!=list(RATIOS)
            or len(data['layers'])!=32):
        raise ValueError('unsupported or mismatched model-specific ADC calibration')
    result={}
    for e in data['layers']:
        i=e['layer']
        if i in result or not isinstance(i,int) or not 0<=i<32 or e['selected_ratio'] in (RATIOS[0],RATIOS[-1]):
            raise ValueError('invalid calibrated layer')
        policy=LayerADCPolicy(layer=i,full_scale_aggregate=e['selected_full_scale'],
                   calibration_subset_digest=data['subset_digest'],quantizer=nearest)
        if (policy.digest!=e['policy_digest']
                or e['selected_full_scale']!=e['full_scale']*e['selected_ratio']):
            raise ValueError('ADC policy digest/range mismatch')
        result[i]=policy
    if set(result)!=set(range(32)):raise ValueError('missing model-specific calibrated layers')
    return result


def evaluate(model,kind,controller,tokens,plan,out,backend,quality_provenance):
    calibration_path=out/'calibration.json'
    controller.adc_policies=load_calibration(calibration_path,kind,plan,backend)
    controller.fig16_response=backend[5]
    qualification=json.loads((out/'qualification.json').read_text())
    if qualification.get('status')!='PASS' or qualification.get('dataset_token_stream_sha256')!=plan['token_stream_sha256']:
        raise RuntimeError('model-specific source equivalence qualification required')
    controller.reference_kernel="eager"
    calibration_sha=digest(calibration_path)
    protocol={'model_id':MODEL_REVISIONS[kind][0], 'revision':MODEL_REVISIONS[kind][1],
              'model_class':'MHA' if kind=='opt' else 'GQA',
              'source_transformers_version':TRANSFORMERS_VERSION,
              'source_backend_sha256':BACKEND_SHA256,
              'calibration_sha256':calibration_sha,
              'token_stream_sha256':plan['token_stream_sha256'],
              'evaluation_blocks':plan['evaluation_blocks'],
              'prefixes':plan['prefixes'],'suffix_tokens':plan['suffix'],
              'batch':1,'mode':'scalar_teacher_forced_next_token_nll',
              'r0_kernel':'source-fidelity eager BF16, FP32 softmax',
              'dtype':'bfloat16',
              'cache_policy':'independent R0-R3 cache wrappers with identical prefix content'}
    protocol_hash=hashlib.sha256(canonical(protocol)).hexdigest()
    save_json(out/'protocol.json',dict(protocol,protocol_sha256=protocol_hash))
    units=[]
    for w in plan['evaluation']:
        ids=local_tokens(tokens,w,'cuda:0')
        master=prefix_cache(model,kind,ids,w['prefix_tokens'],controller)
        signature=cache_signature(kind,master)
        for arm in ('R0','R1','R2','R3'):
            path=out/'units'/f"{w['sample_id']}_p{w['prefix_tokens']}__{arm}.json"
            if path.exists():
                result=json.loads(path.read_text())
                if result.get('protocol_sha256')!=protocol_hash:
                    raise ValueError('stale or foreign checkpoint')
            else:
                vector,seconds,active_scores=run_suffix(model,kind,controller,ids,master,w,arm)
                if len(vector)!=w['suffix_tokens'] or not all(math.isfinite(v) for v in vector):
                    raise RuntimeError('non-finite/missing evaluated suffix tokens')
                result={'schema':'evaluated_model_quality_unit/v1',
                        'protocol_sha256':protocol_hash,
                        'model_id':MODEL_REVISIONS[kind][0],
                        'model_revision':MODEL_REVISIONS[kind][1],
                        'batch':1,'arm':arm,'sample_id':w['sample_id'],
                        'prefix_tokens':w['prefix_tokens'],'suffix_tokens':w['suffix_tokens'],
                        'targets_sha256':w['targets_sha256'],
                        'window_sha256':w['window_sha256'],
                        'token_count':len(vector),'nll_sum':math.fsum(vector),
                        'mean_nll':math.fsum(vector)/len(vector),
                        'token_nll':vector,'duration_s':seconds,
                        'visible_prefix_cim_scores_by_layer':active_scores}
                save_json(path,result)
            units.append(result)
            if cache_signature(kind,master)!=signature:
                raise ValueError('master prefix KV cache changed between arms')
            print(f"{kind} {w['sample_id']} lp={w['prefix_tokens']} {arm} nll={result['mean_nll']:.6f}",flush=True)
    s=aggregate_units(units,model_id=MODEL_REVISIONS[kind][0],
         prefixes=PREFIXES[kind],suffix=128,n_samples=8,
         bootstrap_seed=SEED,resamples=RESAMPLES)
    s['protocol_sha256']=protocol_hash
    write_aggregate(s,out,dict(quality_provenance,protocol=protocol,
                              protocol_sha256=protocol_hash))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('opt','mistral'),required=True)
    p.add_argument('--mode',choices=('qualify','calibrate','evaluate','aggregate'),required=True)
    p.add_argument('--upstream-root',type=Path,required=False)
    p.add_argument('--model-root',type=Path)
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--work-root',type=Path,default=Path('results/quality/extended'))
    args=p.parse_args()
    out=args.work_root/args.model
    if args.mode=='aggregate':
        records=list((out/'units').glob('*.json'))
        data=[json.loads(x.read_text()) for x in records]
        result=aggregate_units(data,model_id=MODEL_REVISIONS[args.model][0],
                   prefixes=PREFIXES[args.model],suffix=128,n_samples=8)
        protocol=json.loads((out/'protocol.json').read_text())
        if any(u['protocol_sha256']!=protocol['protocol_sha256'] for u in data):
            raise ValueError('mixed provenance in persisted units')
        write_aggregate(result,out,{'protocol':protocol,'reaggregated_without_model_execution':True})
        return
    if args.upstream_root is None:
        p.error('--upstream-root required for genuine calibration and inference')
    backend=load_dependencies(args.upstream_root)
    torch=backend[0]
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,tokenizer=load_model(args.model,args.model_root,args.allow_download)
    controller=Intervention(model=args.model,executor=backend[4],fig16_response=backend[5])
    install_model_adapter(model,args.model,controller)
    tokens,text_digest=load_tokens(tokenizer,cache_dir=args.work_root/'dataset_cache',
                                   allow_download=args.allow_download)
    plan=fixed_windows(tokens,model_id=MODEL_REVISIONS[args.model][0],
                       prefixes=PREFIXES[args.model],suffix=128,
                       n_evaluation=8,n_development=2)
    plan.update({'model_revision':MODEL_REVISIONS[args.model][1],
                 'dataset_id':DATASET_ID,'dataset_revision':DATASET_REV,
                 'dataset_config':DATASET_CONFIG,'joined_text_sha256':text_digest})
    save_json(out/'window_manifest.json',plan)
    if max(PREFIXES[args.model])+128>model.config.max_position_embeddings:
        raise ValueError('quality window exceeds source model context')
    if args.mode=='qualify':
        qualify(model,args.model,controller,tokens,plan,out)
    elif args.mode=='calibrate':
        qualification=json.loads((out/'qualification.json').read_text())
        if qualification.get('status')!='PASS' or qualification.get('dataset_token_stream_sha256')!=plan['token_stream_sha256']:
            raise RuntimeError('run model-specific equivalence qualification first')
        calibrate(model,args.model,controller,tokens,plan,out,backend)
    elif args.mode=='evaluate':
        provenance={'model_id':MODEL_REVISIONS[args.model][0],
             'revision':MODEL_REVISIONS[args.model][1],
             'dataset_revision':DATASET_REV,'token_stream_sha256':plan['token_stream_sha256'],
             'source_backend_sha256':BACKEND_SHA256,
             'physical_circuit_response':'source-digitized simulated Sinangil Fig.16; not measured model quality on silicon',
             'ADC':'model-specific development calibrated project assumption',
             'inference_on':'full HF model, scalar teacher forcing, all four causal arms',
             'compute_dtype':'bfloat16',
             'source_model_dtype':'float16' if args.model=='opt' else 'bfloat16'}
        evaluate(model,args.model,controller,tokens,plan,out,backend,provenance)


if __name__=='__main__':
    main()
