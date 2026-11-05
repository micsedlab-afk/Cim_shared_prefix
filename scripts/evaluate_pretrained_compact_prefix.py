#!/usr/bin/env python3
"""Genuine pretrained GQA evidence: exact causal baseline vs compact-prefix CIM.

CPU-only public SmolLM2-135M pretrained checkpoint, pinned revision.
Perform (1) real post-RoPE QKV activation error on *different* calibration
and held-out paragraphs, (2) source-macro-budgeted per-layer centroid
selection, (3) full 30-layer teacher-forced post-prefix NLL with hooked
attention outputs, including full-P exact control and ideal signed Q4K4.

This is genuine model inference. It is NOT simulated Llama-8B quality or
characterized analog O4 ADC; no GPU/CIM runtime acceleration is claimed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cim_shared_prefix.compact_prefix_cim import (
    cluster_prefix, centroid_prefix_context, exact_prefix_context,
    output_error_metrics, select_layer_representatives, macro_summary_area,
)

MODEL="HuggingFaceTB/SmolLM2-135M"
REVISION="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TONE_A=(
 "An electrical engineer studies a small array of semiconductor memories. "
 "Every cell stores a charge, and the readout circuitry compares its response "
 "with a reference. The design uses row decoders, column switches, and sense "
 "amplifiers. When the clock changes, data moves through a finite set of "
 "registers. Latency and energy depend on the physical wires and the number "
 "of simultaneous commands. Measurements must distinguish voltage, power, "
 "time, area, and operations. A controller verifies causality before "
 "retiring a transaction. "
)
TONE_B=(
 "In a quiet library, researchers examine an old map and several notebooks. "
 "The map shows rivers, villages, bridges, and roads that cross the hills. "
 "Each notebook records observations from a different season. A historian "
 "compares details across the records and studies where the routes changed. "
 "The team creates a catalog of reliable places, dates, and events before "
 "writing its summary. Small mistakes in the ordering of records can change "
 "the meaning of the account. "
)


def source_qkv(module,hidden,position_embeddings):
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    B,S,_=hidden.shape
    D=module.head_dim
    q=module.q_proj(hidden).view(B,S,module.num_heads,D).transpose(1,2)
    k=module.k_proj(hidden).view(B,S,module.num_key_value_heads,D).transpose(1,2)
    v=module.v_proj(hidden).view(B,S,module.num_key_value_heads,D).transpose(1,2)
    if position_embeddings is None:
        raise ValueError("pinned Llama RoPE source position embeddings missing")
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    return q.float(),k.float(),v.float()


def token_sequence(tokenizer,text,n):
    seq=tokenizer.encode(text,add_special_tokens=False)
    if not seq:
        raise ValueError("empty tokenizer result")
    target=(seq*ceil_div(n,len(seq)))[:n]
    return target


def ceil_div(n,d):return (n+d-1)//d


@contextmanager
def capture_attention(model):
    data={}
    handles=[]
    for lid,layer in enumerate(model.model.layers):
        def mk_hook(i):
            def hook(module,args,kwargs):
                hidden=kwargs.get("hidden_states")
                if hidden is None:
                    hidden=args[0]
                q,k,v=source_qkv(module,hidden,kwargs.get("position_embeddings"))
                data[i]=(q.detach().cpu().numpy(),k.detach().cpu().numpy(),v.detach().cpu().numpy())
            return hook
        handles.append(layer.self_attn.register_forward_pre_hook(mk_hook(lid),with_kwargs=True))
    try:yield data
    finally:
        for h in handles:h.remove()


def measurement_matrix(activations,P:int,O:int,reps=(16,32,64,128),
                       mode="q4k4"):
    L=len(activations)
    arr=np.empty((L,len(reps)),dtype=np.float64)
    extra=[]
    for lid in range(L):
        q,k,v=activations[lid]
        B,Hq,S,D=q.shape
        Hkv=k.shape[1]
        g=Hq//Hkv
        if B!=1 or Hq%Hkv or S<P+O:
            raise ValueError("pretrained activation/GQA shape mismatch")
        error_by_reps=[[] for _ in reps]
        for head in range(Hkv):
            qgroup=q[0,head*g:(head+1)*g,P:P+O,:].reshape(-1,D)
            key=k[0,head,:P,:]
            val=v[0,head,:P,:]
            for j,M in enumerate(reps):
                summ=cluster_prefix(key,val,M)
                met=output_error_metrics(qgroup,key,val,summ,mode)
                error_by_reps[j].append(met["relative_L2_mean"])
                extra.append({
                    "layer":lid,"KV_head":head,"M":M,"mode":mode,
                    "context_relative_L2_mean":met["relative_L2_mean"],
                    "context_relative_L2_p95":met["relative_L2_p95"],
                    "LSE_abs_mean":met["LSE_abs_mean"],
                    "query_count":met["queries"],
                })
        arr[lid]=[np.mean(z) for z in error_by_reps]
    return arr,extra


@contextmanager
def replace_attention_prefix(model, P:int, M_by_layer:list[int],mode:str):
    """Change real model attention output at t>=P; prefix remains exact."""
    import torch
    if len(M_by_layer)!=len(model.model.layers):
        raise ValueError("one representative count per pretrained layer")
    handles=[]
    for lid,layer in enumerate(model.model.layers):
        M=int(M_by_layer[lid])
        if M==0:
            continue  # exact pretrained GPU attention; no CIM data stored
        def mk_hook(mcount):
            def hook(mod,args,kwargs,out):
                hidden=kwargs.get("hidden_states")
                if hidden is None:hidden=args[0]
                B,S,_=hidden.shape
                if S<=P:return out
                if B!=1:raise ValueError("pretrained pilot only supports B1")
                q,k,v=source_qkv(mod,hidden,kwargs.get("position_embeddings"))
                q=q.detach().cpu().numpy(); k=k.detach().cpu().numpy(); v=v.detach().cpu().numpy()
                Hq=mod.num_heads;Hkv=mod.num_key_value_heads;D=mod.head_dim;g=Hq//Hkv
                output_heads=np.empty((S-P,Hq,D),dtype=np.float32)
                for h in range(Hkv):
                    kk=k[0,h];vv=v[0,h]
                    summ=cluster_prefix(kk[:P],vv[:P],mcount)
                    qhead=q[0,h*g:(h+1)*g]
                    for t in range(P,S):
                        qq=qhead[:,t,:]
                        pref,pref_lse=centroid_prefix_context(qq,summ,mode=mode)
                        suffix,suffix_lse=exact_prefix_context(
                            qq,kk[P:t+1],vv[P:t+1]
                        )
                        both=np.maximum(pref_lse,suffix_lse)
                        pa=np.exp(pref_lse-both)
                        sa=np.exp(suffix_lse-both)
                        fused=(pa[:,None]*pref+sa[:,None]*suffix)/(pa+sa)[:,None]
                        output_heads[t-P,h*g:(h+1)*g]=fused
                output=torch.from_numpy(output_heads).to(out[0].device,dtype=out[0].dtype)
                projection=mod.o_proj(output.reshape(1,S-P,Hq*D))
                changed=out[0].clone()
                changed[:,P:,:]=projection
                return (changed,*out[1:])
            return hook
        handles.append(layer.self_attn.register_forward_hook(mk_hook(M),with_kwargs=True))
    try:yield
    finally:
        for h in handles:h.remove()


def nll_and_logits(model,ids,prefix):
    import torch
    with torch.no_grad():
        logits=model(input_ids=ids,use_cache=False).logits
    target=ids[:,prefix+1:]
    x=logits[:,prefix:-1,:].float()
    nll=torch.nn.functional.cross_entropy(
        x.reshape(-1,x.shape[-1]),target.reshape(-1),reduction="mean")
    return float(nll),x.detach().cpu()


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(22848)
    np.random.seed(22848)
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,torch_dtype=torch.float32,
        attn_implementation="eager",trust_remote_code=False)
    model.eval()
    dims={"layers":model.config.num_hidden_layers,
          "KV_heads":model.config.num_key_value_heads,
          "query_heads":model.config.num_attention_heads,
          "D":model.config.hidden_size//model.config.num_attention_heads}
    if dims!={"layers":30,"KV_heads":3,"query_heads":9,"D":64}:
        raise ValueError("pinned pretrained GQA model geometry differs")
    P=args.prefix;O=args.output
    if P<128 or P%16 or O<3 or P+O>8192:
        raise ValueError("must evaluate meaningful aligned prefix and tail")
    token_a=torch.tensor([token_sequence(tokenizer,TONE_A,P+O)],dtype=torch.long)
    token_b=torch.tensor([token_sequence(tokenizer,TONE_B,P+O)],dtype=torch.long)

    with capture_attention(model) as ca:
        calibrated,_=nll_and_logits(model,token_a,P)
    with capture_attention(model) as cb:
        reference_nll,reference_logits=nll_and_logits(model,token_b,P)

    reps=(16,32,64,128)
    calibration,cal_rows=measurement_matrix(ca,P,O,reps,mode="q4k4")
    heldout,test_rows=measurement_matrix(cb,P,O,reps,mode="q4k4")
    base_area=macro_summary_area(dims["layers"],dims["KV_heads"],dims["D"],64)
    allocation=select_layer_representatives(
        calibration,reps,dims["KV_heads"],dims["D"],
        base_area["signed_K_V_metadata_macro_area_mm2"]+1e-8,
    )
    choices=allocation["per_layer_representatives"]
    uniform_error=float(heldout[:,reps.index(64)].mean())
    selected_error=float(np.mean([heldout[i,reps.index(m)]
                                  for i,m in enumerate(choices)]))

    # Full-P should reproduce exact baseline attention algebra and NLL.
    with replace_attention_prefix(model,P,[P]*dims["layers"],"fp32"):
        oracle_nll,oracle_logits=nll_and_logits(model,token_b,P)
    max_logit_delta=float((reference_logits-oracle_logits).abs().max())
    if max_logit_delta>0.025 or abs(reference_nll-oracle_nll)>0.002:
        raise AssertionError(
            "full-P attention reconstruction does not match pretrained "
            f"reference: max_logit_abs_diff={max_logit_delta} "
            f"delta_NLL={oracle_nll-reference_nll}"
        )

    with replace_attention_prefix(model,P,[64]*dims["layers"],"q4k4"):
        uniform_nll,_=nll_and_logits(model,token_b,P)
    with replace_attention_prefix(model,P,choices,"q4k4"):
        selected_nll,_=nll_and_logits(model,token_b,P)

    output={
        "schema":"source-pretrained-smollm2-135m-CIM-centroid-prefix-pilot/v1",
        "checkpoint":MODEL,"revision":REVISION,"transformers_version":"4.46.3",
        "model_geometry":dims,"prefix":P,"tail":O,"seed":22848,
        "actual_checkpoint_weights_used":True,
        "attention_mode":"eager; real pretrained post-RoPE QKV; modified tail outputs propagated through all layers",
        "prefix_inference":"full true attention; no compressed prefill",
        "suffix":"exact suffix self-attention at every tail position with LSE fusion",
        "q4k4_mode":"ideal signed integer reconstruction; not characterized 4-bit ADC",
        "calibration_prompt":"original authored electrical engineering passage",
        "heldout_prompt":"distinct original authored library passage",
        "evaluated_next_token_targets":O-1,
        "calibration_reference_NLL":calibrated,
        "heldout_exact_NLL":reference_nll,
        "heldout_full_P_reconstruction_NLL":oracle_nll,
        "heldout_full_P_max_logit_abs_diff":max_logit_delta,
        "heldout_uniform_M64_NLL":uniform_nll,
        "heldout_allocated_NLL":selected_nll,
        "heldout_uniform_M64_delta_NLL":uniform_nll-reference_nll,
        "heldout_allocated_delta_NLL":selected_nll-reference_nll,
        "calibration_uniform_M64_Q4_context_rel_L2_mean":float(calibration[:,reps.index(64)].mean()),
        "heldout_uniform_M64_Q4_context_rel_L2_mean":uniform_error,
        "heldout_allocated_Q4_context_rel_L2_mean":selected_error,
        "allocation":allocation,
        "equal_area_uniform_M64":base_area,
        "scientific_status":"actual pretrained-model teacher-forced NLL and attention-error pilot; limited to 135M GQA, small prompt set",
        "not_claimed":["Llama 8B PPL","hardware O4 response","real GPU/CIM latency","novel centroid clustering","published representative quality threshold"],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    with args.per_layer.open("w",newline="") as fd:
        w=csv.DictWriter(fd,fieldnames=["layer","M","calibration_relative_L2","heldout_relative_L2","allocated_M"])
        w.writeheader()
        for lid in range(dims["layers"]):
            for j,m in enumerate(reps):
                w.writerow({"layer":lid,"M":m,
                    "calibration_relative_L2":calibration[lid,j],
                    "heldout_relative_L2":heldout[lid,j],
                    "allocated_M":choices[lid]})
    print("PRETRAINED_COMPACT_SUMMARY "+json.dumps(output,sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--prefix",type=int,default=128)
    ap.add_argument("--output",type=int,default=8)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--per-layer",type=Path,required=True)
    evaluate(ap.parse_args())
