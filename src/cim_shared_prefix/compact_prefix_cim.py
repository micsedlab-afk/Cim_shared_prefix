"""Hardware-budgeted shared-prefix centroid attention with deterministic Q4 mapping.

Algorithmic provenance: count-weighted key-centroid/mean-value attention is
existing prior art (Multipole Attention, NeurIPS 2025). This module tests
whether such compression changes *physically possible* 7nm signed-K4 CIM
residency, and implements an exact finite-capacity dynamic-programming policy
over measured (not assumed) per-layer attention approximation errors.

No claim of a novel clustering algorithm or characterized O4 ADC arithmetic.
Q4 is separately labeled ideal signed-integer arithmetic with explicit
per-centroid scales and per-query scales, before analog circuit effects.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor
import numpy as np
from .area_optimizer import CIM_7NM, SRAM_7NM


@dataclass
class CentroidSummary:
    keys: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    radius: np.ndarray
    assignments: np.ndarray
    signed_k4: np.ndarray
    key_scale: np.ndarray
    value_storage: str

    @property
    def num_representatives(self) -> int:
        return int(len(self.counts))


def cluster_prefix(keys, values, representatives:int, iterations:int=8,\n                   value_storage:str="fp16")->CentroidSummary:
    """Deterministic farthest-point initialization and Lloyd centroids.

    Keys and values are *post-RoPE* and share KV-head identity. No positional
    flattening across KV heads. Empty clusters are rejected and reinitialized.
    """
    if value_storage not in ("fp16","fp32"):
        raise ValueError("value storage must be explicit FP16 hardware or FP32 oracle")
    k=np.asarray(keys,dtype=np.float32)
    v=np.asarray(values,dtype=np.float32)
    if (k.ndim!=2 or v.shape!=k.shape or
        not np.all(np.isfinite(k)) or not np.all(np.isfinite(v))):
        raise ValueError("finite aligned two-dimensional keys and values required")
    p,d=k.shape
    if not (type(representatives) is int and 1<=representatives<=p and
            type(iterations) is int and iterations>=1):
        raise ValueError("positive number of representatives and iterations required")
    if representatives==p:
        idx=np.arange(p)
        centers=k.copy()
        assign=idx.copy()
    else:
        # Seed a representative close to the key mean; add farthest points.
        centers=np.empty((representatives,d),dtype=np.float32)
        first=int(np.argmin(np.sum((k-k.mean(axis=0))**2,axis=1)))
        centers[0]=k[first]
        mind2=np.sum((k-centers[0])**2,axis=1)
        for i in range(1,representatives):
            pick=int(np.argmax(mind2))
            centers[i]=k[pick]
            mind2=np.minimum(mind2,np.sum((k-centers[i])**2,axis=1))
        assign=np.full(p,-1,dtype=np.int64)
        for _ in range(iterations):
            dist=(np.sum(k*k,axis=1)[:,None]+
                  np.sum(centers*centers,axis=1)[None,:]-
                  2*k@centers.T)
            candidate=dist.argmin(axis=1)
            # Ensure nonempty clusters, reassign from largest clusters'
            # most distant members if initialization became degenerate.
            counts=np.bincount(candidate,minlength=representatives)
            missing=np.flatnonzero(counts==0)
            for empty in missing:
                losses=dist[np.arange(p),candidate].copy()
                losses[counts[candidate]<=1]=-np.inf
                far=int(np.argmax(losses))
                counts[candidate[far]]-=1
                candidate[far]=int(empty)
                counts[empty]+=1
            assign=candidate
            for j in range(representatives):
                centers[j]=k[assign==j].mean(axis=0)
    n=np.bincount(assign,minlength=representatives)
    if np.any(n==0):
        raise AssertionError("empty centroid after deterministic reassignment")
    kc=np.stack([k[assign==i].mean(axis=0) for i in range(representatives)])
    vc=np.stack([v[assign==i].mean(axis=0) for i in range(representatives)])
    # V storage is physically FP16: round every centroid value to the
    # actual 16-bit representation before subsequent attention readout.
    if value_storage=="fp16":
        vc=vc.astype(np.float16).astype(np.float32)
    rad=np.array([np.linalg.norm(k[assign==i]-kc[i],axis=1).max()
                  for i in range(representatives)],dtype=np.float32)
    scale=np.maximum(np.max(np.abs(kc),axis=1)/7.,1e-12)
    # Count fits 16 bits for all 128/2048-prefix experiments; analog
    # per-column K scale is stored with FP16 metadata in the memory model.
    if np.max(n)>65535:
        raise ValueError("centroid multiplicity exceeds physical uint16 metadata")
    scale=scale.astype(np.float16).astype(np.float32)
    k4=np.clip(np.rint(kc/scale[:,None]),-7,7).astype(np.int8)
    return CentroidSummary(kc,vc,n.astype(np.int32),rad,assign,k4,
                           scale,value_storage)


def _softmax_context(logits,values):
    x=np.asarray(logits,dtype=np.float64)
    m=np.max(x,axis=-1,keepdims=True)
    ex=np.exp(x-m)
    z=np.sum(ex,axis=-1,keepdims=True)
    o=(ex/z)@np.asarray(values,dtype=np.float64)
    lse=(m+np.log(z)).squeeze(-1)
    return o.astype(np.float32),lse.astype(np.float32)


def exact_prefix_context(queries, keys, values):
    q=np.asarray(queries,dtype=np.float32)
    k=np.asarray(keys,dtype=np.float32)
    if q.ndim!=2 or k.ndim!=2 or q.shape[-1]!=k.shape[-1]:
        raise ValueError("head dimensions must match")
    return _softmax_context(q@k.T/np.sqrt(q.shape[-1]),values)


def centroid_prefix_context(queries, summary:CentroidSummary,
                            mode:str="fp32"):
    """Return per-query context and logsumexp for shared-prefix fusion.

    mode='q4k4' is IDEAL integer signed multiplication (not O4 ADC response).
    The per-centroid K scale and per-query Q scale must also be implemented
    by downstream digital peripherals; no charge is hidden in this function.
    """
    q=np.asarray(queries,dtype=np.float32)
    if q.ndim!=2 or q.shape[1]!=summary.keys.shape[1]:
        raise ValueError("query/centroid head dimensions differ")
    d=q.shape[1]
    if mode=="fp32":
        s=q@summary.keys.T / np.sqrt(d)
    elif mode=="k4":
        dec=summary.signed_k4.astype(np.float32)*summary.key_scale[:,None]
        s=q@dec.T/np.sqrt(d)
    elif mode=="q4k4":
        qs=np.maximum(np.max(np.abs(q),axis=1)/7.,1e-12)
        qs=qs.astype(np.float16).astype(np.float32)
        q4=np.clip(np.rint(q/qs[:,None]),-7,7).astype(np.int32)
        prod=q4@summary.signed_k4.astype(np.int32).T
        s=prod*(qs[:,None]*summary.key_scale[None,:])/np.sqrt(d)
    else:
        raise ValueError("unknown arithmetic mode")
    return _softmax_context(s+np.log(summary.counts.astype(np.float32))[None,:],
                            summary.values)


def output_error_metrics(queries,keys,values,summary:CentroidSummary,
                         mode:str="fp32")->dict:
    a,al=exact_prefix_context(queries,keys,values)
    b,bl=centroid_prefix_context(queries,summary,mode=mode)
    delta=b-a
    norms=np.linalg.norm(a,axis=1)
    rel=np.linalg.norm(delta,axis=1)/np.maximum(norms,1e-6)
    return {
       "M":summary.num_representatives,"mode":mode,
       "relative_L2_mean":float(rel.mean()),
       "relative_L2_p95":float(np.quantile(rel,.95)),
       "absolute_L2_mean":float(np.linalg.norm(delta,axis=1).mean()),
       "LSE_abs_mean":float(np.abs(al-bl).mean()),
       "queries":int(len(queries)),
    }


def native_macro_tile_areas(kv_heads:int,head_dim:int,representative_tile:int=16)->tuple[float,int,int]:
    """Complete K+/K- and FP16 V macros for one layer and 16 rep positions.

    Require V allocation in full 512x512 macros; no fractional SRAM cells.
    This uses native 64×16 4b CIM K tiling, with two signed K banks.
    """
    if min(kv_heads,head_dim,representative_tile)<=0 or representative_tile!=16:
        raise ValueError("requires native 16-representative tile")
    k_mac=kv_heads*ceil(head_dim/64)*2
    v_mac=ceil(kv_heads*16*head_dim*2/(SRAM_7NM.capacity_bits//8))
    return k_mac*CIM_7NM.area_mm2+v_mac*SRAM_7NM.area_mm2,k_mac,v_mac


def macro_summary_area(num_layers:int,kv_heads:int,head_dim:int,
                       representatives:int)->dict:
    if representatives%16 or not 16<=representatives:
        raise ValueError("representatives must be >=16 and 16-aligned")
    unit,k,v=native_macro_tile_areas(kv_heads,head_dim)
    tiles=representatives//16*num_layers
    # Two 16-bit metadata entries per centroid: multiplicity and K scale.
    meta=num_layers*kv_heads*representatives*4
    meta_macros=ceil(meta/(SRAM_7NM.capacity_bits//8))
    return {
       "layers":num_layers,"M":representatives,
       "signed_K_CIM_macros":tiles*k,
       "FP16_V_SRAM_macros":tiles*v,
       "metadata_SRAM_macros":meta_macros,
       "signed_K_V_metadata_macro_area_mm2":tiles*unit+meta_macros*SRAM_7NM.area_mm2,
       "macro_only":True,
       "unmodeled_peripheral_area_mm2":None,
    }


def select_layer_representatives(error_by_layer:np.ndarray,
                                 candidate_representatives:tuple[int,...],
                                 kv_heads:int,head_dim:int,
                                 storage_budget_mm2:float)->dict:
    """Exact integer DP for minimizing source *measured* error proxy.

    Error rows must come from held-out real queries. Every layer gets >=16
    representatives. DP uses exact integral macro tile units and separately
    reserves metadata capacity *after* the tile allocation. No processing
    latency or quality guarantees are inferred.
    """
    e=np.asarray(error_by_layer,dtype=np.float64)
    choices=tuple(candidate_representatives)
    if (e.ndim!=2 or e.shape[1]!=len(choices) or
        len(set(choices))!=len(choices) or
        any(m%16 or m<16 for m in choices) or
        not np.all(np.isfinite(e)) or np.any(e<0)):
        raise ValueError("invalid layer error/representative matrix")
    tile_area,_,_=native_macro_tile_areas(kv_heads,head_dim)
    nlayer=len(e)
    # Metadata reserve is rounded conservatively at the largest candidate
    # every layer: required capacity cannot exceed this figure.
    max_metadata=nlayer*kv_heads*max(choices)*4
    metamac=ceil(max_metadata/(SRAM_7NM.capacity_bits//8))
    available=storage_budget_mm2-metamac*SRAM_7NM.area_mm2
    cap=floor((available+1e-12)/tile_area)
    if cap<0:
        raise ValueError("insufficient source macro area for metadata")
    dp={0:(0.,())}
    for layer in range(nlayer):
        nxt={}
        for used,(err,history) in dp.items():
            for c,m in enumerate(choices):
                cur=used+m//16
                if cur>cap:continue
                objective=err+float(e[layer,c])
                prev=nxt.get(cur)
                if prev is None or objective<prev[0]-1e-13:
                    nxt[cur]=(objective,history+(m,))
        dp=nxt
        if not dp:raise ValueError("no feasible per-layer summarization")
    best_used,(best_error,allocation)=min(dp.items(),key=lambda p:(p[1][0],p[0]))
    return {
       "per_layer_representatives":list(allocation),
       "total_representative_layer_tiles":best_used,
       "tile_capacity_after_conservative_metadata":cap,
       "error_proxy_sum":float(best_error),
       "error_proxy_uniform_64_sum":(
           float(e[:,choices.index(64)].sum()) if 64 in choices else None
       ),
       "macro_area_including_max_metadata_mm2":best_used*tile_area+metamac*SRAM_7NM.area_mm2,
       "budget_mm2":storage_budget_mm2,
       "objective":"minimize held-out measured per-layer mean relative-L2 prefix-context error; not PPL",
       "metadata_reserved_worst_case":True,
    }
