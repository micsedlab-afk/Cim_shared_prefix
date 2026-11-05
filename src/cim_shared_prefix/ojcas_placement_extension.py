"""Paper-preserving full-layer residency optimization for the original 128-macro S2.

Fixed: submitted 128 macros per 128-token engine, area and QK-only endpoint.
Signedness and score reconstruction remain in the original *external* path.
Alternative K+/K- on-macro replication MUST NOT be imported or substituted.

This is a genuine finite-resource policy/ablation on the manuscript's engines.
Outputs are exact logical engine-wave and K4-tile-write COUNTS, plus optional
dimensionless costs. There is no assumed measured clock, bus, TPOT or energy.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from math import ceil,isfinite
from typing import Iterable

from .ojcas_manuscript_baseline import (
    BATCHES,PREFIXES,AREAS_MM2,N_LAYERS,S1_K_TILE_BYTES,
    CIM_ENGINE_AREA_MM2,paper_capacity,
)


@dataclass(frozen=True)
class Candidate:
    prefix_tokens:int
    batch:int
    area_mm2:int
    total_engines:int
    tiles_per_layer:int
    pinned_complete_layers:int
    streamed_engines:int
    decode_batch_steps:int
    compute_tile_waves_per_batch_step:int
    compute_wave_units_total:int
    cold_K4_tile_writes:int
    recurring_K4_tile_writes:int
    K4_tile_writes_total:int
    K4_tile_writes_saved_vs_no_pin:int
    source_K4_programming_bytes_total:int
    def record(self)->dict:return asdict(self)


def candidates(batch:int,prefix:int,area_mm2:int,*,decode_batch_steps:int)->tuple[Candidate,...]:
    """Pin k FULL layers; hold all other macros as a reusable scratch pool.

    All 32 transformer layers are causally ordered. Pinned layer engines are
    unavailable as scratch for other layers (no free eviction/reprogramming).
    Each unpinned layer is serviced in ceil(tiles_per_layer/scratch_engines)
    waves, with every B-query batch costing B wave units per wave. Tile
    programming uses one source K4 copy, no physical signed-bank replication.
    First batch step programs all 32 layers once; each subsequent step
    reprograms unpinned layers. A repeated batch sharing the same prefix can
    retain pinned layers across groups only under explicit residency.
    """
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError("batch is outside original OJCAS grid")
    if type(prefix) is not int or prefix not in PREFIXES:
        raise ValueError("prefix is outside original OJCAS grid")
    if type(area_mm2) is not int or area_mm2 not in AREAS_MM2:
        raise ValueError("area is outside original OJCAS grid")
    if type(decode_batch_steps) is not int or decode_batch_steps<1:
        raise ValueError("decode_batch_steps must be positive integer")
    e=paper_capacity(area_mm2).s2_original_engine_count
    t=ceil(prefix/128)
    max_pins=min(N_LAYERS,(e-1)//t) # scratch >=1 if unpinned layers
    result=[]
    for k in range(max_pins+1):
        scratch=e-k*t
        if k==N_LAYERS:  # if the complete prefix fits, no scratch necessary
            scratch=0
        if k<N_LAYERS and scratch<1:
            raise AssertionError("unfittable scratch plan")
        waves_per_step=k+(N_LAYERS-k)*ceil(t/scratch) if k<N_LAYERS else N_LAYERS
        cold=N_LAYERS*t
        recurring=(decode_batch_steps-1)*(N_LAYERS-k)*t
        writes=cold+recurring
        full_writes=decode_batch_steps*N_LAYERS*t
        assert 0<=writes<=full_writes
        result.append(Candidate(
            prefix_tokens=prefix,batch=batch,area_mm2=area_mm2,
            total_engines=e,tiles_per_layer=t,pinned_complete_layers=k,
            streamed_engines=scratch,decode_batch_steps=decode_batch_steps,
            compute_tile_waves_per_batch_step=waves_per_step,
            compute_wave_units_total=waves_per_step*decode_batch_steps*batch,
            cold_K4_tile_writes=cold,recurring_K4_tile_writes=recurring,
            K4_tile_writes_total=writes,
            K4_tile_writes_saved_vs_no_pin=full_writes-writes,
            source_K4_programming_bytes_total=writes*S1_K_TILE_BYTES,
        ))
    return tuple(result)


def dimensionless_cost(p:Candidate,lambda_write_per_tile:float)->float:
    """Cost unit: one complete batch-B=1 *compute wave*.

    lambda is the time of one 128-token K4 tile program relative to a B=1
    compute-wave time; arbitrary nonnegative dimensionless sensitivity.
    Multiply compute waves by batch. No claims of physical bandwidth/clock.
    """
    if not isinstance(lambda_write_per_tile,(float,int)) or isinstance(lambda_write_per_tile,bool) or not isfinite(lambda_write_per_tile) or lambda_write_per_tile<0:
        raise ValueError("nonnegative finite dimensionless lambda required")
    return p.compute_wave_units_total+lambda_write_per_tile*p.K4_tile_writes_total


def optimize(batch:int,prefix:int,area_mm2:int,*,decode_batch_steps:int,
             lambda_write_per_tile:float)->tuple[Candidate,Candidate,float]:
    plans=candidates(batch,prefix,area_mm2,decode_batch_steps=decode_batch_steps)
    baseline=plans[0]
    chosen=min(plans,key=lambda q:(dimensionless_cost(q,lambda_write_per_tile),
                                  q.pinned_complete_layers))
    base=dimensionless_cost(baseline,lambda_write_per_tile)
    chosen_cost=dimensionless_cost(chosen,lambda_write_per_tile)
    assert chosen_cost <= base*(1+1e-12)
    return baseline,chosen,(base-chosen_cost)/base


def first_pin_threshold(batch:int,prefix:int,area_mm2:int,*,decode_batch_steps:int)->float|None:
    """Exact k=0 versus k=1 switch under fixed-scratch scheduling.

    inf: pin1 increases compute but there are no warm steps (no opportunity
    to amortize one-time pinned K), None: cannot physically pin one layer.
    Threshold is a NORMALIZED price, not a runtime measurement.
    """
    p=candidates(batch,prefix,area_mm2,decode_batch_steps=decode_batch_steps)
    if len(p)<2:return None
    old,new=p[:2]
    dwrite=old.K4_tile_writes_total-new.K4_tile_writes_total
    dwave=new.compute_wave_units_total-old.compute_wave_units_total
    assert dwrite>=0 and dwave>=0
    if dwrite==0:
        return float("inf") if dwave>0 else None
    return dwave/dwrite


def run_grid(*,decode_batch_steps:int=128,lambda_values:Iterable[float]=(1.,10.,40.))->tuple[list[dict],dict]:
    lambdas=tuple(lambda_values)
    if not lambdas:raise ValueError("at least one lambda required")
    rows=[]
    summary={}
    for a in AREAS_MM2:
        acc={str(v):{"positive_gain":0,"selected_pins_hist":{}} for v in lambdas}
        can_pin=0
        for p in PREFIXES:
            for b in BATCHES:
                baseline=candidates(b,p,a,decode_batch_steps=decode_batch_steps)[0]
                threshold=first_pin_threshold(b,p,a,decode_batch_steps=decode_batch_steps)
                if threshold is not None:can_pin+=1
                row={
                    "B":b,"Lp":p,"Aadd_mm2":a,"Lu":128,
                    "original_engine_macros":128,"original_engines":baseline.total_engines,
                    "tiles_per_layer":baseline.tiles_per_layer,
                    "baseline_compute_wave_units_total":baseline.compute_wave_units_total,
                    "baseline_source_K4_writes_total":baseline.K4_tile_writes_total,
                    "first_pin_threshold_lambda_normalized":threshold if threshold is not None and isfinite(threshold) else None,
                    "candidate_full_layers_possible":len(candidates(b,p,a,decode_batch_steps=decode_batch_steps))-1,
                    "original_manuscript_S2_stage_speedup":None,
                    "new_policy_measured_TPOT":None,
                }
                for val in lambdas:
                    _,q,gain=optimize(b,p,a,decode_batch_steps=decode_batch_steps,
                                       lambda_write_per_tile=val)
                    key=f"lambda_{val:g}"
                    row[key+"_pinned_layers"]=q.pinned_complete_layers
                    row[key+"_compute_wave_units_total"]=q.compute_wave_units_total
                    row[key+"_source_K4_writes_total"]=q.K4_tile_writes_total
                    row[key+"_dimensionless_cost_gain_fraction"]=gain
                    if gain>1e-12:acc[str(val)]["positive_gain"]+=1
                    counts=acc[str(val)]["selected_pins_hist"]
                    k=str(q.pinned_complete_layers)
                    counts[k]=counts.get(k,0)+1
                rows.append(row)
        summary[str(a)]={"cases":len(PREFIXES)*len(BATCHES),
                          "feasible_to_pin_one_full_layer_cases":can_pin,
                          "lambda_sensitivities":acc}
    assert len(rows)==728
    return rows,{
        "schema":"ojcas_original_S2_128_macro_layer_residency_ablation/v1",
        "endpoint":"original shared-prefix QK^T, 128-macro engine, external signed handling",
        "evidence_class":"exact area/mapping combinatorial tile/wave counts + dimensionless cost sensitivity, not GPU/CIM runtime",
        "decode_batch_steps":decode_batch_steps,
        "lambda_definition":"time per source packed K4 128-token tile programming divided by one B=1 compute-wave time; NOT measured",
        "interpretation":"sequential layers, whole-layer pins, scratch engine tiles only, no other area or extra sign bank charged",
        "same_shared_prefix_reused_across_batches":True,
        "uncertainty":"missing physical interface and compute times; no decoder speedup is qualified",
        "original_S2_728_stage_results":"preserved unchanged in ojcas_manuscript_baseline.py",
        "areas":summary,
    }
