"""Finite-residency prefix attention (CIM QK + digital online softmax/PV).

Physical storage: published 7nm signed Q4/K4 Sinangil complete macros for K,
published 7nm SRAM complete macro capacity for *FP16 V*, and independently
rounded interstage FIFOs. No V quantization or native CIM softmax is invented.

Causal schedule per 16-prefix-token/KV-head wave:
unsigned QK -> signed scale -> digital reduction -> SRAM V fetch ->
stable online softmax -> BF16-input/FP32-accumulation PV. Finite 2-wave
FIFOs enforce producer backpressure. Context FP16 and FP32 LSE transfer
once at final prefix attention endpoint. Suffix attention and stable LSE
fusion execute on GPU, outside this conditional prefix engine.

GPU full-prefix comparison takes source-labeled operator models. Every
digital/peripheral rate and support-area parameter remains a sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, floor, inf, isfinite, exp, log
from typing import Sequence

from .area_optimizer import CIM_7NM, SRAM_7NM, CIM_PUBLISHED_ACCESS_S
from .contracts import Workload
from .integrated_schedule import SerialRates


@dataclass(frozen=True)
class ContextRates:
    """All digital, external and local SRAM bandwidths are *scenario* rates."""
    key_supply_Bps: float
    key_program_Bps: float
    v_supply_Bps: float
    v_program_Bps: float
    query_Bps: float
    local_v_read_Bps: float
    exp_per_s: float
    scalar_ops_per_s: float
    pv_mac_per_s: float
    context_output_Bps: float
    partial_scalings_per_s: float
    signed_adds_per_s: float
    controller_s_per_layer: float = 0.0
    source: str = "explicit unqualified integration-rate sensitivity"

    def __post_init__(self):
        for key in (
            "key_supply_Bps", "key_program_Bps", "v_supply_Bps",
            "v_program_Bps", "query_Bps", "local_v_read_Bps",
            "exp_per_s", "scalar_ops_per_s", "pv_mac_per_s",
            "context_output_Bps", "partial_scalings_per_s", "signed_adds_per_s",
        ):
            v = getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<=0:
                raise ValueError("positive finite scenario rate required: "+key)
        if not isfinite(self.controller_s_per_layer) or self.controller_s_per_layer<0:
            raise ValueError("controller must be nonnegative")
        if not self.source.strip():
            raise ValueError("scenario rate provenance required")


@dataclass(frozen=True)
class ContextWave:
    group_count: int
    query_microbatch: int
    native_slots: int
    scores: int
    macs_pv: int
    fetched_v_bytes: int
    raw_o4_bytes: int
    fp32_partials_bytes: int
    complete_scores_bytes: int
    digital_exps: int
    digital_scalar_ops: int
    durations_s: tuple[float, ...]


@dataclass(frozen=True)
class ContextPipeline:
    model: str
    area_budget_mm2: float
    external_support_mm2: float
    total_physical_area_mm2: float
    cim_k_macros: int
    cim_k_area_mm2: float
    sram_v_macros: int
    sram_v_area_mm2: float
    sram_pipeline_macros: int
    sram_pipeline_area_mm2: float
    max_pinned_layers: int
    active_macro_ports: int
    wave_groups: int
    query_microbatch: int
    total_waves: int
    full_v_bytes_per_layer: int
    per_token_v_read_bytes: int
    prefix_score_link_bytes_avoided: int
    context_and_lse_link_bytes: int
    source_gpu_prefix_qk_s: float | None
    warm_attention_context_s: float
    cold_KV_supply_and_program_s: float
    mean_context_s: float
    prefix_context_vs_gpu_QK_model_difference_s: float | None
    digital_MAC_rate_density_required_per_mm2: float
    stage_totals_s: tuple[float, ...]
    stage_names: tuple[str, ...] = (
        "native_unsigned_QK", "signed_partial_scaling",
        "signed_reduction", "local_V_SRAM_fetch",
        "online_stable_softmax", "FP16V_FP32_PV_accumulate",
    )
    evidence: str = "published_7nm_macro_geometry_plus_explicit_digital_bandwidth_and_area_sensitivities"

    def json_record(self):
        return asdict(self)


def _blocking_pipeline(waves: Sequence[ContextWave]) -> float:
    """Earliest-finish six-stage DAG with only two wave-sized FIFO slots per edge."""
    n=6
    starts=[[] for _ in range(n)]
    done=[[] for _ in range(n)]
    for i,wave in enumerate(waves):
        for stage,delay in enumerate(wave.durations_s):
            previous_wave=done[stage][i-1] if i else 0.0
            previous_stage=done[stage-1][i] if stage else 0.0
            begin=max(previous_wave,previous_stage)
            end=begin+delay
            if stage<n-1 and i>=2:
                # Producer cannot retire wave i into a full two-slot FIFO
                # until the consumer starts processing wave i-2.
                end=max(end,starts[stage+1][i-2])
            starts[stage].append(begin)
            done[stage].append(end)
    return done[-1][-1] if waves else 0.0


def simulate_prefix_context(
    w: Workload,
    *,
    area_mm2: float,
    support_mm2: float,
    ports: int,
    rates: ContextRates,
    v_bits: int = 16,
    gpu_prefix_qk_s: float | None = None,
    external_setup_s_per_token: float = 0.0,
) -> ContextPipeline | None:
    """Search B microbatches × contiguous KV-score wave widths.

    The complete K± and V layers are resident; a full layer cannot be
    duplicated between outputs or priced using bitcell-only area.
    P must be divisible by 16 to keep the complete token/group mapping.
    No GPU remaining-stage latency or system-level speedup is inferred.
    """
    if (w.key_bits!=4 or w.prefix_tokens%16 or
        w.head_dim%64 or w.query_heads%w.kv_heads):
        raise ValueError("requires tiled native signed Q4K4 GQA geometry")
    if v_bits != 16:
        raise ValueError("V remains FP16; new V quantization not qualified")
    if type(ports) is not int or ports<1:
        raise ValueError("positive port count required")
    for label,v in (("area",area_mm2),("support",support_mm2),
                    ("external_setup",external_setup_s_per_token)):
        if not isinstance(v,(int,float)) or not isfinite(v) or v<0:
            raise ValueError("invalid "+label)
    if area_mm2<=0 or support_mm2>=area_mm2:
        return None
    if gpu_prefix_qk_s is not None and (not isinstance(gpu_prefix_qk_s,(int,float))
                        or not isfinite(gpu_prefix_qk_s) or gpu_prefix_qk_s<=0):
        raise ValueError("GPU source QK must be positive finite or null")

    dt=w.head_dim//64
    ptiles=w.prefix_tokens//16
    groups=w.kv_heads*ptiles
    k_macros=groups*dt*2
    k_area=k_macros*CIM_7NM.area_mm2
    v_bytes=w.kv_heads*w.prefix_tokens*w.head_dim*2
    v_macros=ceil(v_bytes/(SRAM_7NM.capacity_bits//8))
    v_area=v_macros*SRAM_7NM.area_mm2
    fixed_area=k_area+v_area+support_mm2
    if fixed_area>=area_mm2:
        return None

    usable_ports=min(ports,k_macros)
    max_groups=min(groups,usable_ports//(2*dt))
    if not max_groups:
        return None
    qbytes=ceil(w.batch*w.query_heads*w.head_dim*4/8)
    input_s=qbytes/rates.query_Bps
    scores=w.batch*w.query_heads*w.prefix_tokens
    score_link_bytes=scores*4
    context_bytes=w.batch*w.query_heads*(w.head_dim*2+4)
    output_s=context_bytes/rates.context_output_Bps
    # Streaming online softmax must retain FP32 m,l,o for every query.
    state_bytes=w.batch*w.query_heads*(w.head_dim+2)*4
    options=[]
    for microbatch in range(1,w.batch+1):
        for width in range(1,max_groups+1):
            waves=[]
            for bstart in range(0,w.batch,microbatch):
                qb=min(microbatch,w.batch-bstart)*w.gqa_ratio
                left=groups
                idx=0
                while left:
                    count=min(left,width)
                    nscore=count*16*qb
                    npart=nscore*4*dt
                    nmac=nscore*w.head_dim
                    # V is loaded once per 16-token KV group and broadcast
                    # across its GQA queries, not multiplied by B*G.
                    vfetch=count*16*w.head_dim*2
                    first_h=idx//ptiles
                    last_h=(idx+count-1)//ptiles
                    qrows=(last_h-first_h+1)*qb
                    nexps=nscore+2*qrows
                    nscalars=5*nscore+3*qrows*w.head_dim
                    durations=(
                        2*qb*CIM_PUBLISHED_ACCESS_S,
                        npart/rates.partial_scalings_per_s,
                        (npart-nscore)/rates.signed_adds_per_s,
                        vfetch/rates.local_v_read_Bps,
                        nexps/rates.exp_per_s+nscalars/rates.scalar_ops_per_s,
                        nmac/rates.pv_mac_per_s,
                    )
                    waves.append(ContextWave(count,min(microbatch,w.batch-bstart),
                        2*qb,nscore,nmac,vfetch,ceil(npart/2),
                        npart*4,nscore*4,nexps,nscalars,durations))
                    idx+=count
                    left-=count
            if sum(x.scores for x in waves)!=scores:
                raise AssertionError("not all prefix score rows processed")
            # Every processing boundary has two slots; full QK digitized
            # partials and V chunks are charged independently. Online
            # accumulators persist across every wave.
            max_fifo=max(x.raw_o4_bytes+x.fp32_partials_bytes+
                         x.complete_scores_bytes+x.fetched_v_bytes for x in waves)
            stage_bytes=2*max_fifo+state_bytes
            buffer_macros=ceil(stage_bytes/(SRAM_7NM.capacity_bits//8))
            buffer_area=buffer_macros*SRAM_7NM.area_mm2
            total_area=fixed_area+buffer_area
            if total_area>area_mm2+1e-12:
                continue
            warm=input_s+_blocking_pipeline(waves)+output_s+rates.controller_s_per_layer+external_setup_s_per_token
            # First token programs complete physical images of all K± and
            # FP16 V SRAM macros. No preloading is silently free.
            k_bytes=k_macros*(CIM_7NM.capacity_bits//8)
            v_phys_bytes=v_macros*(SRAM_7NM.capacity_bits//8)
            cold=(k_bytes/rates.key_supply_Bps+k_bytes/rates.key_program_Bps+
                  v_phys_bytes/rates.v_supply_Bps+v_phys_bytes/rates.v_program_Bps)
            mean=warm+cold/w.output_tokens
            options.append((mean,warm,-width,-microbatch,width,microbatch,
                            waves,buffer_macros,buffer_area,total_area,cold))
    if not options:
        return None
    mean,warm,_nwidth,_nmicro,width,microbatch,waves,buffer_macros,buffer_area,total_area,cold=min(options)
    gpu_diff=mean-gpu_prefix_qk_s if gpu_prefix_qk_s is not None else None
    totals=tuple(sum(wv.durations_s[i] for wv in waves) for i in range(6))
    return ContextPipeline(
        model="published_7nm_Q4K4_prefix_QK_plus_digital_online_softmax_FP16V_PV",
        area_budget_mm2=area_mm2,external_support_mm2=support_mm2,
        total_physical_area_mm2=total_area,
        cim_k_macros=k_macros,cim_k_area_mm2=k_area,
        sram_v_macros=v_macros,sram_v_area_mm2=v_area,
        sram_pipeline_macros=buffer_macros,sram_pipeline_area_mm2=buffer_area,
        max_pinned_layers=min(w.layers,int(floor((area_mm2-support_mm2+1e-12)/
                                             (k_area+v_area+buffer_area)))),
        active_macro_ports=usable_ports,wave_groups=width,
        query_microbatch=microbatch,total_waves=len(waves),
        full_v_bytes_per_layer=v_bytes,
        per_token_v_read_bytes=sum(wv.fetched_v_bytes for wv in waves),
        prefix_score_link_bytes_avoided=score_link_bytes,
        context_and_lse_link_bytes=context_bytes,
        source_gpu_prefix_qk_s=gpu_prefix_qk_s,
        warm_attention_context_s=warm,
        cold_KV_supply_and_program_s=cold,
        mean_context_s=mean,
        prefix_context_vs_gpu_QK_model_difference_s=gpu_diff,
        digital_MAC_rate_density_required_per_mm2=rates.pv_mac_per_s/support_mm2 if support_mm2>0 else inf,
        stage_totals_s=totals,
    )


def online_prefix_context(scores: Sequence[float],
                          values: Sequence[Sequence[float]],
                          tile_size: int = 16) -> tuple[list[float],float]:
    """Numerical oracle for streaming stable prefix softmax+PV (one query head).

    Scores are *post*-ADC/reconstruction real scores as supplied by the
    approved signed K mapping; no claim of hardware numerical equivalence.
    V remains floating-point and can be quantized by its caller to FP16.
    """
    if type(tile_size) is not int or tile_size<1 or len(scores)==0 or len(scores)!=len(values):
        raise ValueError("nonempty aligned scores/V with positive tile size required")
    d=len(values[0])
    if d==0 or any(len(v)!=d for v in values):
        raise ValueError("consistent nonzero value feature count required")
    if any(not isfinite(s) for s in scores) or any(not all(isfinite(y) for y in v) for v in values):
        raise ValueError("finite scores and values required")
    m=-inf
    denom=0.0
    numer=[0.0]*d
    for first in range(0,len(scores),tile_size):
        ss=scores[first:first+tile_size]
        vv=values[first:first+tile_size]
        mt=max(m,max(ss))
        rescale=exp(m-mt) if m!=-inf else 0.0
        numer=[v*rescale for v in numer]
        denom*=rescale
        for score,row in zip(ss,vv):
            weight=exp(score-mt)
            denom+=weight
            for k,v in enumerate(row):
                numer[k]+=weight*v
        m=mt
    return [x/denom for x in numer],m+log(denom)


def fuse_prefix_suffix(prefix_context:Sequence[float],prefix_lse:float,
                       suffix_context:Sequence[float],suffix_lse:float)->list[float]:
    """Stable exact prefix/suffix decomposition in real arithmetic."""
    if len(prefix_context)!=len(suffix_context) or not prefix_context:
        raise ValueError("context dimension mismatch")
    if not isfinite(prefix_lse) or not isfinite(suffix_lse):
        raise ValueError("finite nonempty attention partitions required")
    m=max(prefix_lse,suffix_lse)
    a=exp(prefix_lse-m)
    b=exp(suffix_lse-m)
    return [(a*x+b*y)/(a+b) for x,y in zip(prefix_context,suffix_context)]
