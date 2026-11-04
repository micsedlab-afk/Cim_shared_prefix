"""Exact-geometry source-based macro occupancy and conditional S0/S1/S2 decoder.

Hardware facts are from the Sinangil 7-nm published 64x64 I4/W4/O4
macro. Its reported 5.5-ns native access characterizes *one* unsigned
64-by-16 array operation; available port concurrency, placement/programming,
reductions, links and all GPU operators require separately identified inputs.
Missing sources remain ``None``, never zero. No 7-nm digital-area inference.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, isfinite
from typing import Mapping

from .contracts import Workload

SINANGIL_DOI = '10.1109/JSSC.2020.3031290'
SINANGIL_DESCRIPTOR = ('micsedlab-afk/shared_prefix_cim@a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4:'
                      'inputs/sinangil7nm/target_descriptor.json')
SINANGIL_MAPPING = ('micsedlab-afk/shared_prefix_cim@a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4:'
                    'experiments/sram2llm_shared_prefix/sinangil_mappings.py')
SRAM_AREA_SOURCE = ('Yokoyama et al., VLSI 2020, DOI:10.1109/VLSICircuits18222.2020.9162985;'
                    ' complete 512x512 7nm SRAM macro density 29.2 Mbit/mm2')
LLAMA_CONFIG_SOURCE = ('micsedlab-afk/shared_prefix_cim@a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4:'
                       'docs/sram2llm/SCIENTIFIC_CONTRACT.md')


def require_positive(name: str, value: int | float, *, integer: bool = False) -> None:
    if isinstance(value, bool) or (integer and not isinstance(value, int)) or not isinstance(value,(int,float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive finite'+(' integer' if integer else ''))


@dataclass(frozen=True)
class Llama31:
    model_id: str = 'meta-llama/Llama-3.1-8B'
    d_model: int = 4096
    ffn_dim: int = 14336
    layers: int = 32
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 128256
    primary_reference_precision: str = 'bf16'

    def validate(self, w: Workload) -> None:
        if (w.layers,w.query_heads,w.kv_heads,w.head_dim)!=(self.layers,self.query_heads,self.kv_heads,self.head_dim):
            raise ValueError('not the source-frozen Llama-3.1-8B GQA geometry')
        if self.query_heads*self.head_dim!=self.d_model or self.query_heads//self.kv_heads!=4:
            raise ValueError('invalid Llama head geometry')
        if w.key_bits != 4:
            raise ValueError('Sinangil Q4/K4 path requires K4')


@dataclass(frozen=True)
class MacroSpec:
    rows: int = 64
    physical_cols: int = 64
    logical_outputs: int = 16
    weight_bits: int = 4
    input_bits: int = 4
    output_bits: int = 4
    capacity_bytes: int = 512
    area_mm2: float = 0.0032
    native_access_s: float = 5.5e-9
    source: str = SINANGIL_DESCRIPTOR
    evidence: str = 'reported_measured_silicon_macro_only'

    def __post_init__(self) -> None:
        if (self.rows,self.physical_cols,self.logical_outputs,self.weight_bits,self.input_bits,self.output_bits,self.capacity_bytes)!=(64,64,16,4,4,4,512):
            raise ValueError('incompatible or altered physical macro')
        if self.area_mm2!=0.0032 or self.native_access_s!=5.5e-9 or not self.source:
            raise ValueError('published macro parameters changed')


@dataclass(frozen=True)
class PhysicalCIM:
    macro_budget: int
    active_macro_ports: int
    q_bits: int = 4
    source: str = 'explicit physical macro-budget/concurrency sensitivity'

    def __post_init__(self) -> None:
        for n in ('macro_budget','active_macro_ports'):
            require_positive(n,getattr(self,n),integer=True)
        if self.q_bits not in (4,8):
            raise ValueError('only native Q4 or two-pass Q8 supported')
        if not self.source:
            raise ValueError('physical allocation source required')


@dataclass(frozen=True)
class MacroPlacement:
    layer_total_macros: int
    macros_required: int
    macro_budget: int
    pinned_per_layer: tuple[int,...]
    streamed_per_layer: tuple[int,...]
    scratch_macros: int
    initial_placement_bytes: int
    recurring_refill_bytes_per_token: int
    native_macro_calls_per_token: int
    native_cycles_floor_per_token: int
    native_cycles_single_scratch_schedule_per_token: int
    native_access_floor_s_per_token: float
    native_single_scratch_schedule_s_per_token: float
    complete_macro_area_required_mm2: float
    provisioned_macro_area_mm2: float
    source: str


def placement(w: Workload, physical: PhysicalCIM, macro: MacroSpec = MacroSpec()) -> MacroPlacement:
    """Fair whole-macro pinning across layers, one reusable scratch on misses.

    A scratch macro reloads each unpinned bank exactly once per generated step.
    It services all B x GQA queries before eviction; stored K is not broadcast-
    replicated. Each signed K bank receives Q+ and Q- per native query nibble.
    The compute floor credits ideal independent concurrency but no write time.
    The explicit single-scratch schedule serializes streamed bank compute.
    """
    if w.key_bits!=4 or w.query_heads%w.kv_heads or w.head_dim<=0:
        raise ValueError('unsupported Sinangil mapping geometry')
    d_tiles=ceil(w.head_dim/macro.rows)
    p_tiles=ceil(w.prefix_tokens/macro.logical_outputs)
    per_layer=w.kv_heads*d_tiles*p_tiles*2
    full=per_layer*w.layers
    if physical.macro_budget>=full:
        pinned=[per_layer]*w.layers; streamed=[0]*w.layers;scratch=0
    else:
        scratch=1
        pin_budget=physical.macro_budget-1
        pinned=[pin_budget//w.layers + int(l<pin_budget%w.layers) for l in range(w.layers)]
        if any(p>per_layer for p in pinned):
            raise AssertionError('invalid per-layer pin selection')
        streamed=[per_layer-n for n in pinned]
    per_bank_calls=2*w.batch*(w.query_heads//w.kv_heads)*(1 if physical.q_bits==4 else 2)
    native_calls=full*per_bank_calls
    optimistic_cycles=0
    scratch_cycles=0
    for pin,stream in zip(pinned,streamed):
        # Full layer dependency: layer l+1 cannot begin until l scores are ready.
        # Upper bound on concurrent independently selectable macro *ports*.
        active=min(physical.active_macro_ports,physical.macro_budget,per_layer)
        optimistic_cycles+=max(per_bank_calls,ceil(per_layer*per_bank_calls/active))
        if pin:
            scratch_cycles+=ceil(pin*per_bank_calls/min(pin,physical.active_macro_ports))
        scratch_cycles+=stream*per_bank_calls
    assert native_calls == w.layers*w.batch*w.query_heads*p_tiles*d_tiles*4*(1 if physical.q_bits==4 else 2)
    return MacroPlacement(per_layer,full,physical.macro_budget,tuple(pinned),tuple(streamed),scratch,
                          sum(pinned)*macro.capacity_bytes,sum(streamed)*macro.capacity_bytes,
                          native_calls,optimistic_cycles,scratch_cycles,
                          optimistic_cycles*macro.native_access_s,
                          scratch_cycles*macro.native_access_s,
                          full*macro.area_mm2,physical.macro_budget*macro.area_mm2,
                          f'{macro.source}; {physical.source}; signed mapping: {SINANGIL_MAPPING}')


def sram_storage(w: Workload, *, copies: int = 1) -> dict[str,object]:
    """Per-layer, per-KV-head physical SRAM macro storage, no port assumptions."""
    require_positive('copies',copies,integer=True)
    storage_per_head=ceil(w.prefix_tokens*w.head_dim*w.key_bits/8)
    macros_per_head=ceil(storage_per_head/(512*512//8))
    count=w.layers*w.kv_heads*macros_per_head*copies
    return {'macro_count':count,'macro_bits':512*512,'macro_area_mm2':(512*512)/(29.2e6),
            'total_macro_area_mm2':count*(512*512)/(29.2e6),
            'physical_key_bytes_one_copy':storage_per_head*w.kv_heads*w.layers,
            'source':SRAM_AREA_SOURCE,
            'status':'published_complete_sram_macro_footprint_only'}



@dataclass(frozen=True)
class SRAMPlacement:
    """Full-source-precision macro images, one shared K copy, across layers.

    This is a conservative full-macro-image storage schedule; physical read
    bandwidth, port geometry, control and GPU delivery still require evidence.
    """
    required_macros: int
    provisioned_macros: int
    pinned_per_layer: tuple[int, ...]
    streamed_per_layer: tuple[int, ...]
    scratch_macros: int
    initial_macro_image_bytes: int
    recurring_refill_macro_image_bytes: int
    physical_key_bytes_transferred_to_gpu_per_token: int
    provisioned_macro_area_mm2: float


def sram_placement(w: Workload, macro_budget: int) -> SRAMPlacement:
    require_positive('SRAM macro_budget',macro_budget,integer=True)
    s=sram_storage(w)
    nhead=ceil(ceil(w.prefix_tokens*w.head_dim*w.key_bits/8)/32768)
    per_layer=w.kv_heads*nhead
    required=w.layers*per_layer
    assert required==s['macro_count']
    if macro_budget>=required:
        pinned=[per_layer]*w.layers; streamed=[0]*w.layers;scratch=0
    else:
        budget=macro_budget-1;scratch=1
        pinned=[budget//w.layers+int(l<budget%w.layers) for l in range(w.layers)]
        streamed=[per_layer-n for n in pinned]
    return SRAMPlacement(required,macro_budget,tuple(pinned),tuple(streamed),scratch,
        sum(pinned)*32768,sum(streamed)*32768,s['physical_key_bytes_one_copy'],
        macro_budget*s['macro_area_mm2'])


def counts(w: Workload, physical: PhysicalCIM) -> dict[str,int|float|str]:
    p=placement(w,physical)
    scores=w.layers*w.batch*w.query_heads*w.prefix_tokens
    partials_per_score=4*ceil(w.head_dim/64)*(1 if physical.q_bits==4 else 2)
    return {'score_count_per_token':scores,
            'complete_score_bytes_per_token':scores*4,
            'logical_signed_qk_macs_per_token':scores*w.head_dim,
            'signed_partial_output_values_per_token':scores*partials_per_score,
            'digital_signed_combines_per_token':scores*(partials_per_score-1),
            'digital_partial_rescale_multiplies_per_token':scores*partials_per_score,
            'query_input_bytes_per_token':w.layers*ceil(w.batch*w.query_heads*w.head_dim*physical.q_bits/8),
            'signed_physical_key_bytes':p.macros_required*512,
            'refill_bytes_per_token':p.recurring_refill_bytes_per_token,
            'initial_placement_bytes':p.initial_placement_bytes,
            'native_macro_calls_per_token':p.native_macro_calls_per_token,
            'native_access_floor_s_per_token':p.native_access_floor_s_per_token,
            'native_single_scratch_schedule_s_per_token':p.native_single_scratch_schedule_s_per_token,
            'macro_area_mm2_provisioned':p.provisioned_macro_area_mm2,
            'macro_area_mm2_full_residency':p.complete_macro_area_required_mm2,
            'macro_evidence':'reported_measured_silicon_macro_only',
            'signed_mapping_evidence':'paper-grounded K split; project-assumption Q split',
            'native_floor_meaning':'macro-only access cycles with optimistic independent parallel ports, no I/O/programming/reduction'}


@dataclass(frozen=True)
class RateSensitivity:
    """All non-native rates are scenario assumptions, not silicon measurements."""
    prefix_link_Bps: float
    sram_to_gpu_Bps: float
    query_link_Bps: float
    score_link_Bps: float
    program_Bps: float
    reduction_adds_per_s: float
    partial_scale_mults_per_s: float
    quantization_s_per_layer: float
    provenance: Mapping[str,str]
    sram_write_Bps: float|None = None
    sram_write_provenance: str|None = None
    cim_controller_s_per_layer: float = 0.0
    sram_controller_s_per_layer: float = 0.0

    def __post_init__(self) -> None:
        for key in ('prefix_link_Bps','sram_to_gpu_Bps','query_link_Bps','score_link_Bps','program_Bps',
                    'reduction_adds_per_s','partial_scale_mults_per_s'):
            require_positive(key,getattr(self,key))
        v=self.quantization_s_per_layer
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<0:
            raise ValueError('invalid quantization timing')
        keys=('prefix_link','sram_to_gpu','query_link','score_link','program','reduction_adds','partial_scales','quantization')
        if not isinstance(self.provenance,Mapping) or any(not self.provenance.get(k) for k in keys):
            raise ValueError('every rate requires explicit provenance, including zero-cost assumptions')
        if self.sram_write_Bps is not None:
            require_positive('sram_write_Bps',self.sram_write_Bps)
            if not self.sram_write_provenance:
                raise ValueError('SRAM write service provenance required')
        for name in ('cim_controller_s_per_layer','sram_controller_s_per_layer'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or value<0:
                raise ValueError('invalid controller service '+name)
            if not self.provenance.get(name):
                raise ValueError('controller service provenance required, including zero-cost sensitivities')


def s2_components(w: Workload, physical: PhysicalCIM, rates: RateSensitivity) -> dict[str,float]:
    """Serial conditional CIM-side time, excluding GPU computation.

    Uses one scratch bank, no impossible instantaneous macro reprogramming.
    Initial placement is amortized over finite output length; steady-state
    generated tokens incur full streamed refills. ADC/O4 conversion is part
    of the published native macro and is not separately charged. Scales and
    sign/tile additions are distinct off-macro digital operations.
    """
    p=placement(w,physical); n=counts(w,physical)
    initial=p.initial_placement_bytes/rates.prefix_link_Bps + p.initial_placement_bytes/rates.program_Bps
    recurring=p.recurring_refill_bytes_per_token/rates.prefix_link_Bps + p.recurring_refill_bytes_per_token/rates.program_Bps
    return {'amortized_initial_prefix_placement_s':p.initial_placement_bytes/rates.prefix_link_Bps/w.output_tokens,
            'amortized_initial_programming_s':p.initial_placement_bytes/rates.program_Bps/w.output_tokens,
            'refill_s':p.recurring_refill_bytes_per_token/rates.prefix_link_Bps,
            'refill_programming_s':p.recurring_refill_bytes_per_token/rates.program_Bps,
            'macro_native_access_s':p.native_single_scratch_schedule_s_per_token,
            'query_transfer_s':n['query_input_bytes_per_token']/rates.query_link_Bps,
            'score_transfer_s':n['complete_score_bytes_per_token']/rates.score_link_Bps,
            'signed_reduce_s':n['digital_signed_combines_per_token']/rates.reduction_adds_per_s,
            'partial_rescale_s':n['digital_partial_rescale_multiplies_per_token']/rates.partial_scale_mults_per_s,
            'query_quantization_s':w.layers*rates.quantization_s_per_layer,
            'cim_controller_s':w.layers*rates.cim_controller_s_per_layer}


# Native Llama source defines SwiGLU, RMSNorm and RoPE. These are explicit
# required stages; GeLU and LayerNorm are not equivalent operators.
REQUIRED_NON_SCORE_STAGES = ('q_proj','k_proj','v_proj','out_proj','gate_proj','up_proj','down_proj',
    'attention_apply','softmax','rmsnorm_pre','rmsnorm_post','silu','gate_mul','rope',
    'kv_update','score_concat_scale','lm_head','sampling')


@dataclass(frozen=True)
class GQAGPUPoint:
    """Per-generated-token GPU full decoder profile; all stages per *layer* except head/sampling.

    Whole-model LM head and sampling are charged only once per token. Profile
    values may be None when a native tool does not implement an exact operator.
    ``score_s`` values must refer to full, suffix, or Q4K4 *GPU* prefix QK,
    separately and without borrowed precision or MHA extrapolation.
    """
    context: int
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    d_model: int
    ffn_dim: int
    layers: int
    vocab: int
    gpu_precision: str
    stage_seconds: Mapping[str,float|None]
    score_full_s: float|None
    score_suffix_s: float|None
    score_prefix_int4_s: float|None
    source: str
    model_id: str = 'meta-llama/Llama-3.1-8B'
    score_precision: str = 'bf16'
    prefix_gpu_precision: str = 'q4k4_signed'
    operator_evidence: str = 'native_vendor_model'

    def validate(self,w:Workload,index:int) -> None:
        if (self.context,self.batch,self.query_heads,self.kv_heads,self.head_dim,self.d_model,
            self.ffn_dim,self.layers,self.vocab)!=(w.prefix_tokens+w.suffix_tokens+index+1,w.batch,32,8,128,4096,14336,32,128256):
            raise ValueError('exact Llama GQA geometry/context mismatch')
        if self.gpu_precision not in ('bf16','fp16') or self.score_precision!=self.gpu_precision:
            raise ValueError('GPU operator precision must match score precision')
        if self.prefix_gpu_precision!='q4k4_signed':
            raise ValueError('SRAM-only GPU prefix QK needs exact signed Q4K4 kernel')
        if self.model_id!='meta-llama/Llama-3.1-8B' or not self.source or not self.operator_evidence:
            raise ValueError('untraceable model/profile')
        if set(self.stage_seconds)!=set(REQUIRED_NON_SCORE_STAGES):
            raise ValueError('required stage coverage missing')
        for name,v in {**self.stage_seconds,'score_full':self.score_full_s,'score_suffix':self.score_suffix_s,
                       'score_prefix_int4':self.score_prefix_int4_s}.items():
            if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<0):
                raise ValueError('invalid stage value '+name)
        if w.suffix_tokens+index==0 and self.score_suffix_s not in (0,None):
            raise ValueError('nonexistent suffix cannot have positive score time')

    @property
    def missing(self)->list[str]:
        return [k for k,v in self.stage_seconds.items() if v is None]

    def non_score_total_s(self, layers:int)->float|None:
        if self.missing:return None
        one=sum(float(v) for k,v in self.stage_seconds.items() if k not in ('lm_head','sampling'))
        return layers*one + float(self.stage_seconds['lm_head']) + float(self.stage_seconds['sampling'])


def compare(w:Workload,physical:PhysicalCIM, *,gpu_points:tuple[GQAGPUPoint,...]|None=None,
            rates:RateSensitivity|None=None, sram_budget_macros:int|None=None) -> dict[str,object]:
    """Matched completed-score S0/S1/S2 comparisons, failing closed on evidence gaps.

    S0: unmodified all-GPU full-context attention at the given GPU precision.
    S1: signed Q4K4 K in SRAM, transferred to GPU and computed with a *native*
        Q4K4 GPU prefix score kernel; suffix score remains native GPU.
    S2: mapped signed Q4K4 CIM prefix, 32-bit scores sent to GPU; suffix GPU.
    Every common non-QK decoder operator uses the *same* exact GQA record.
    """
    Llama31().validate(w)
    p=placement(w,physical)
    n=counts(w,physical)
    sram=sram_storage(w)
    sram_res=sram_placement(w,sram_budget_macros if sram_budget_macros is not None else sram["macro_count"])
    base={'schema':'cim-shared-prefix-source-bound-system/v1','model':Llama31().model_id,
          'coordinates':asdict(w),'geometry_source':LLAMA_CONFIG_SOURCE,
          'mapping_source':SINANGIL_MAPPING,'macro_source':SINANGIL_DESCRIPTOR,
          'macro_placement':asdict(p),'physical_work':n,'sram_storage':sram,
          'sram_placement':asdict(sram_res),
          'source_status':{'macro':'published_measured_macro','signed_mapping':'source_and_project_architecture',
                           'periphery':'unspecified' if rates is None else 'sensitivity_only',
                           'gpu':'unavailable' if gpu_points is None else 'external_stage_records'},
          'S0_GPU':{'tpot_s':None,'status':'unsupported_without_exact_GQA_GPU_stages'},
          'S1_SRAM':{'tpot_s':None,'status':'unsupported_without_native_Q4K4_GPU_score_and_SRAM_service'},
          'S2_CIM':{'tpot_s':None,'status':'unsupported_without_full_GPU_and_periphery_timing'},
          'S0_over_S2':None,'S1_over_S2':None,
          'evidence_class':'geometry_plus_published_macro_bound',
          'overlap':'serial_only'}
    if rates is not None:
        s2=s2_components(w,physical,rates)
        base['S2_CIM']['sensitivity_components_s']=s2
        base['S2_CIM']['sensitivity_periphery_and_macro_s']=sum(s2.values())
        base['S2_CIM']['sensitivity_bottleneck_excluding_GPU']=max(s2,key=s2.get)
    if gpu_points is None:
        base['missing_evidence']=['GQA-native per-output operator profiles', 'BF16-compatible full/suffix GPU scores',
                                   'signed INT4 GPU prefix score for S1', 'physically delivered SRAM/macro programming/link/reduction rates']
        return base
    if len(gpu_points)!=w.output_tokens:
        raise ValueError('one exact GPU profile per generated token required')
    for i,pt in enumerate(gpu_points):pt.validate(w,i)
    if any(pt.gpu_precision!=gpu_points[0].gpu_precision for pt in gpu_points):
        raise ValueError('mixed GPU precisions')
    if rates is None:
        base['missing_evidence']=['CIM/SRAM interconnect and programming/reduction rates']
        return base
    c2=s2_components(w,physical,rates)
    c2mean=sum(c2.values())
    # SRAM logical K4 is stored once per KV head/layer; GPU data-path loading
    # occurs for each token unless an exact caching schedule separately proves otherwise.
    sram_k_per_token=sram_res.physical_key_bytes_transferred_to_gpu_per_token
    sram_link=sram_k_per_token/rates.sram_to_gpu_Bps
    if rates.sram_write_Bps is not None:
        sram_initial=(sram_res.initial_macro_image_bytes/rates.prefix_link_Bps +
                      sram_res.initial_macro_image_bytes/rates.sram_write_Bps)/w.output_tokens
        sram_refill=(sram_res.recurring_refill_macro_image_bytes/rates.prefix_link_Bps +
                     sram_res.recurring_refill_macro_image_bytes/rates.sram_write_Bps)
    else:
        sram_initial=None;sram_refill=None
    result={k:[] for k in ('S0_GPU','S1_SRAM','S2_CIM')}
    for gpu in gpu_points:
        other=gpu.non_score_total_s(w.layers)
        if other is None:continue
        if gpu.score_full_s is not None:
            result['S0_GPU'].append(other+w.layers*gpu.score_full_s)
        if gpu.score_suffix_s is not None and gpu.score_prefix_int4_s is not None and sram_initial is not None:
            result['S1_SRAM'].append(other+w.layers*(gpu.score_suffix_s+gpu.score_prefix_int4_s)
                                     +sram_link+sram_initial+sram_refill + w.layers*(rates.quantization_s_per_layer+rates.sram_controller_s_per_layer))
        if gpu.score_suffix_s is not None:
            result['S2_CIM'].append(other+w.layers*gpu.score_suffix_s+c2mean)
    for key,values in result.items():
        if len(values)==w.output_tokens:
            base[key]['tpot_s']=sum(values)/len(values)
            base[key]['status']='conditional_model_only_unqualified_rate_sensitivity'
    if all(base[k]['tpot_s'] is not None for k in result):
        base['S0_over_S2']=base['S0_GPU']['tpot_s']/base['S2_CIM']['tpot_s']
        base['S1_over_S2']=base['S1_SRAM']['tpot_s']/base['S2_CIM']['tpot_s']
        base['evidence_class']='conditional_model_with_explicit_unqualified_periphery'
    base['missing_evidence']=sorted({name for pt in gpu_points for name in pt.missing})
    if rates.sram_write_Bps is None:base['missing_evidence'].append('physical SRAM-write service and explicit source')
    return base
