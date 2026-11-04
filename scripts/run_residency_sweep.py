"""Deterministic physical-area / conditional digital break-even sweep.

Usage: PYTHONPATH=src python scripts/run_residency_sweep.py [output_directory]
Everything outside the complete published SRAM/CIM macro is sensitivity-only.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict
from itertools import product
from math import isfinite
from pathlib import Path
import sys

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.area_optimizer import CIM_7NM, SRAM_7NM
from cim_shared_prefix.residency_area import (
    ExternalArea, RateSet, layer_layout, accounting, residency_plan,
    enumerate_plans, best_cim, best_digital_requirement, boundary,
)


def support(total: float) -> ExternalArea:
    # Explicit per-component decomposition of a sensitivity. The division is
    # not a physical estimate, nor is it allowed to qualify digital iso-area.
    return ExternalArea(*(total / 6 for _ in range(6)))


def clean(x):
    """Stable CSV/strict JSON representation of unattainable conditions."""
    if isinstance(x, float) and not isfinite(x):
        return 'unattainable'
    return x


def build_row(w, budget, support_area, prog_rate, out_rate, sram_rate, reuse):
    ca = support(support_area)
    da = support(support_area)
    rates = RateSet(prog_rate, sram_rate, out_rate, out_rate, 1e12)
    geom = layer_layout(w, kind='cim')
    srgeom = layer_layout(w, kind='sram')
    a = accounting(w, digital_reuse=reuse)
    b = best_cim(w, budget_mm2=budget, support=ca, rates=rates)
    # Every row records same area budget and interface policy, independently
    # optimized layer-pinning. Digital core area remains physically UNKNOWN.
    fixed_deadline_s = 2e-3  # explicit architectural target sensitivity
    base = {
        'budget_mm2': budget, 'prefix_tokens': w.prefix_tokens, 'batch': w.batch,
        'layers': w.layers, 'output_tokens': w.output_tokens,
        'support_each_path_mm2': support_area, 'programming_Bps_sensitivity': prog_rate,
        'score_output_Bps_sensitivity': out_rate, 'sram_Bps_sensitivity': sram_rate,
        'key_read_reuse': reuse, 'deadline_s_sensitivity': fixed_deadline_s,
        'cim_all_layer_macro_mm2': geom.all_layer_macro_mm2,
        'cim_one_layer_macro_mm2': geom.full_layer_macro_mm2,
        'sram_all_layer_macro_mm2': srgeom.all_layer_macro_mm2,
        'sram_one_layer_macro_mm2': srgeom.full_layer_macro_mm2,
        'native_unsigned_work_per_token': a.native_unsigned_mac_per_token,
        'native_padded_work_per_token': a.native_issued_padded_mac_per_token,
        'native_occupancy_fraction': a.mapping_occupancy,
        'signed_partials_per_score': a.signed_partial_values_per_score,
        'cim_reduction_adds_per_token': a.reduction_additions_per_token,
        'output_bytes_per_token': a.complete_score_bytes_per_token,
        'digital_key_sram_read_bytes_per_token': a.key_read_bytes_per_token,
        'qualified_physical_digital_isoarea': False,
    }
    if b is None:
        base.update(cim_status='area_infeasible', cim_pinned_layers=None,
                    cim_instantiated_macro_mm2=None, cim_stream_waves=None,
                    cim_computed_time_s=None, cim_program_Bps_min_to_2ms=None,
                    cim_output_Bps_min_to_2ms=None,
                    digital_rho_to_beat_CIM=None)
    else:
        cp, stages = b
        bounds = boundary(w, cp, rates, target_s=fixed_deadline_s)
        base.update(cim_status=bounds.status, cim_pinned_layers=cp.pinned_layers,
                    cim_instantiated_macro_mm2=cp.physical_macro_mm2,
                    cim_instantiated_macro_count=cp.instantiated_macros,
                    cim_stream_waves=cp.stream_waves_per_unpinned_layer,
                    cim_full_active_scratch=cp.full_active_layer_in_scratch,
                    cim_programming_B_per_token=cp.programming_bytes_per_generated_token,
                    cim_native_compute_floor_s=stages.compute_s,
                    cim_computed_time_s=stages.serial_completed_score_s,
                    cim_program_Bps_min_to_2ms=clean(bounds.minimum_program_Bps),
                    cim_output_Bps_min_to_2ms=clean(bounds.minimum_output_Bps),
                    cim_signed_useful_GMAC_per_s_floor=a.logical_signed_mac_per_token/stages.compute_s/1e9,
                    cim_signed_useful_GMAC_per_s_per_macro_mm2_floor=(
                        a.logical_signed_mac_per_token/stages.compute_s/cp.physical_macro_mm2/1e9),
                    )
    # Digital conditional density versus an external 2ms deadline.
    db = best_digital_requirement(w, budget_mm2=budget, support=da, rates=rates,
                                  target_s=fixed_deadline_s, reuse=reuse)
    if db is None:
        base.update(digital_status='area_infeasible')
    else:
        dp, d = db
        base.update(digital_status=d.status, digital_pinned_layers=dp.pinned_layers,
                    digital_instantiated_macro_mm2=dp.physical_macro_mm2,
                    digital_instantiated_macro_count=dp.instantiated_macros,
                    digital_stream_waves=dp.stream_waves_per_unpinned_layer,
                    digital_full_active_scratch=dp.full_active_layer_in_scratch,
                    digital_programming_B_per_token=dp.programming_bytes_per_generated_token,
                    digital_MAC_core_area_budget_mm2=d.mac_core_area_mm2,
                    digital_nonMAC_s=d.nonmac_s,
                    digital_rho_to_2ms_MACps_per_mm2=clean(d.required_mac_per_s_per_mm2),
                    digital_prog_Bps_min_to_2ms=clean(d.minimum_program_Bps),
                    digital_sram_Bps_min_to_2ms=clean(d.minimum_sram_Bps),
                    digital_output_Bps_min_to_2ms=clean(d.minimum_output_Bps),
                    )
    if b is not None:
        comparable = best_digital_requirement(w, budget_mm2=budget, support=da,
                    rates=rates, target_s=stages.serial_completed_score_s, reuse=reuse)
        if comparable is not None:
            q, dcmp = comparable
            base['digital_rho_to_beat_CIM_MACps_per_mm2'] = clean(dcmp.required_mac_per_s_per_mm2)
            base['digital_competition_status'] = dcmp.status
            base['digital_competition_pinned_layers'] = q.pinned_layers
            base['cim_time_target_is_modeled_sensitivity'] = True
    return base


def produce(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    # 1..16 mm2 includes physical macro-only feasibility and explicit external
    # area, programming, SRAM and completed-score link sensitivities.
    for budget, prefix, batch, length, overhead, program, out, read in product(
        range(1,17), (512,2048,8192), (1,8), (32,128), (0.0,0.5),
        (0.1e12,1e12), (0.25e12,1e12), (0.25e12,1e12)):
        w=Workload(batch, prefix, 0, 32, 8, 128, 32, length, key_bits=4)
        # Both minimum and maximal logical reuse are mapped; the full-reuse
        # assumption needs actual SCALE-Sim scheduling verification.
        for reuse in (1, batch*w.gqa_ratio):
            rows.append(build_row(w,budget,overhead,program,out,read,reuse))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    outcsv=output/'area_budget_residency_1_16mm2.csv'
    with outcsv.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)

    # Compact 48-row reference cross-section for review and Git diffability.
    reference=[row for row in rows if row['batch']==1 and row['output_tokens']==128
               and row['support_each_path_mm2']==0.5
               and row['programming_Bps_sensitivity']==1e12
               and row['score_output_Bps_sensitivity']==1e12
               and row['sram_Bps_sensitivity']==1e12
               and row['key_read_reuse']==4]
    assert len(reference)==48
    with (output/'review_48_budget_points.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(reference)

    # Discrete architectural boundaries with both valid-bit and whole-macro
    # programming to expose padding/porting penalties for irregular shapes.
    edge=[]
    for prefix, dim, budget in product((513,2048), (96,128), (1,4,8,16)):
        w=Workload(1,prefix,0,32,8,dim,32,128,key_bits=4)
        for kind in ('cim','sram'):
            plans=enumerate_plans(w,kind=kind,budget_mm2=budget,support=support(0.0))
            for mode in ('valid_bits','full_macros'):
                p=residency_plan(w,kind=kind,budget_mm2=budget,
                                 support=support(0.0),pinned_layers=0,
                                 programming_policy=mode)
                edge.append(dict(prefix_tokens=prefix,head_dim=dim,budget_mm2=budget,
                                 kind=kind,programming_policy=mode,
                                 all_layer_area_mm2=layer_layout(w,kind=kind).all_layer_macro_mm2,
                                 full_layer_area_mm2=layer_layout(w,kind=kind).full_layer_macro_mm2,
                                 number_feasible_residency_policies=len(plans),
                                 instantiated_macros=p.instantiated_macros if p else None,
                                 streaming_waves=p.stream_waves_per_unpinned_layer if p else None,
                                 programming_B_per_token=p.programming_bytes_per_generated_token if p else None,
                                 native_CIM_occupancy=layer_layout(w,kind='cim').useful_64x16_occupancy))
    with (output/'irregular_tile_occupancy.csv').open('w',newline='') as f:
        d=csv.DictWriter(f,fieldnames=edge[0].keys(),lineterminator='\n');d.writeheader();d.writerows(edge)
    # Deterministic categorical summary with no invented measured values.
    status={name:sum(r.get('cim_status')==name for r in rows)
            for name in sorted(set(r.get('cim_status') for r in rows))}
    dstatus={name:sum(r.get('digital_status')==name for r in rows)
             for name in sorted(set(r.get('digital_status') for r in rows))}
    sample = Workload(1,2048,0,32,8,128,32,128,key_bits=4)
    metadata = {
        'model': 'analytical macro-residency + serial physical traffic lower bounds; conditional digital break-even',
        'technology': 'published 7nm SRAM/CIM macros; external integration unsupported',
        'source_anchors': {
            'cim': {'name':'Dong et al., ISSCC 2020; Sinangil et al., JSSC 2021',
                    'url':CIM_7NM.source, 'jssc_doi':'https://doi.org/10.1109/JSSC.2020.3031290',
                    'physical_macro_mm2':0.0032,'macro_bits':4096,'unsigned_4b4b_mav_per_cycle':1024,
                    'access_seconds':5.5e-9,'headline_GOPS':372.4,'ops_count_per_native_mav':2,
                    'signed_key_planes':2,'native_evaluations_per_signed_Q4K4':4,
                    'output_precision':'macro 4-bit ADC/average; 32-bit completed-score postprocess UNCHARACTERIZED'},
            'sram': {'name':'Yokoyama et al., VLSI 2020','url':SRAM_7NM.source,
                     'macro_bits':262144,'macro_Mbit_per_mm2':29.2,
                     'macro_area_mm2':SRAM_7NM.area_mm2,'read_write_porting':'NOT qualified for required bandwidth'},
        },
        'key_mapping':'CIM 64 reduction rows x 16 4b score columns; 2 banks per signed tile',
        'area_scope':'physical macro count once, 6 external-area components as explicit sensitivities; GPU die excluded',
        'time_scope':'serial layer visits and stages; no assumed interstage overlap; initial programming amortized over output_tokens',
        'first_token':'cold', 'programming_mode':'valid_bits optimistic; whole_macro irregular sensitivity',
        'full_model_instantiated_macro_footprints_mm2':{
            'CIM_P2048':layer_layout(sample,kind='cim').all_layer_macro_mm2,
            'CIM_active_P2048':layer_layout(sample,kind='cim').full_layer_macro_mm2,
            'SRAM_P2048':layer_layout(sample,kind='sram').all_layer_macro_mm2,
            'SRAM_active_P2048':layer_layout(sample,kind='sram').full_layer_macro_mm2,
        },
        'sweep':{'rows':len(rows),'budgets_mm2':list(range(1,17)),
                 'prefix_tokens':[512,2048,8192],'batch':[1,8],'output_tokens':[32,128],
                 'support_each_path_mm2':[0,0.5],
                 'programming_Bps':[0.1e12,1e12], 'output_Bps':[0.25e12,1e12],
                 'SRAM_Bps':[0.25e12,1e12], 'query_Bps':'same value as output_Bps',
                 'reduction_adds_per_s':1e12,'controller_s':0,
                 'digital_reuse':'1 or batch*GQA; unqualified mapping extremes',
                 'target_s_sensitivity':2e-3,'CIM_statuses':status,'digital_statuses':dstatus,
                 'physical_digital_isoarea_count':0},
        'qualification':{
            'physical_macro_area':'published complete macro; yes',
            'complete_CIM_signed_Q4K4_layout':'no',
            'actual_7nm_digital_MAC_physical_area':'absent',
            'completed_score_accuracy':'not established by 4-bit CIM ADC',
            'real_MAC_speedup':'not claimed',
            'qualified_digital_isoarea':'not available',
            'SCALE-Sim_executed':'not executed; package unavailable' if importlib.util.find_spec('scalesim') is None else 'package detected; no comparable config/report executed',
        },
    }
    (output/'residency_area_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')

    # Publication-ready sensitivity figures; these reflect the analytical
    # source macro floor and explicit user-controllable rates only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=output/'figures';figdir.mkdir(exist_ok=True)
    budgets=list(range(1,17))
    w=Workload(1,2048,0,32,8,128,32,128,key_bits=4)
    fig, ax=plt.subplots(figsize=(8,4.6))
    for ov in (0.0,0.5):
        p=[];wave=[]
        for b in budgets:
            cc=best_cim(w,budget_mm2=b,support=support(ov),
                    rates=RateSet(1e12,1e12,1e12,1e12,1e12))
            p.append(cc[0].pinned_layers if cc else float('nan'))
            wave.append(cc[0].stream_waves_per_unpinned_layer if cc else float('nan'))
        ax.plot(budgets,wave,marker='o',label=f'External area={ov:g} mm²')
    ax.axvline(layer_layout(w,kind='cim').full_layer_macro_mm2,linestyle='--',linewidth=1,
               label='Physical one-layer macro threshold')
    ax.set(xlabel='Total incremental area budget (mm²)',ylabel='Streaming tile waves per layer',
           title='CIM P=2048; 32 layers; B=1; two signed K planes')
    ax.set_xticks(budgets);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'streaming_waves_vs_area.png',dpi=180);fig.savefig(figdir/'streaming_waves_vs_area.svg');plt.close(fig)

    fig, ax=plt.subplots(figsize=(8,4.6))
    # One selected scenario: deadline 2ms, all rates held at 1 TB/s except varied
    # programming throughput. Physical area and source native latency unchanged.
    for prog in (0.1e12, 1e12):
        times=[]
        for b in budgets:
            bb=best_cim(w,budget_mm2=b,support=support(0.5),
                        rates=RateSet(prog,1e12,1e12,1e12,1e12))
            times.append(bb[1].serial_completed_score_s*1e3 if bb else float('nan'))
        ax.plot(budgets,times,marker='o',label=f'Programming={prog/1e12:g} TB/s')
    ax.set(xlabel='Total incremental area budget (mm²)',ylabel='Optimistic serial score time (ms/token)',
           title='CIM model sensitivity; P=2048, B=1, 128 output tokens')
    ax.set_xticks(budgets);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'cim_latency_vs_programming_bw.png',dpi=180);fig.savefig(figdir/'cim_latency_vs_programming_bw.svg');plt.close(fig)

    fig, ax=plt.subplots(figsize=(8,4.6))
    for out_rate in (0.25e12,1e12):
        dens=[]
        rates=RateSet(1e12,1e12,out_rate,out_rate,1e12)
        for b in budgets:
            entry=best_digital_requirement(w,budget_mm2=b,support=support(0.5),
                      rates=rates,target_s=2e-3,reuse=w.gqa_ratio)
            val=entry[1].required_mac_per_s_per_mm2 if entry else None
            dens.append(val/1e12 if val is not None else float('nan'))
        ax.plot(budgets,dens,marker='o',label=f'Score link={out_rate/1e12:g} TB/s')
    ax.set(xlabel='Total incremental area budget (mm²)',
           ylabel='Required digital signed TMAC/s/mm² (conditional)',
           title='Digital Q4×K4 physical-core area requirement; 2ms sensitivity')
    ax.set_xticks(budgets);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'digital_density_vs_area_bw.png',dpi=180);fig.savefig(figdir/'digital_density_vs_area_bw.svg');plt.close(fig)
    return metadata


if __name__=='__main__':
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'results/area_final')
    print(json.dumps(produce(output),indent=2))