#!/usr/bin/env python3
"""Generate source-geometry S0/S1/S2 evidence and periphery sensitivities.

The source QK study fixes Lp=(1024,2048,4096,8192,16384) and
Lu=(128,256). Batch/output/sizing/BW are *added* serving sensitivities.
There are no invented GPU stage times; complete TPOT/speedup remain NULL.
"""
from __future__ import annotations
import argparse,csv,json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.source_grounded import PhysicalCIM, RateSensitivity,compare,placement,counts,LLAMA_CONFIG_SOURCE,SINANGIL_DESCRIPTOR,SRAM_AREA_SOURCE

PREFIXES=(1024,2048,4096,8192,16384)
SUFFIXES=(128,256)
BATCHES=(1,2,4,8)
OUTPUTS=(1,32,128)
RATE_BW=(1e11,1e12) # declared sensitivity axes only, not measured transfer rates

def sensitivity(bw):
    return RateSensitivity(prefix_link_Bps=bw,sram_to_gpu_Bps=bw,query_link_Bps=bw,
        score_link_Bps=bw,program_Bps=bw,reduction_adds_per_s=1e12,
        partial_scale_mults_per_s=1e12,quantization_s_per_layer=0,
        provenance={key:'unqualified rate sensitivity; no hardware evidence' for key in (
            'prefix_link','sram_to_gpu','query_link','score_link','program','reduction_adds','partial_scales','quantization','cim_controller_s_per_layer','sram_controller_s_per_layer')},
        sram_write_Bps=bw,sram_write_provenance='unqualified SRAM write rate sensitivity; no measured write service')

def sweep()->tuple[list[dict[str,object]],dict[str,object]]:
    rows=[]
    original=[]
    for P in PREFIXES:
        for U in SUFFIXES:
            for b in BATCHES:
                for o in OUTPUTS:
                    w=Workload(b,P,U,32,8,128,32,o,key_bits=4)
                    demand=placement(w,PhysicalCIM(1,1)).macros_required
                    for mode,capacity in (('cap_4096',4096),('cap_32768',32768),('full_residency',demand)):
                        for bw in RATE_BW:
                            phy=PhysicalCIM(capacity,min(256,capacity),source='sensitivity macro capacity; no external integrated area proof')
                            data=compare(w,phy,rates=sensitivity(bw))
                            c=data['physical_work'];pl=data['macro_placement'];s2=data['S2_CIM']
                            stage=s2['sensitivity_components_s']
                            row={
                                'Lp':P,'Lu':U,'B_sensitivity':b,'O_sensitivity':o,'macro_budget_policy':mode,
                                'macro_count':capacity,'active_macro_ports_assumed':phy.active_macro_ports,
                                'bandwidth_Bps_sensitivity':bw,
                                'physical_macro_count_full_residency':pl['macros_required'],
                                'physical_macro_area_full_residency_mm2':pl['complete_macro_area_required_mm2'],
                                'provisioned_macro_area_mm2':pl['provisioned_macro_area_mm2'],
                                'pinned_macros':sum(pl['pinned_per_layer']),
                                'streamed_macros_per_token':sum(pl['streamed_per_layer']),
                                'native_macro_calls_per_token':c['native_macro_calls_per_token'],
                                'macro_native_floor_s':pl['native_access_floor_s_per_token'],
                                'macro_scratch_serial_access_s':pl['native_single_scratch_schedule_s_per_token'],
                                'score_bytes_per_token':c['complete_score_bytes_per_token'],
                                'signed_reductions_per_token':c['digital_signed_combines_per_token'],
                                'scale_multiplies_per_token':c['digital_partial_rescale_multiplies_per_token'],
                                'sram_only_storage_area_mm2':data['sram_storage']['total_macro_area_mm2'],
                                **{name:v for name,v in stage.items()},
                                'S2_conditional_nonGPU_s':s2['sensitivity_periphery_and_macro_s'],
                                'nonGPU_bottleneck':s2['sensitivity_bottleneck_excluding_GPU'],
                                'S0_full_TPOT_s':None,'S1_full_TPOT_s':None,'S2_full_TPOT_s':None,
                                'S0_S2_speedup':None,'S1_S2_speedup':None,
                                'result_status':'INCOMPLETE: no exact full Llama GQA GPU and INT4 score operator evidence',
                                'evidence_class':'published_macro_floor_plus_unqualified_sensitivity'
                            }
                            rows.append(row)
            # Original (Lp,Lu) only, fixed B=1/O=1 for shape reporting, no latency claim.
            w0=Workload(1,P,U,32,8,128,32,1)
            c0=counts(w0,PhysicalCIM(placement(w0,PhysicalCIM(1,1)).macros_required,256))
            original.append({'Lp':P,'Lu':U,'score_transfer_bytes_per_token':c0['complete_score_bytes_per_token'],
                             'native_macro_calls_per_token':c0['native_macro_calls_per_token'],
                             'full_residency_macro_area_mm2':c0['macro_area_mm2_full_residency'],
                             'source':LLAMA_CONFIG_SOURCE})
    return rows,{'schema':'cim-shared-prefix-architecture-sweep/v1','original_axes':{'prefix_tokens':PREFIXES,'suffix_tokens':SUFFIXES},
          'source_axes':'original QK numerical intervention Lp/Lu, not original decode serving measurements',
          'extended_serving_sensitivity_axes':{'B':BATCHES,'O':OUTPUTS,'macro_budget':('4096','32768','full_residency'),
                                                'link_and_program_Bps':RATE_BW,'active_ports_assumed':256},
          'coordinate_count':len(rows),'original_coordinate_count':len(original),'original_coordinates':original,
          'physical_macro':SINANGIL_DESCRIPTOR,'sram_macro':SRAM_AREA_SOURCE,
          'missing_model_sources':['fully geometry-matched BF16/H100 native decoder GPU stage data',
                                   'signed Q4K4 GPU prefix score timing for S1',
                                   'SRAM bank port/links/controller physical traces',
                                   'CIM signed accumulators and external programming/source-verified performance',
                                   'GPU RoPE/SwiGLU/RMSNorm/LM-head/sampling exact stage profiles'],
          'native_macro_ideal_reference_executed':'results/native_sinangil_arithmetic_receipt.json',
          'qualified_full_tpot_coordinates':0,'qualified_speedup_coordinates':0,
          'all_synthetic_periphery_results_are_sensitivity_only':True,
          'overlap_policy':'strictly serial; no overlap credited'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows,manifest=sweep()
    f=a.output_dir/'source_grounded_sensitivity.csv'
    with f.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output_dir/'source_grounded_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'coordinates':len(rows),'original':manifest['original_coordinate_count'],
                      'full_tpot_qualified':manifest['qualified_full_tpot_coordinates'],
                      'output_csv':str(f),'output_json':str(a.output_dir/'source_grounded_manifest.json')}))
