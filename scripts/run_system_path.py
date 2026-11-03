#!/usr/bin/env python3
"""Run a source-manifest decoder scenario; never generate calibration constants.

python scripts/run_system_path.py --manifest scenario.json --output breakdown.json
Required manifest fields: workload, sram_bytes, tile_tokens, calibration,
and gpu_profile_files (one MAGMA native A100 JSON per output token).
"""
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cim_shared_prefix import Workload, SignedMapping, ScoreInterface
from cim_shared_prefix.system import (PathCalibration, OverlapEvidence, plan_residency,
                                      magma_a100_block, evaluate_decoder)


def run(manifest: Path) -> dict:
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    if spec.get('schema') != 'cim-shared-prefix-system-scenario/v1':
        raise ValueError('unsupported system manifest schema')
    w = Workload(**spec['workload'])
    mapping = SignedMapping(**spec.get('signed_mapping', {}))
    scores = ScoreInterface(**spec.get('scores', {}))
    calib = PathCalibration(**spec['calibration'])
    plan = plan_residency(w, spec['sram_bytes'], tile_tokens=spec['tile_tokens'], mapping=mapping,
                          source=spec.get('placement_source','pinned whole tiles + one scratch tile; code implementation'))
    sources = spec.get('gpu_profile_files')
    if not isinstance(sources,list) or len(sources) != w.output_tokens:
        raise ValueError('require one native MAGMA/LLMCompass GPU point for each generated token')
    blocks=[]
    for i, record in enumerate(sources):
        origin=(manifest.parent/record['path']).resolve()
        data=json.loads(origin.read_text(encoding='utf-8'))
        blocks.append(magma_a100_block(data,w,i,
            suffix_score_s=record.get('suffix_score_s'),
            suffix_source=record.get('suffix_source',''),
            structural_s_per_token=record.get('structural_s_per_token'),
            structural_source=record.get('structural_source','')))
    overlap = OverlapEvidence(**spec['overlap']) if 'overlap' in spec else None
    r=evaluate_decoder(w,plan,calib,tuple(blocks),mapping=mapping,scores=scores,overlap=overlap)
    return {'schema':'cim-shared-prefix-system-result/v1',
            'workload':asdict(w), 'signed_mapping':asdict(mapping), 'scores':asdict(scores),
            'placement':{**asdict(r.residence), 'resident_bytes':r.residence.resident_bytes,
                         'streamed_bytes_per_token':r.residence.streamed_bytes_per_token},
            'mean_tpot_s':r.scheduled_mean_tpot_s,
            'mean_serial_tpot_s':r.serial_mean_tpot_s,
            'total_decode_s':r.total_decode_s, 'first_token_s':r.first_token_s,
            'batch_tokens_per_second':w.batch/r.scheduled_mean_tpot_s,
            'schedule':r.timing_policy, 'components_mean_s':r.aggregate_components_s,
            'tokens':[dict(asdict(t),serial_s=t.serial_s,scheduled_s=t.scheduled_s) for t in r.tokens],
            'sources':list(r.sources),
            'caveats': ['LLMCompass A100 records are vendor-modeled FP16/HBM2e, not a measured GPU trace',
                        'CIM numeric values require independently qualified hardware timing and programming sources',
                        'Structural GPU operations must be separately sourced; source labels are not independent qualification']}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=run(args.manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'mean_tpot_s':output['mean_tpot_s'],
                      'schedule':output['schedule']}))

if __name__=='__main__': main()
