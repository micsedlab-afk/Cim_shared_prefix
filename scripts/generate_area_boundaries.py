"""Produce reproducible parameter *boundaries*, not fabricated digital measurements.

Usage: PYTHONPATH=src python scripts/generate_area_boundaries.py [output_dir]
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.area_optimizer import (
    CIM_7NM, SRAM_7NM, SupportArea, compare_physical_areas, cim_native_floor)
from cim_shared_prefix.digital_architecture import (
    DigitalSchedule, RateEvidence, architecture_boundary)
from cim_shared_prefix.digital_threshold import EvidenceValue


def sensitivity_area(value: float) -> EvidenceValue:
    return EvidenceValue(value, 'mm2', 'explicit zero-to-high external area sensitivity, not a layout', 'sensitivity')


def support(total: float) -> SupportArea:
    # Identical component names are mandatory. Controller/reduction/link etc
    # are all charged outside each complete physical macro.
    # Equal splitting is ONLY a sensitivity allocation; no physical estimate.
    pieces = [total / 6] * 6
    return SupportArea(*(sensitivity_area(p) for p in pieces))


def rate(value: float, unit: str) -> RateEvidence:
    return RateEvidence(value, unit, 'explicit bandwidth/throughput sensitivity, not fabricated measurement', 'sensitivity')


def produce(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    fields = ('batch', 'prefix_tokens', 'cim_copies', 'sram_copies', 'cim_support_mm2',
              'digital_support_mm2', 'sram_bandwidth_TBps', 'score_link_TBps',
              'key_read_reuse', 'reduction_tile_k', 'cim_target_multiple_of_link_floor',
              'cim_macro_count', 'sram_macro_count', 'cim_macro_area_mm2',
              'sram_macro_area_mm2', 'cim_area_budget_mm2', 'digital_core_area_mm2',
              'cim_completed_target_s', 'digital_nonmac_time_s',
              'score_bytes', 'key_read_bytes', 'signed_cim_evaluations',
              'status', 'required_MAC_per_s_per_mm2',
              'min_sram_Bps', 'min_score_link_Bps', 'min_reduction_adds_s',
              'physical_digital_isoarea_qualified')
    records = []
    for prefix, batch, ccopy, dcopy, cext, dext, bw, reuse_kind, tile, factor in product(
        (512, 2048, 8192), (1, 8), (1, 2), (1, 2), (0.0, 100.0),
        (0.0, 200.0), (1e12, 8e12, 32e12), (1, 4), (64, 128), (2, 4)):
        w = Workload(batch, prefix, 0, 32, 8, 128, 32, 128, key_bits=4)
        area = compare_physical_areas(w, cim_external=support(cext),
                                      digital_external=support(dext),
                                      cim_copies=ccopy, digital_copies=dcopy)
        reuse = 1 if reuse_kind == 1 else batch * w.gqa_ratio
        link = rate(1e12, 'bytes/s')
        floor = cim_native_floor(w, area.cim)
        # The nominal 5.5 ns access is macro-only; score-link time is
        # imposed on the same completed-score endpoint. No actual CIM
        # system latency has been provided. Multiplier is sensitivity.
        reference_floor = max(floor.best_case_macro_compute_floor_s,
                              floor.output_score_bytes_per_token / link.value)
        target = EvidenceValue(factor * reference_floor, 'seconds',
                               'sensitivity multiple of optimistic CIM compute/score-link floor',
                               'sensitivity')
        schedule = DigitalSchedule(
            key_read_reuse=reuse, reduction_tile_k=tile,
            sram_aggregate_bandwidth=rate(bw, 'bytes/s'),
            score_link_bandwidth=link,
            reduction_throughput=rate(1e12, 'adds/s'),
            controller_latency=EvidenceValue(0.0, 'seconds',
                'optimistic controller latency lower bound, sensitivity', 'sensitivity'),
            overlap='serial')
        result = architecture_boundary(w, area, cim_completed_score=target, schedule=schedule)
        records.append(dict(zip(fields, (
            batch, prefix, ccopy, dcopy, cext, dext, bw / 1e12, 1.0,
            reuse, tile, factor,
            area.cim.macro_count, area.digital.macro_count,
            area.cim.area_mm2, area.digital.area_mm2,
            area.cim_total_mm2, result.requirement.core_area_available_mm2,
            target.value, result.nonmac.nonmac_total_s,
            result.traffic.completed_score_bytes, result.traffic.key_sram_read_bytes,
            floor.native_unsigned_mav_per_token,
            result.status, result.requirement.required_mac_per_second_per_mm2,
            result.minimum_sram_bandwidth_bytes_s,
            result.minimum_score_link_bandwidth_bytes_s,
            result.minimum_reduction_adds_s, False
        ))))
    with (output / 'digital_boundary_sensitivity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    statuses = {status: sum(r['status'] == status for r in records)
                for status in sorted({r['status'] for r in records})}
    summary = {
        'scope': 'conditional parameter sensitivity, not physical digital iso-area or measured latency',
        'technology_nm': 7,
        'storage_sources': {
            'SRAM': {'doi': SRAM_7NM.source, 'macro_capacity_bits': SRAM_7NM.capacity_bits,
                     'macro_area_mm2': SRAM_7NM.area_mm2,
                     'area_basis': 'published 29.2 Mb/mm2 complete macro density, 512x512'},
            'CIM': {'doi': CIM_7NM.source, 'macro_capacity_bits': CIM_7NM.capacity_bits,
                    'macro_area_mm2': CIM_7NM.area_mm2, 'signed_storage_factor': 2,
                    'native_evaluations_per_signed_Q4xK4': 4,
                    'published_4b4b_GOPS': 372.4, 'ops_per_native_MAV': 2}},
        'rows': len(records), 'status_counts': statuses,
        'bounds': {'CIM_support_mm2': [0, 100], 'digital_support_mm2': [0, 200],
                   'SRAM_aggregate_Bps': [1e12, 8e12, 32e12],
                   'score_link_Bps': 1e12, 'reduction_adds_per_s': 1e12,
                   'target_time': '2x or 4x max(CIM macro-only compute floor, score-link transfer floor)',
                   'overlap_policy': 'serial; controller latency=0 optimistic lower bound'},
        'excluded': ['physical digital MAC area', '7nm synthesis',
                     'unverified SCALE-Sim measurement', 'chip-area per MAC division',
                     'actual completed-score CIM system measurement'],
    }
    (output / 'physical_area_provenance.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('area_outputs')
    print(json.dumps(produce(out), indent=2))
