"""SCALE-Sim GEMM mapping and observed trace ingestion (never area inference)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .contracts import Workload
from .digital_threshold import EvidenceValue


def gemm_topology(workload: Workload) -> str:
    """One KV head at one decoder layer: Q[M,K] @ Kt[K,N].

    Repeat the simulation across workload.layers * workload.kv_heads. There is
    no implied common K reuse across replicas; SCALE-Sim's dataflow/trace and
    configured buffers determine actual reuse and stalls.
    """
    if workload.key_bits != 4:
        raise ValueError('mapping is for Q4xK4 only')
    m = workload.batch * workload.gqa_ratio
    n = workload.prefix_tokens
    k = workload.head_dim
    return f'Layer, M, N, K\nqk_per_kv_head, {m}, {n}, {k}\n'


@dataclass(frozen=True)
class ScaleSimObservation:
    cycles_per_kv_head_layer: int
    mapping_efficiency_pct: float
    total_cycles_serial_all_heads_layers: int
    report_source: str
    clock: EvidenceValue
    def __post_init__(self):
        if self.cycles_per_kv_head_layer <= 0 or self.clock.unit != 'seconds':
            raise ValueError('positive cycles and explicit clock period required')

    @property
    def serial_execution_seconds(self) -> float:
        return self.total_cycles_serial_all_heads_layers * self.clock.value


def read_scalesim_compute_report(path: str | Path, *, workload: Workload,
                                 clock_period: EvidenceValue) -> ScaleSimObservation:
    """Consume real SCALE-Sim COMPUTE_REPORT.csv (single KV-head QK GEMM).

    The reported cycles INCLUDE MAC execution. They must not be inserted as
    non-MAC time into the algebra, which would double-count arithmetic.
    """
    if clock_period.unit != 'seconds' or clock_period.value <= 0:
        raise ValueError('clock period must be positive seconds')
    p = Path(path)
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if len(rows) != 1:
        raise ValueError('expected one QK GEMM row per KV head and layer')
    row = {k.strip(): (v or '').strip() for k, v in rows[0].items() if k is not None}
    try:
        cyc = int(row['Total Cycles'])
        efficiency = float(row['Mapping Efficiency %'])
    except (KeyError, ValueError) as e:
        raise ValueError('invalid SCALE-Sim COMPUTE_REPORT fields') from e
    if cyc <= 0 or not 0 <= efficiency <= 100:
        raise ValueError('invalid compute cycles or efficiency')
    return ScaleSimObservation(cyc, efficiency, cyc * workload.kv_heads * workload.layers,
                               str(p), clock_period)
