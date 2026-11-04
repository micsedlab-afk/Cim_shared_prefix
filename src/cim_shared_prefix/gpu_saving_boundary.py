"""GPU attention saving required to amortize the published-macro CIM path.

All values are derived algebraically from an already executed 7nm
source-calibrated conditional serial schedule. No GPU hardware numbers
are inferred or substituted. The comparison is between full native GPU
attention and the GPU attention remainder after completed prefix scores.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

NATIVE_LAYERS = {
    "meta-llama/Llama-3.1-8B": 32,
    "mistralai/Mistral-7B-v0.1": 32,
    "facebook/opt-2.7b": 32,
}


def _float(row: Mapping[str, object], key: str) -> float:
    try:
        x = float(row[key])
    except (KeyError, ValueError, TypeError) as err:
        raise ValueError("missing or invalid architectural input " + key) from err
    if not isfinite(x):
        raise ValueError("nonfinite architectural input " + key)
    return x


@dataclass(frozen=True)
class SavingThreshold:
    model_id: str
    layers: int
    modeled_cim_prefix_s: float
    source_macro_floor_s: float
    required_gpu_attention_saving_us_per_layer: float
    optimistic_floor_us_per_layer: float
    score_delivery_us_per_layer: float
    source_plus_program_us_per_layer: float
    serial_schedule_headroom_ratio: float
    status: str = "algebraic_A_minus_remainder_threshold; external macro rates unqualified"

    def compare_gpu(self, mean_native_full_attention_s: float,
                    mean_completed_score_gpu_remainder_s: float,
                    *, gpu_source: str) -> str:
        """Match single-layer GPU attention times to 32-layer average CIM cost.

        If GPU saving exceeds the serial CIM threshold, the *specified*
        no-overlap S2 policy wins on its isolated attention endpoint.
        This cannot establish a silicon-implemented accelerator.
        """
        if not gpu_source.strip():
            raise ValueError("provenance of GPU timings required")
        if not all(isinstance(v, (float, int)) and
                   not isinstance(v, bool) and isfinite(v) and v >= 0 for v in
                   (mean_native_full_attention_s, mean_completed_score_gpu_remainder_s)):
            raise ValueError("nonnegative finite same-endpoint GPU timing required")
        actual_saving_us = (mean_native_full_attention_s -
                            mean_completed_score_gpu_remainder_s) * 1e6
        return ("CONDITIONALLY_SERIAL_CIM_FASTER" if
                actual_saving_us > self.required_gpu_attention_saving_us_per_layer
                else "SERIAL_CIM_NOT_FASTER")


def from_integrated_architecture_row(row: Mapping[str, object]) -> SavingThreshold:
    model = str(row.get("model", ""))
    if model not in NATIVE_LAYERS:
        raise ValueError("unknown native model geometry")
    layers = NATIVE_LAYERS[model]
    if str(row.get("evidence", "")) != (
        "published_complete_macro_plus_conditional_serial_rate_sensitivity"
    ):
        raise ValueError("not a source-calibrated macro sensitivity record")
    serial = _float(row, "serial_completed_prefix_scores_s")
    floor = _float(row, "optimistic_resource_lower_bound_s")
    score = _float(row, "stage_completed_score_delivery_s")
    key_supply = _float(row, "stage_source_key_supply_s")
    programming = _float(row, "stage_macro_programming_s")
    if not (serial > 0 and 0 <= floor <= serial * (1 + 1e-11)):
        raise ValueError("invalid necessary/serial latency ordering")
    if min(score, key_supply, programming) < 0:
        raise ValueError("negative causal stage")
    return SavingThreshold(
        model_id=model, layers=layers,
        modeled_cim_prefix_s=serial, source_macro_floor_s=floor,
        required_gpu_attention_saving_us_per_layer=1e6 * serial / layers,
        optimistic_floor_us_per_layer=1e6 * floor / layers,
        score_delivery_us_per_layer=1e6 * score / layers,
        source_plus_program_us_per_layer=1e6 * (key_supply + programming) / layers,
        serial_schedule_headroom_ratio=serial / floor if floor else float("inf"),
    )
