#!/usr/bin/env python3
"""Reproducible 1--16 mm2 physically *conditional* integrated architecture study.

All numbers are algebra or a constructive serial schedule using published
complete 7nm macro areas and native access time. All added rates are explicit
unqualified sensitivities. GPU TPOT and speedups remain null.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.area_optimizer import CIM_7NM, SRAM_7NM
from cim_shared_prefix.profiles import PROFILES, visible_prefix
from cim_shared_prefix.residency_area import (
    ExternalArea, RateSet, residency_plan, best_digital_requirement,
)
from cim_shared_prefix.integrated_schedule import SerialRates, serial_schedule, best_serial
from cim_shared_prefix.feasibility import PhysicalBudget, lower_bound


DEADLINE_S = 0.002  # Hypothetical service deadline, NOT a measured GPU baseline.
PORT_CAP = 256
BUDGETS = tuple(range(1, 17))
BANDWIDTHS = (1e11, 1e12)
SUPPORTS = (0.0, 0.5)
BATCHES = (1, 8)
OUTPUTS = (1, 128)


def support(mm2):
    # Six equally charged sensitivity slots; zero does NOT imply zero real area.
    return ExternalArea(*(mm2/6 for _ in range(6)),
                        source="equal-split external-area SENSITIVITY, unqualified")


def rates(bw):
    return SerialRates(
        prefix_supply_Bps=bw, programming_Bps=bw, query_Bps=bw,
        completed_score_Bps=bw, signed_reduction_adds_per_s=1e12,
        partial_scalings_per_s=1e12, controller_s_per_layer=0.0,
        source=f"unqualified common {bw:g} B/s interface sensitivity")


def required_rate(bytes_or_ops, target, other):
    if target <= other:
        return None
    return bytes_or_ops/(target-other)


def finite(x):
    return x if x is None or (isinstance(x, (int, float)) and math.isfinite(x)) else None


def inspect_coordinate(profile, *, prefix, suffix, batch, output, budget, supp, bw):
    if prefix + suffix + output > profile.context_limit:
        raise ValueError("source model context violated")
    srates = rates(bw)
    area = support(supp)
    visible = [visible_prefix(profile, prefix, suffix, t) for t in range(output)]
    if not all(p > 0 for p in visible):
        raise ValueError("zero-visible-prefix requires GPU-only suffix stage")
    sliding = len(set(visible)) > 1

    def at(p, o):
        return Workload(batch, p, suffix, profile.query_heads, profile.kv_heads,
                        profile.head_dim, profile.layers, o, key_bits=4)

    # Fixed prefix: one causal multi-token schedule, cold once + warm refills.
    # SWA: every position is independently cold-reprogrammed. This provides a
    # deliberately conservative constructive schedule without assuming any
    # undocumented eviction/reuse placement across changing visible contexts.
    geometries = [at(p, 1) for p in visible] if sliding else [at(visible[0], output)]
    results = []
    pin0_results = []
    floors = []
    for w in geometries:
        found = best_serial(w, budget_mm2=budget, support=area, rates=srates,
                            active_macro_ports=PORT_CAP)
        if found is None:
            return None
        plan, trace = found
        results.append((plan, trace))
        naive = residency_plan(w, kind="cim", budget_mm2=budget, support=area,
                               pinned_layers=0)
        if naive is None:
            raise AssertionError("optimizer feasible but scratch-only infeasible")
        pin0_results.append(serial_schedule(w, naive, srates,
                                            active_macro_ports=PORT_CAP))
        if sliding:
            # Per-token cold-program floors cannot be summed as necessary
            # bounds when persistent reuse across positions may be possible.
            # Discard programming entirely in the SWA exclusion certificate.
            floors.append(max(
                trace.stages_s["query_delivery"],
                trace.stages_s["completed_score_delivery"],
                lower_bound(w, PhysicalBudget(budget, supp), srates.ceilings(),
                            active_macro_port_cap=min(PORT_CAP, plan.instantiated_macros)
                            ).macro_compute_floor_s
            ))
        else:
            floors.append(trace.optimistic_resource_floor_s)

    steps = len(results)
    means = {stage: sum(trace.stages_s[stage] for _, trace in results)/steps
             for stage in results[0][1].stages_s}
    serial = sum(means.values())
    low = sum(floors)/steps
    pin0 = sum(t.serial_completed_score_s for t in pin0_results)/steps
    byte_stages = ("source_key_supply", "macro_programming",
                   "query_delivery", "completed_score_delivery")
    byte_total = sum(means[k] * bw for k in byte_stages)
    required_equal_bw = required_rate(byte_total, DEADLINE_S,
                                      serial-sum(means[k] for k in byte_stages))
    required_output_bw = required_rate(
        sum(t.completed_score_bytes_per_token for _, t in results)/steps,
        DEADLINE_S, serial-means["completed_score_delivery"])
    required_program_bw = required_rate(
        sum(p.programming_bytes_per_generated_token for p, _ in results)/steps,
        DEADLINE_S, serial-means["macro_programming"])

    digital = None
    if not sliding:
        # A digital core is NOT assigned fabricated silicon area or speed.
        dw = geometries[0]
        drates = RateSet(bw, bw, bw, bw, 1e12, source="unqualified equal-rate sensitivity")
        bound = best_digital_requirement(dw, budget_mm2=budget, support=area,
                                         rates=drates, target_s=DEADLINE_S,
                                         reuse=dw.gqa_ratio)
        if bound:
            dplan, dbound = bound
            digital = {"status": dbound.status,
                       "mac_area_available_mm2": dbound.mac_core_area_mm2,
                       "required_MAC_per_s_per_mm2": finite(dbound.required_mac_per_s_per_mm2),
                       "minimum_sram_Bps": finite(dbound.minimum_sram_Bps)}

    first, last = results[0][0], results[-1][0]
    row = {
        "model": profile.model_id, "revision": profile.revision,
        "native_attention": "MHA" if profile.gqa_ratio == 1 else "GQA",
        "evidence": "published_complete_macro_plus_conditional_serial_rate_sensitivity",
        "budget_mm2": budget, "support_mm2_sensitivity": supp,
        "batch": batch, "prefix_tokens_original": prefix, "suffix_tokens": suffix,
        "output_tokens": output, "visible_prefix_first": visible[0],
        "visible_prefix_last": visible[-1],
        "sliding_reprogram_policy": "each_visible_position_cold" if sliding else "fixed_prefix_persistent",
        "physical_macros_first": first.instantiated_macros,
        "physical_macros_last": last.instantiated_macros,
        "pinned_layers_first": first.pinned_layers,
        "pinned_layers_last": last.pinned_layers,
        "scratch_tile_units_first": first.scratch_tile_units,
        "active_macro_ports_first": results[0][1].active_macro_ports,
        "source_bandwidth_Bps_sensitivity": bw,
        "serial_completed_prefix_scores_s": serial,
        "scratch_only_serial_s": pin0,
        "optimistic_resource_lower_bound_s": low,
        "conditional_excluded_at_2ms_deadline": low > DEADLINE_S,
        "serial_achievable_at_2ms_deadline": serial <= DEADLINE_S,
        "minimum_common_byte_bandwidth_Bps_for_2ms_serial": finite(required_equal_bw),
        "minimum_completed_score_Bps_for_2ms_serial": finite(required_output_bw),
        "minimum_program_Bps_for_2ms_serial": finite(required_program_bw),
        "digital_status": digital["status"] if digital else "unavailable_for_dynamic_SWA",
        "digital_MAC_per_s_per_mm2_threshold": (digital or {}).get("required_MAC_per_s_per_mm2"),
        "digital_MAC_core_area_mm2_sensitivity": (digital or {}).get("mac_area_available_mm2"),
        "digital_min_sram_Bps": (digital or {}).get("minimum_sram_Bps"),
        "gpu_hydragen_matched_TPOT_s": None,
        "gpu_decoder_remainder_s": None,
        "S0_GPU_TPOT_s": None, "S1_SRAM_TPOT_s": None, "S2_CIM_TPOT_s": None,
        "GPU_to_CIM_speedup": None,
        "qualified_physical_digital_isoarea": False,
    }
    row.update({f"stage_{k}_s": v for k, v in means.items()})
    return row


def existing_quality(repo_root):
    source = repo_root/"results/quality"
    absolute = list(csv.DictReader((source/"absolute_quality.csv").open(newline="")))
    paired = list(csv.DictReader((source/"paired_effects.csv").open(newline="")))
    if len(absolute) != 12 or {r["arm"] for r in absolute} != {"R0", "R1", "R2", "R3"}:
        raise AssertionError("recovered Llama R0--R3 not complete")
    status = json.loads((source/"extended_status.json").read_text())
    if status["preserved_llama_evaluated_units"] != 96:
        raise AssertionError("Llama provenance mismatch")
    return {
        "evaluated_Llama_units": 96,
        "Llama_R0_R3": [{"prefix": int(x["prefix_tokens"]),
                         "arm": x["arm"], "nll": float(x["mean_nll"]),
                         "ppl": float(x["ppl"]),
                         "nll_ci95": [float(x["nll_ci95_low"]), float(x["nll_ci95_high"])]}
                        for x in absolute],
        "paired_effects": [{"prefix": int(x["prefix_tokens"]),
                            "effect": x["effect"],
                            "relative_ppl": float(x["relative_ppl"]),
                            "ci95": [float(x["relative_ppl_ci95_low"]),
                                     float(x["relative_ppl_ci95_high"])]}
                           for x in paired],
        "other_models": status["models"],
        "source": "results/quality/source_evidence.json",
    }


def render_figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = output/"figures"
    figdir.mkdir(exist_ok=True)
    def pick(p, b, o, bw, s=0.5):
        return sorted((r for r in rows if
                       r["model"] == "meta-llama/Llama-3.1-8B" and
                       r["prefix_tokens_original"] == p and r["batch"] == b and
                       r["output_tokens"] == o and r["source_bandwidth_Bps_sensitivity"] == bw
                       and r["support_mm2_sensitivity"] == s),
                      key=lambda r: r["budget_mm2"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for bw in BANDWIDTHS:
        values = pick(2048, 1, 128, bw)
        ax.plot([x["budget_mm2"] for x in values],
                [x["serial_completed_prefix_scores_s"]*1e3 for x in values],
                marker="o", label=f"Interface {bw/1e12:g} TB/s (assumed)")
    ax.set(xlabel="Total incremental accelerator area (mm²)",
           ylabel="Constructive serial completed-score time (ms/token)",
           title="Llama-3.1-8B, 2048 shared tokens, B=1, O=128")
    ax.set_xticks(BUDGETS);ax.grid(alpha=0.25);ax.legend()
    fig.tight_layout()
    fig.savefig(figdir/"conditional_score_latency.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for p in (512, 2048, 8192):
        values = pick(p, 1, 128, 1e12)
        ax.plot([x["budget_mm2"] for x in values],
                [x["minimum_common_byte_bandwidth_Bps_for_2ms_serial"]/1e12
                 if x["minimum_common_byte_bandwidth_Bps_for_2ms_serial"] else float("nan") for x in values],
                marker="o", label=f"Prefix {p}")
    ax.set(xlabel="Total incremental accelerator area (mm²)",
           ylabel="Required common interface throughput (TB/s)",
           title="2 ms *hypothetical* completed-score deadline; Llama B=1, O=128")
    ax.set_xticks(BUDGETS);ax.grid(alpha=0.25);ax.legend()
    fig.tight_layout()
    fig.savefig(figdir/"conditional_bandwidth_boundary.svg")
    plt.close(fig)


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for profile in PROFILES:
        prefixes = (256, 512, 1024) if profile.gqa_ratio == 1 else (512, 2048, 8192)
        for p in prefixes:
            for b in BATCHES:
                for o in OUTPUTS:
                    for bw in BANDWIDTHS:
                        for supp in SUPPORTS:
                            for budget in BUDGETS:
                                row = inspect_coordinate(profile, prefix=p, suffix=128,
                                                         batch=b, output=o, budget=budget,
                                                         supp=supp, bw=bw)
                                if row is not None:
                                    all_rows.append(row)
    with (out/"architecture_1_16mm2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    render_figures(all_rows, out)
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema": "integrated_conditional_physical_envelope/v1",
        "rows": len(all_rows), "per_model": dict(Counter(r["model"] for r in all_rows)),
        "budgets_mm2": list(BUDGETS), "physical_ports_cap_sensitivity": PORT_CAP,
        "bandwidths_Bps_sensitivity": list(BANDWIDTHS),
        "external_support_mm2_sensitivity": list(SUPPORTS),
        "reported_macro": {"source": CIM_7NM.source, "area_mm2": CIM_7NM.area_mm2,
                            "signed_key_planes": 2, "unsigned_evaluations": 4,
                            "native_access_s": 5.5e-9},
        "reported_sram": {"source": SRAM_7NM.source, "area_mm2": SRAM_7NM.area_mm2},
        "deadline_s": DEADLINE_S,
        "deadline_qualification": "hypothetical parameter sensitivity, NOT a GPU timing",
        "gpu_tpot_qualified_coordinates": 0,
        "speedup_qualified_coordinates": 0,
        "digital_isoarea_qualified_coordinates": 0,
        "conditional_excluded_coordinates": sum(r["conditional_excluded_at_2ms_deadline"] for r in all_rows),
        "conditional_serial_achievable_coordinates": sum(r["serial_achievable_at_2ms_deadline"] for r in all_rows),
        "illustrative_conditional_points": [
            {
                "model": row["model"],
                "budget_mm2": row["budget_mm2"],
                "support_mm2_sensitivity": row["support_mm2_sensitivity"],
                "prefix_tokens": row["prefix_tokens_original"],
                "batch": row["batch"],
                "output_tokens": row["output_tokens"],
                "bandwidth_Bps_sensitivity": row["source_bandwidth_Bps_sensitivity"],
                "macro_count": row["physical_macros_first"],
                "pinned_layers": row["pinned_layers_first"],
                "scratch_tile_units": row["scratch_tile_units_first"],
                "resource_floor_ms": row["optimistic_resource_lower_bound_s"]*1e3,
                "conditional_serial_ms": row["serial_completed_prefix_scores_s"]*1e3,
                "minimum_common_interface_TBps_for_2ms": (
                    row["minimum_common_byte_bandwidth_Bps_for_2ms_serial"]/1e12
                    if row["minimum_common_byte_bandwidth_Bps_for_2ms_serial"] is not None
                    else None
                ),
                "stage_ms": {
                    name.removeprefix("stage_").removesuffix("_s"): value*1e3
                    for name, value in row.items()
                    if name.startswith("stage_") and name.endswith("_s")
                },
            }
            for row in all_rows
            if row["prefix_tokens_original"] == (
                512 if row["model"].startswith("facebook/opt") else 2048
            )
            and row["batch"] == 1 and row["output_tokens"] == 128
            and row["source_bandwidth_Bps_sensitivity"] == 1e12
            and row["support_mm2_sensitivity"] == 0.5
            and row["budget_mm2"] in (1, 4, 8, 16)
        ],
        "end_to_end_decoder_status": "unqualified: Hydragen-style GPU full attention and remaining operators missing",
        "SWA": "Mistral each position uses causally visible prefix; changing window uses cold per-token constructive schedules; necessary bound omits recurring programming",
        "opt_limit": "prefix+suffix+generated tokens <= 2048",
        "quality": existing_quality(root),
        "no_physical_implementation_or_GPU_speedup_claim": True,
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": manifest["rows"], "per_model": manifest["per_model"],
                      "conditionally_excluded": manifest["conditional_excluded_coordinates"],
                      "conditional_serial_achievable": manifest["conditional_serial_achievable_coordinates"],
                      "qualified_GPU_TPOT": 0}))
    return manifest


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/integration"))
