"""Validated, source-traceable exports of archived *evaluated* R0--R3 quality.

No surrogate NLL generation or extrapolation to new model geometries.
"""
from __future__ import annotations

import csv
import hashlib
import json
from math import exp, isclose, isfinite
from pathlib import Path

ARMS = ("R0", "R1", "R2", "R3")
EFFECTS = {"quantization": ("R0", "R1"), "macro": ("R1", "R2"),
           "circuit": ("R2", "R3"), "hardware_only": ("R1", "R3"),
           "total": ("R0", "R3")}
SOURCE_REPOSITORY = "micsedlab-afk/shared_prefix_cim"
SOURCE_BRANCH = "results/final-quality-scalar-n8-20260923"
SOURCE_DIR = "results/sram2llm/final_quality_scalar"
SOURCE_URL = f"https://github.com/{SOURCE_REPOSITORY}/tree/{SOURCE_BRANCH}/{SOURCE_DIR}"


def verify_sha256(path: str | Path, expected: str) -> None:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid expected SHA256")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError(f"archive checksum mismatch: {path}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError(f"missing/non-finite {name}")
    return float(value)


def validate_summary(summary: dict, protocol: dict | None = None,
                     provenance: dict | None = None) -> None:
    if summary.get("schema") != "final_quality_scalar_summary/v1":
        raise ValueError("unrecognized evaluated-quality schema")
    n = summary.get("n_samples")
    if not isinstance(n, int) or n < 2 or summary.get("suffix_length", 0) <= 0:
        raise ValueError("invalid paired sample or suffix count")
    prefixes = summary.get("prefix_lengths")
    if (not isinstance(prefixes, list) or not prefixes or
            sorted(set(prefixes)) != sorted(prefixes)):
        raise ValueError("invalid prefix lengths")
    by_prefix = summary.get("quality_by_prefix_length", {})
    if set(by_prefix) != {str(x) for x in prefixes}:
        raise ValueError("incomplete quality geometry")
    if len(summary.get("sample_ids", [])) != n or len(set(summary["sample_ids"])) != n:
        raise ValueError("incomplete sample identities")
    for prefix in prefixes:
        group = by_prefix[str(prefix)]
        arms = group.get("arms", {})
        if set(arms) != set(ARMS):
            raise ValueError("missing numerical arm")
        if len(group.get("window_ids", [])) != n or len(set(group["window_ids"])) != n:
            raise ValueError("missing/duplicate paired windows")
        for arm in ARMS:
            a = arms[arm]
            count = a.get("token_count", 0)
            if not isinstance(count, int) or count <= 0 or a.get("window_count") != n:
                raise ValueError("invalid token/window count")
            mu = _finite(a.get("mean_nll"), "mean_nll")
            total = _finite(a.get("nll_sum"), "nll_sum")
            ppl = _finite(a.get("ppl"), "ppl")
            if (not isclose(total / count, mu, rel_tol=1e-10, abs_tol=1e-10) or
                    not isclose(exp(mu), ppl, rel_tol=1e-10)):
                raise ValueError("mean-NLL/PPL inconsistency")
            nll_ci, ppl_ci = a.get("nll_ci", []), a.get("ppl_ci", [])
            if len(nll_ci) != 2 or len(ppl_ci) != 2:
                raise ValueError("missing uncertainty bounds")
            lo, hi = map(lambda x: _finite(x, "NLL interval"), nll_ci)
            pl, ph = map(lambda x: _finite(x, "PPL interval"), ppl_ci)
            if (lo > mu or mu > hi or pl > ppl or ppl > ph or
                    not isclose(exp(lo), pl, rel_tol=1e-9) or
                    not isclose(exp(hi), ph, rel_tol=1e-9)):
                raise ValueError("NLL/PPL confidence-interval inconsistency")
        for effect, (ref, candidate) in EFFECTS.items():
            value = group.get(effect)
            if not isinstance(value, dict) or value.get("comparison") not in (None, f"{candidate}-{ref}"):
                raise ValueError("missing or mismatched causal decomposition")
            mu_delta = arms[candidate]["mean_nll"] - arms[ref]["mean_nll"]
            rel_ppl = arms[candidate]["ppl"] / arms[ref]["ppl"] - 1
            # Historical hardware_only summary does not include delta_nll.
            if "delta_nll" in value and not isclose(value["delta_nll"], mu_delta, abs_tol=1e-10):
                raise ValueError("causal NLL mismatch")
            if not isclose(value["relative_ppl"], rel_ppl, abs_tol=1e-10):
                raise ValueError("causal PPL mismatch")
            ci = value.get("relative_ppl_ci")
            if not isinstance(ci, list) or len(ci) != 2 or not ci[0] <= rel_ppl <= ci[1]:
                raise ValueError("causal uncertainty mismatch")
    if protocol is not None:
        if (protocol.get("schema") != "final_quality_scalar_protocol/v1" or
                summary["protocol_sha256"] != protocol_sha256(protocol) or
                protocol.get("prefix_lengths") != prefixes or
                protocol.get("suffix_length") != summary["suffix_length"] or
                protocol.get("arms") != list(ARMS) or
                protocol.get("batch_size") != 1 or
                protocol.get("bootstrap_unit") != "held_out_evaluation_sample"):
            raise ValueError("inconsistent scalar protocol")
    if provenance is not None:
        if (provenance.get("protocol_sha256") != summary["protocol_sha256"] or
                provenance.get("n_samples") != n or
                provenance.get("unit_count") != n * len(prefixes) * len(ARMS)):
            raise ValueError("inconsistent archive provenance")


def protocol_sha256(protocol: dict) -> str:
    """Original archive stores the raw file hash; caller must verify raw bytes.

    Canonical JSON serialization is not assumed to equal the upstream file bytes.
    """
    return protocol.get("__verified_file_sha256", "")


def validate_sample_csv(csv_path: str | Path, summary: dict) -> None:
    """Independently reconcile original window-level nll_sum before publishing."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n = summary["n_samples"]
    lengths = summary["prefix_lengths"]
    if len(rows) != n * len(lengths) * len(ARMS):
        raise ValueError("raw archive has incomplete evaluated units")
    seen = set()
    grouped: dict[tuple[int, str], tuple[float, int]] = {}
    for row in rows:
        prefix, arm, sample = int(row["Lp"]), row["arm"], row["sample_id"]
        key = (sample, prefix, arm)
        if key in seen or sample not in summary["sample_ids"] or prefix not in lengths or arm not in ARMS:
            raise ValueError("duplicate/unrecognized paired sample")
        seen.add(key)
        token_count, nll_sum = int(row["token_count"]), float(row["nll_sum"])
        if token_count <= 0 or not isfinite(nll_sum) or not isclose(
                nll_sum / token_count, float(row["mean_nll"]), abs_tol=1e-10):
            raise ValueError("invalid raw evaluated score")
        a, b = grouped.get((prefix, arm), (0.0, 0))
        grouped[(prefix, arm)] = (a + nll_sum, b + token_count)
    for prefix in lengths:
        for arm in ARMS:
            total, count = grouped[(prefix, arm)]
            archived = summary["quality_by_prefix_length"][str(prefix)]["arms"][arm]
            if count != archived["token_count"] or not isclose(total, archived["nll_sum"], rel_tol=1e-10):
                raise ValueError("raw samples do not reconcile with summary")


def quality_rows(summary: dict) -> tuple[list[dict], list[dict]]:
    validate_summary(summary)
    points, effects = [], []
    for prefix in summary["prefix_lengths"]:
        group = summary["quality_by_prefix_length"][str(prefix)]
        for arm in ARMS:
            a = group["arms"][arm]
            points.append({"model_id": "meta-llama/Llama-3.1-8B",
                           "evidence_class": "evaluated_scalar_n8", "batch": 1,
                           "prefix_tokens": prefix, "suffix_tokens": summary["suffix_length"],
                           "output_tokens": "not_evaluated_for_generation",
                           "arm": arm, "sample_count": summary["n_samples"],
                           "scored_tokens": a["token_count"], "mean_nll": a["mean_nll"],
                           "nll_ci95_low": a["nll_ci"][0], "nll_ci95_high": a["nll_ci"][1],
                           "ppl": a["ppl"], "ppl_ci95_low": a["ppl_ci"][0],
                           "ppl_ci95_high": a["ppl_ci"][1], "source_url": SOURCE_URL})
        for effect, (ref, cand) in EFFECTS.items():
            value = group[effect]
            effects.append({"model_id": "meta-llama/Llama-3.1-8B",
                            "evidence_class": "evaluated_scalar_n8", "prefix_tokens": prefix,
                            "suffix_tokens": summary["suffix_length"], "effect": effect,
                            "reference_arm": ref, "candidate_arm": cand,
                            "relative_ppl": value["relative_ppl"],
                            "relative_ppl_ci95_low": value["relative_ppl_ci"][0],
                            "relative_ppl_ci95_high": value["relative_ppl_ci"][1],
                            "source_url": SOURCE_URL})
    return points, effects


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_quality(summary: dict, destination: str | Path, *, protocol: dict | None = None,
                   provenance: dict | None = None, sample_csv: str | Path | None = None) -> None:
    validate_summary(summary, protocol=protocol, provenance=provenance)
    if sample_csv is not None:
        validate_sample_csv(sample_csv, summary)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    points, effects = quality_rows(summary)
    _csv(target / "absolute_quality.csv", points)
    _csv(target / "paired_effects.csv", effects)
    (target / "quality.json").write_text(json.dumps({
        "schema": "cim_shared_prefix_quality/v1", "evidence_class": "evaluated_model_quality",
        "source_url": SOURCE_URL, "source_branch": SOURCE_BRANCH,
        "archive_protocol_sha256": summary["protocol_sha256"],
        "model_id": "meta-llama/Llama-3.1-8B",
        "batch": 1, "suffix_tokens": summary["suffix_length"],
        "bootstrap": {"unit": "paired_heldout_sample", "confidence": 0.95,
                      "resamples": 10000, "seed": 20260917},
        "sample_count": summary["n_samples"],
        "quality": points, "causal_effects": effects,
        "restriction": "No numerical extrapolation to new models, GQA ratios, batches, or lengths",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality_svg(points, target / "fig_absolute_ppl.svg")
    effect_svg(effects, target / "fig_r3_relative_ppl.svg")


def quality_svg(points: list[dict], path: Path) -> None:
    """Vector publication plot: absolute PPL and marginal 95% paired bootstrap CI."""
    xmap = {512: 115, 2048: 320, 16384: 525}
    colors = {"R0": "#24374a", "R1": "#2580a3", "R2": "#e18b34", "R3": "#b8404c"}
    def y(v): return 385 - (v - 3.0) * 41
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="740" height="440" viewBox="0 0 740 440">',
             '<rect width="740" height="440" fill="white"/>',
             '<text x="75" y="32" font-family="Arial" font-size="19" font-weight="bold">Evaluated Llama-3.1-8B quality (n=8)</text>',
             '<text x="75" y="54" font-family="Arial" font-size="12">BF16 vs Q4/K4 and mapped macro response; suffix=128, batch=1</text>',
             '<path d="M85 70 V385 H560" stroke="#444" stroke-width="1.2" fill="none"/>']
    for n in (3, 5, 7, 9):
        lines += [f'<line x1="85" y1="{y(n):.1f}" x2="560" y2="{y(n):.1f}" stroke="#ededed"/>',
                  f'<text x="70" y="{y(n)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{n}</text>']
    for k, x in xmap.items():
        lines.append(f'<text x="{x}" y="410" text-anchor="middle" font-family="Arial" font-size="12">{k:,}</text>')
    for j, arm in enumerate(ARMS):
        data = sorted((r for r in points if r["arm"] == arm), key=lambda r: r["prefix_tokens"])
        pathdata = " ".join(("M" if i == 0 else "L") + f'{xmap[d["prefix_tokens"]]},{y(d["ppl"]):.2f}' for i, d in enumerate(data))
        lines.append(f'<path d="{pathdata}" fill="none" stroke="{colors[arm]}" stroke-width="2.3"/>')
        for d in data:
            x, p = xmap[d["prefix_tokens"]], y(d["ppl"])
            lo, hi = y(d["ppl_ci95_low"]), y(d["ppl_ci95_high"])
            lines += [f'<path d="M{x} {lo:.2f} V{hi:.2f} M{x-4} {lo:.2f} h8 M{x-4} {hi:.2f} h8" stroke="{colors[arm]}" stroke-width="1.25" opacity=".8"/>',
                      f'<circle cx="{x}" cy="{p:.2f}" r="3.5" fill="{colors[arm]}"/>']
        lines += [f'<line x1="595" y1="{113+j*28}" x2="618" y2="{113+j*28}" stroke="{colors[arm]}" stroke-width="3"/>',
                  f'<text x="627" y="{117+j*28}" font-family="Arial" font-size="13">{arm}</text>']
    lines += ['<text x="285" y="434" font-family="Arial" font-size="12">Prefix length (tokens)</text>',
              '<text x="17" y="240" transform="rotate(-90 17 240)" font-family="Arial" font-size="12">Perplexity</text>',
              '</svg>']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def effect_svg(effects: list[dict], path: Path) -> None:
    data = sorted((row for row in effects if row["effect"] == "total"), key=lambda r: r["prefix_tokens"])
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="360" viewBox="0 0 680 360">',
             '<rect width="680" height="360" fill="white"/>',
             '<text x="65" y="32" font-family="Arial" font-size="19" font-weight="bold">R3 vs R0: paired perplexity change</text>',
             '<text x="65" y="53" font-family="Arial" font-size="12">Llama-3.1-8B, 8 evaluation samples, 95% paired-bootstrap CI</text>',
             '<path d="M70 80 V295 H570" stroke="#444" fill="none"/>']
    def y(v): return 295 - v * 20
    for n in (0, 3, 6, 9):
        lines += [f'<line x1="70" y1="{y(n)}" x2="570" y2="{y(n)}" stroke="#eeeeee"/>',
                  f'<text x="59" y="{y(n)+4}" text-anchor="end" font-family="Arial" font-size="11">{n}%</text>']
    for i, r in enumerate(data):
        x = 155 + i * 160
        lo, hi, middle = [100*r[k] for k in ("relative_ppl_ci95_low", "relative_ppl_ci95_high", "relative_ppl")]
        lines += [f'<path d="M{x} {y(lo):.1f} V{y(hi):.1f} M{x-7} {y(lo):.1f} h14 M{x-7} {y(hi):.1f} h14" stroke="#b8404c" stroke-width="2"/>',
                  f'<circle cx="{x}" cy="{y(middle):.1f}" r="5" fill="#b8404c"/>',
                  f'<text x="{x}" y="317" text-anchor="middle" font-family="Arial" font-size="12">{r["prefix_tokens"]:,}</text>']
    lines += ['<text x="255" y="347" font-family="Arial" font-size="12">Prefix length (tokens)</text>', '</svg>']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
