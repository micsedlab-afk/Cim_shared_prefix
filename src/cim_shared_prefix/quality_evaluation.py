"""Deterministic, paired, model-specific quality evidence; no fabricated NLL."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ARMS = ("R0", "R1", "R2", "R3")
EFFECTS = {
    "quantization": ("R0", "R1"),
    "mapping_adc": ("R1", "R2"),
    "characterized_response": ("R2", "R3"),
    "total": ("R0", "R3"),
}
SEED = 20260917
RESAMPLES = 10000


def token_digest(ids: Sequence[int]) -> str:
    import numpy as np
    arr = np.asarray(ids, dtype="<i8")
    if arr.ndim != 1 or bool((arr < 0).any()):
        raise ValueError("invalid token stream")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def fixed_windows(ids: Sequence[int], *, model_id: str,
                  prefixes: tuple[int, ...], suffix: int = 128,
                  n_evaluation: int = 8, n_development: int = 2) -> dict:
    """Anchor suffix text across Lp, keep dev and heldout blocks disjoint.

    Tokenizers are model-specific, so exact token IDs cannot be shared across
    OPT and Mistral; the pairing unit is a single model-specific text block.
    """
    if not model_id or not prefixes or any(p <= 0 for p in prefixes) or suffix <= 0:
        raise ValueError("invalid deterministic workload")
    if sorted(set(prefixes)) != sorted(prefixes) or n_evaluation < 2:
        raise ValueError("invalid prefix or sample selection")
    block_len = max(prefixes) + suffix + 1
    available = len(ids) // block_len
    if available < n_evaluation + n_development:
        raise ValueError(f"only {available} eligible corpus blocks for {n_evaluation}+{n_development}")
    dev = list(range(n_development))
    remaining = range(n_development, available)
    ranked = sorted(remaining, key=lambda i: hashlib.sha256(
        f"{model_id}:quality-v1:{i}:{SEED}".encode()).hexdigest())
    evaluated = ranked[:n_evaluation]

    def build(indices: list[int], prefix: str):
        output = []
        for i, block in enumerate(indices):
            anchor = block * block_len + max(prefixes)
            sample_id = f"{prefix}_{i:03d}"
            for lp in prefixes:
                start, end = anchor-lp, anchor+suffix+1
                window_ids = ids[start:end]
                if len(window_ids) != lp + suffix + 1:
                    raise ValueError("incomplete corpus window")
                output.append({"sample_id": sample_id, "block_index": block,
                               "prefix_tokens": lp, "suffix_tokens": suffix,
                               "start": start, "end": end,
                               "window_sha256": token_digest(window_ids),
                               "targets_sha256": token_digest(window_ids[lp+1:])})
        return output

    return {"schema": "model_quality_windows/v1", "model_id": model_id,
            "seed": SEED, "token_count": len(ids), "token_stream_sha256": token_digest(ids),
            "prefixes": list(prefixes), "suffix": suffix, "eligible_blocks": available,
            "development_blocks": dev, "evaluation_blocks": evaluated,
            "development": build(dev, "dev"), "evaluation": build(evaluated, "eval")}


def _percentile(draws: list[float], p: float) -> float:
    seq = sorted(draws)
    pos = (len(seq)-1) * p
    lo = int(math.floor(pos))
    return seq[lo] + (seq[min(lo+1, len(seq)-1)]-seq[lo])*(pos-lo)


def _bounds(draws: list[float]) -> list[float]:
    return [_percentile(draws, .025), _percentile(draws, .975)]


def validate_units(units: list[dict], *, model_id: str, prefixes: tuple[int, ...],
                   suffix: int, n_samples: int = 8) -> dict[int, list[dict[str, dict]]]:
    groups: dict[tuple[int,str],dict[str,dict]] = defaultdict(dict)
    if len(units) != n_samples * len(prefixes) * len(ARMS):
        raise ValueError("incomplete four-arm numerical campaign")
    for u in units:
        if u.get("model_id") != model_id or u.get("arm") not in ARMS or u.get("batch") != 1:
            raise ValueError("mixed model, arm, or batch")
        lp = u["prefix_tokens"]
        if lp not in prefixes or u["suffix_tokens"] != suffix:
            raise ValueError("mixed numerical geometry")
        key = lp, u["sample_id"]
        if u["arm"] in groups[key]:
            raise ValueError("duplicate result")
        if (not isinstance(u["token_count"], int) or u["token_count"] != suffix
                or not math.isfinite(u["nll_sum"]) or u["nll_sum"] < 0
                or not isinstance(u.get("targets_sha256"),str)
                or len(u["targets_sha256"]) != 64):
            raise ValueError("invalid tokens or NLL")
        groups[key][u["arm"]] = u
    out = {}
    for lp in prefixes:
        candidates = sorted((s, arms) for (p,s),arms in groups.items() if p==lp)
        if len(candidates) != n_samples:
            raise ValueError("incomplete paired sample set")
        for sample, arms in candidates:
            if set(arms) != set(ARMS) or len({a["targets_sha256"] for a in arms.values()})!=1:
                raise ValueError("unpaired targets across R0--R3")
            if sum(a["token_count"] for a in arms.values()) != suffix * 4:
                raise ValueError("invalid target counts")
        out[lp] = [arms for _,arms in candidates]
    sample_names = [{arms["R0"]["sample_id"] for arms in values} for values in out.values()]
    if any(s != sample_names[0] for s in sample_names):
        raise ValueError("cross-prefix paired sample identities differ")
    return out


def aggregate_units(units: list[dict], *, model_id: str, prefixes: tuple[int,...],
                    suffix: int = 128, n_samples: int = 8,
                    bootstrap_seed: int = SEED, resamples: int = RESAMPLES) -> dict:
    paired = validate_units(units, model_id=model_id, prefixes=prefixes,
                            suffix=suffix, n_samples=n_samples)
    if resamples < 100:
        raise ValueError("insufficient bootstrap iterations")
    rng = random.Random(bootstrap_seed)
    draws = [tuple(rng.randrange(n_samples) for _ in range(n_samples))
             for _ in range(resamples)]
    rows, effects = [], []
    for lp in prefixes:
        samples = paired[lp]
        means = {a: math.fsum(row[a]["nll_sum"] for row in samples)/(suffix*n_samples)
                 for a in ARMS}
        boot = {a: [] for a in ARMS}
        for idxs in draws:
            for a in ARMS:
                boot[a].append(math.fsum(samples[i][a]["nll_sum"] for i in idxs) / (suffix*n_samples))
        for a in ARMS:
            lo,hi = _bounds(boot[a])
            rows.append({"model_id":model_id,"evidence_class":"evaluated_scalar_n8",
                         "prefix_tokens":lp,"suffix_tokens":suffix,"batch":1,"arm":a,
                         "samples":n_samples,"tokens":suffix*n_samples,
                         "mean_nll":means[a],"nll_ci95_low":lo,"nll_ci95_high":hi,
                         "ppl":math.exp(means[a]),"ppl_ci95_low":math.exp(lo),
                         "ppl_ci95_high":math.exp(hi)})
        for effect,(a,b) in EFFECTS.items():
            diffs = [v-u for u,v in zip(boot[a],boot[b])]
            rels = [math.expm1(v) for v in diffs]
            d = means[b]-means[a]
            dci = _bounds(diffs)
            rci = _bounds(rels)
            effects.append({"model_id":model_id,"evidence_class":"evaluated_scalar_n8",
                            "prefix_tokens":lp,"suffix_tokens":suffix,"effect":effect,
                            "reference_arm":a,"candidate_arm":b,"delta_nll":d,
                            "delta_nll_ci95_low":dci[0],"delta_nll_ci95_high":dci[1],
                            "relative_ppl":math.expm1(d),"relative_ppl_ci95_low":rci[0],
                            "relative_ppl_ci95_high":rci[1]})
    return {"schema":"model_quality_aggregate/v1","model_id":model_id,
            "evidence_class":"evaluated_model_quality","n_samples":n_samples,
            "prefixes":list(prefixes),"suffix":suffix,
            "bootstrap":{"seed":bootstrap_seed,"resamples":resamples,
                         "unit":"paired evaluation block","ci":.95},
            "absolute":rows,"effects":effects}


def write_aggregate(data: dict, directory: str | Path, provenance: dict) -> None:
    destination=Path(directory)
    destination.mkdir(parents=True,exist_ok=True)
    for key,name in (("absolute","absolute_quality.csv"),("effects","paired_effects.csv")):
        rows=data[key]
        if not rows:
            raise ValueError("do not write empty numerical quality output")
        with (destination/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    final=dict(data)
    final["provenance"]=provenance
    (destination/"quality.json").write_text(json.dumps(final,sort_keys=True,indent=2)+"\n")
    figure_effects(data, destination/"fig_quality_effects.svg")


def figure_effects(data:dict, path:Path) -> None:
    """Plot signed NLL effects, retaining negative R3-R2 values and CIs."""
    effects=[r for r in data["effects"] if r["effect"] in (
        "quantization", "mapping_adc", "characterized_response")]
    if not effects: raise ValueError("no evaluated effects")
    ps=data["prefixes"]
    vals=[v for r in effects for v in (r["delta_nll_ci95_low"],r["delta_nll_ci95_high"])]
    ymin=min(min(vals),0)-.01; ymax=max(max(vals),0)+.01
    def y(v):return 360-280*(v-ymin)/(ymax-ymin)
    cols={"quantization":"#247ba0","mapping_adc":"#ed8f27","characterized_response":"#a94350"}
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="790" height="430" viewBox="0 0 790 430">',
           '<rect width="790" height="430" fill="white"/>',
           '<text x="65" y="30" font-family="Arial" font-size="18">Evaluated paired NLL effects (95% CI)</text>',
           f'<line x1="75" x2="625" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#555"/>']
    for j,lp in enumerate(ps):
        x=160+185*j
        lines.append(f'<text x="{x}" y="395" font-family="Arial" font-size="12" text-anchor="middle">{lp:,}</text>')
        for i,effect in enumerate(cols):
            r=next(v for v in effects if v["prefix_tokens"]==lp and v["effect"]==effect)
            xx=x+20*(i-1); lo=y(r["delta_nll_ci95_low"]);hi=y(r["delta_nll_ci95_high"]);z=y(r["delta_nll"])
            lines.extend([f'<path d="M{xx} {lo:.1f} V{hi:.1f} M{xx-4} {lo:.1f} h8 M{xx-4} {hi:.1f} h8" stroke="{cols[effect]}" stroke-width="2"/>',
                          f'<circle cx="{xx}" cy="{z:.1f}" r="4" fill="{cols[effect]}"/>'])
    for i,(k,c) in enumerate(cols.items()):
        yy=100+i*30; lines.extend([f'<line x1="640" y1="{yy}" x2="662" y2="{yy}" stroke="{c}" stroke-width="3"/>',
                                    f'<text x="670" y="{yy+4}" font-family="Arial" font-size="11">{k.replace("_"," ")}</text>'])
    lines.extend(['<text x="310" y="416" font-family="Arial" font-size="12">Prefix length (tokens)</text>',
                  '<text x="22" y="250" transform="rotate(-90 22 250)" font-family="Arial" font-size="12">NLL difference</text>','</svg>'])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
