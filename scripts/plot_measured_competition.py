#!/usr/bin/env python3
"""Render measured GPU versus conditional CIM FULL-attention endpoint figure.

Only accepts scripts/compare_measured_attn.py output authenticated by that
program. No synthetic fixture, 2ms target or unqualified speedup curves.
"""
import argparse
import json
from pathlib import Path


def render(source: Path, figure: Path):
    obj=json.loads(source.read_text())
    if obj.get("schema") != "cim-shared-prefix-measured-attention-physical-envelope/v1":
        raise ValueError("only measured CUDA benchmark-backed comparison accepted")
    rows=[x for x in obj.get("rows",[]) if
          x.get("measured_GPU_attention_kernel") and
          x.get("physically_qualified_CIM_interface") is False and
          x.get("GPU_best_full_attention_mean_s") is not None]
    if not rows:
        raise ValueError("no measured GPU attention kernel records")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=[r["area_budget_mm2"] for r in rows]
    gpu=[r["GPU_best_full_attention_mean_s"]*1e6 for r in rows]
    lb=[r["CIM_QK_necessary_floor_mean_per_layer_s"]*1e6 for r in rows]
    conditional=[r["CIM_plus_GPU_remainder_serial_mean_s_sensitivity"]*1e6 for r in rows]
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.plot(x,gpu,marker="o",label="GPU full attention — measured CUDA")
    ax.plot(x,lb,marker="x",label="CIM QK necessary floor — conditional")
    ax.plot(x,conditional,marker="s",label="CIM QK + GPU attention remainder — conditional")
    ax.set(xlabel="Incremental CIM area budget (mm²)",
           ylabel="Full-attention / partial-QK time (µs per layer)",
           title="Measured GPU reference; CIM physical interfaces assumed")
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout()
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--comparison-json",type=Path,required=True)
    p.add_argument("--svg",type=Path,required=True)
    args=p.parse_args()
    render(args.comparison_json,args.svg)
