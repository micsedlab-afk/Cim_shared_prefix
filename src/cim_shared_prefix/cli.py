"""Command-line entrypoint for reproducible geometry and transport accounting."""
from __future__ import annotations
import argparse
import json
from .contracts import Workload, SignedMapping, ScoreInterface


def geometry_report(workload: Workload, *, partials_per_score: int = 1) -> dict:
    mapping = SignedMapping()
    interface = ScoreInterface(partials_per_score=partials_per_score)
    return {
        "schema": "cim-shared-prefix/geometry-v1",
        "workload": dict(vars(workload)),
        "gqa_ratio": workload.gqa_ratio,
        "logical_prefix_key_bytes_per_layer": workload.prefix_key_bytes_per_layer,
        "logical_prefix_key_bytes_all_layers": workload.prefix_key_bytes_all_layers,
        "physical_signed_prefix_key_bytes_all_layers": mapping.physical_key_bytes_all_layers(workload),
        "logical_prefix_qk_macs_per_token": workload.total_prefix_qk_mac_per_generated_token,
        "signed_native_array_work_per_token": mapping.physical_array_work_per_generated_token(workload),
        "prefix_score_values_per_token": workload.layers * workload.prefix_scores_per_layer,
        "score_link_bytes_per_token": interface.bytes_per_generated_token(workload),
        "score_partials_per_value": partials_per_score,
        "timing": "unsupported until source-grounded external models are connected",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--suffix", type=int, required=True)
    parser.add_argument("--q-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--partials", type=int, default=1)
    args = parser.parse_args(argv)
    workload = Workload(
        args.batch, args.prefix, args.suffix, args.q_heads,
        args.kv_heads, args.head_dim, args.layers, args.output_tokens,
        args.key_bits,
    )
    print(json.dumps(geometry_report(workload, partials_per_score=args.partials),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
