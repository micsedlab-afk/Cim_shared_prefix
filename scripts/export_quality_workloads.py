"""Regenerate numerical publication exports from preserved local upstream files.

Use --archive-dir pointing to the original final_quality_scalar directory. The
original window-level result archives stay outside this public repository.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cim_shared_prefix.quality import export_quality, verify_sha256
from cim_shared_prefix.profiles import export_sweep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, help="archived scalar result directory")
    p.add_argument("--out", type=Path, default=Path("results/quality"))
    p.add_argument("--profile-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    export_sweep(args.out / "geometry_sweep.csv")
    if args.profile_only:
        return
    if not args.archive_dir:
        p.error("--archive-dir required for evaluated R0-R3 exports; use --profile-only otherwise")
    root = args.archive_dir
    summary_file, protocol_file, provenance_file = [root / s for s in
        ("summary_n8.json", "protocol.json", "provenance_n8.json")]
    summary = json.loads(summary_file.read_text())
    protocol = json.loads(protocol_file.read_text())
    provenance = json.loads(provenance_file.read_text())
    # Verify the original bytes, not a reserialized protocol.
    original_sha = summary["protocol_sha256"]
    verify_sha256(protocol_file, original_sha)
    protocol["__verified_file_sha256"] = original_sha
    for filename, sha in provenance["artifact_sha256"].items():
        verify_sha256(root / filename, sha)
    export_quality(summary, args.out, protocol=protocol, provenance=provenance,
                   sample_csv=root / "sample_arm_n8.csv")


if __name__ == "__main__":
    main()
