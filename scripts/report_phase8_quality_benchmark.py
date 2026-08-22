"""Publish the interim Phase 8 bilan: per-family margins and benchmark choice.

Reads the persisted ``family-margins-v1`` measurement and the frozen
``phase-8-quality-partitions-v1`` manifest, refuses any drift between them,
and renders the documented choice of the final quality benchmark.
"""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.visualization.phase8 import write_quality_benchmark_choice


def main() -> None:
    args = _parse_args()
    margins_report = json.loads(args.margins.read_text(encoding="utf-8"))
    partitions_manifest = json.loads(args.partitions.read_text(encoding="utf-8"))
    write_quality_benchmark_choice(
        margins_report,
        partitions_manifest,
        args.output,
        margins_link=f"../{args.margins.as_posix()}",
        partitions_link=f"../{args.partitions.as_posix()}",
    )
    print(f"wrote {args.output}")
    retained = sorted(
        family["family_label"] for family in margins_report["families"] if family["retained"]
    )
    print(f"final quality benchmark: {len(retained)} retained families")
    for label in retained:
        print(f"  {label}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margins", type=Path, default=Path("results/phase-8-family-margins.json"))
    parser.add_argument(
        "--partitions", type=Path, default=Path("data/phase-8-partitions/manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/phase-8-quality-benchmark.md"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
