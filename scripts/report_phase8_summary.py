"""Publish the Phase 8 closure summary from persisted, validated sources."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.visualization.phase8 import write_phase8_summary


def main() -> None:
    args = _parse_args()
    margins_report = json.loads(args.margins.read_text(encoding="utf-8"))
    partitions_manifest = json.loads(args.partitions.read_text(encoding="utf-8"))
    write_phase8_summary(
        margins_report,
        partitions_manifest,
        args.output,
        margins_link=args.margins.name,
        partitions_link=f"../{args.partitions.as_posix()}",
    )
    print(f"wrote {args.output}")
    counts = margins_report["counts"]
    print(
        f"families: {counts['family_count']}, "
        f"retained: {counts['retained_family_count']}, "
        f"positive gaps: {counts['positive_gap_count']}/{counts['gap_available_count']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margins", type=Path, default=Path("results/phase-8-family-margins.json"))
    parser.add_argument(
        "--partitions", type=Path, default=Path("data/phase-8-partitions/manifest.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase-8-summary.md"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
