"""Publish the Phase 7 closure summary from the validated exact-gap report."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.visualization.phase7 import write_phase7_summary


def main() -> None:
    args = _parse_args()
    report = json.loads(args.exact_gap.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_phase7_summary(report, args.output, source_path=str(args.exact_gap))
    counts = report["counts"]
    print(f"wrote {args.output}")
    print(
        f"gaps available: {counts['gap_available_count']}, "
        f"zero margins: {counts['zero_gap_count']}, "
        f"positive margins: {counts['positive_gap_count']}, "
        f"excluded: {counts['excluded_instance_count']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-gap", type=Path, default=Path("results/exact-gap.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-7-summary.md"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
