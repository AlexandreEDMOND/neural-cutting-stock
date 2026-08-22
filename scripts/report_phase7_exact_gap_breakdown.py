"""Publish the exact-gap margin breakdown by family and size class."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import build_exact_gap_breakdown
from neural_cutting_stock.visualization.phase7 import write_exact_gap_breakdown_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-gap", type=Path, default=Path("results/exact-gap.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    report = json.loads(args.exact_gap.read_text(encoding="utf-8"))
    breakdown = build_exact_gap_breakdown(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "exact-gap-breakdown.json"
    markdown_path = args.output_dir / "exact-gap-breakdown.md"
    json_path.write_text(json.dumps(breakdown, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_exact_gap_breakdown_markdown(breakdown, markdown_path, source_path=str(args.exact_gap))
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    totals = breakdown["totals"]
    print(
        f"instances: {totals['instance_count']}, "
        f"gaps available: {totals['gap_available_count']}, "
        f"zero margins: {totals['zero_gap_count']}, "
        f"positive margins: {totals['positive_gap_count']}, "
        f"excluded: {totals['excluded_instance_count']}"
    )


if __name__ == "__main__":
    main()
