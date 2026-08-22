"""Measure the classical-vs-reference margin of each new Phase 8 family."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    collect_environment,
    measure_family_margins,
    phase8_family_specs,
)
from neural_cutting_stock.visualization.phase8 import write_family_margins_markdown


def main() -> None:
    args = _parse_args()
    report = measure_family_margins(
        phase8_family_specs(),
        environment=collect_environment(Path.cwd()),
        reduced_cost_tolerance=args.reduced_cost_tolerance,
        integrality_tolerance=args.integrality_tolerance,
        feasibility_tolerance=args.feasibility_tolerance,
        cross_check_with_enumeration=args.cross_check_enumeration,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase-8-family-margins.json"
    markdown_path = args.output_dir / "phase-8-family-margins.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_family_margins_markdown(report, markdown_path, source_path=json_path)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    counts = report["counts"]
    print(
        f"families: {counts['family_count']}, "
        f"instances: {counts['instance_count']}, "
        f"gaps available: {counts['gap_available_count']}, "
        f"positive gaps: {counts['positive_gap_count']}, "
        f"retained families: {counts['retained_family_count']}"
    )
    for family in report["families"]:
        print(
            f"  {family['family_label']}: "
            f"available={family['gap_available_count']}/{family['instance_count']}, "
            f"positive={family['positive_gap_count']}, "
            f"max_gap={family['max_gap_bars']}, "
            f"retained={'yes' if family['retained'] else 'no'}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--integrality-tolerance", type=float, default=1e-9)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-9)
    parser.add_argument("--cross-check-enumeration", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
