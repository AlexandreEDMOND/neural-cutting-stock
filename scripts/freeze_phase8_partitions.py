"""Freeze the train/validation/test partitions of the retained Phase 8 families.

Reads the persisted ``family-margins-v1`` report, keeps only the families its
retention rule marked as retained, assigns every measured cell to a partition
by seed, and validates absence of leakage before writing the versioned
manifest.
"""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    build_quality_partition_plan,
    write_quality_partition_manifest,
)


def main() -> None:
    args = _parse_args()
    margins_report = json.loads(args.margins.read_text(encoding="utf-8"))
    manifest = build_quality_partition_plan(margins_report)
    write_quality_partition_manifest(args.output, manifest)
    print(f"wrote {args.output}")
    statistics = manifest["statistics"]
    print(
        f"families: {statistics['family_count']}, instances: {statistics['instance_count']}, "
        f"plan_id: {manifest['plan_id'][:12]}…"
    )
    for partition in ("train", "validation", "test"):
        print(
            f"  {partition}: {statistics['partition_instance_counts'][partition]} instances "
            f"across {statistics['partition_family_counts'][partition]} families "
            f"(seeds {manifest['seed_partitions'][partition]})"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margins", type=Path, default=Path("results/phase-8-family-margins.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/phase-8-partitions/manifest.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
