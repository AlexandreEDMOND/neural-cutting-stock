"""Evaluate the trained Phase 9 quality policy on a frozen quality partition.

Every instance of the partition is refined through the Neural-QC pipeline:
real classical column-generation start, greedy proposals of the checkpointed
policy, systematic independent verification. The persisted report carries
verified solutions alongside preserved failures and aggregates the mean
bars saved against the classical baseline over the partition, by family and
by size. No duration enters the report; quality is the only metric.
"""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    collect_environment,
    read_quality_partition_manifest,
)
from neural_cutting_stock.learning import (
    DEFAULT_MAX_STEPS,
    NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION,
    NeuralQCBudget,
    checkpoint_sha256,
    evaluate_quality_agent_on_partition,
    quality_agent_from_checkpoint,
)


def main() -> int:
    args = _parse_args()
    manifest = read_quality_partition_manifest(args.manifest)
    agent = quality_agent_from_checkpoint(args.checkpoint)
    print(
        f"evaluating {args.checkpoint} (sha256 {checkpoint_sha256(args.checkpoint)[:16]}...) "
        f"on partition '{args.partition}' of {args.manifest}"
    )

    report = evaluate_quality_agent_on_partition(
        manifest,
        args.partition,
        agent,
        budget=NeuralQCBudget(args.max_steps, args.stall_patience),
        verification_tolerance=args.verification_tolerance,
    )
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "sha256": checkpoint_sha256(args.checkpoint),
    }
    environment = collect_environment(Path.cwd())
    report["environment"] = {
        "code_commit": environment.code_commit,
        "python_version": environment.python_version,
        "dependency_versions": environment.dependency_versions,
        "hardware_id": environment.hardware_id,
    }

    output_path = args.output_dir / f"phase-9-{args.partition}-eval.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert report["schema_version"] == NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION
    counts = report["counts"]
    overall = report["overall"]
    print(f"wrote {output_path}")
    print(
        f"instances: {counts['instance_count']}, "
        f"solutions: {counts['published_solution_count']}, "
        f"failures preserved: {counts['preserved_failure_count']}"
    )
    if overall["published_solution_count"]:
        print(
            f"bars saved total: {overall['total_bars_saved']}, "
            f"mean per instance: {overall['mean_bars_saved']:.6f} "
            f"(improved {overall['improved_count']}, equal {overall['equal_count']})"
        )
    else:
        print("bars saved: no published solution")
    for group in report["by_family"]:
        print(_group_line("family", group))
    for group in report["by_size"]:
        print(_group_line("size", group))
    return 0


def _group_line(dimension: str, group: dict) -> str:
    mean = group["mean_bars_saved"]
    mean_text = "n/a" if mean is None else f"{mean:.6f}"
    return (
        f"  {dimension} {group['key']}: instances={group['instance_count']}, "
        f"solutions={group['published_solution_count']}, "
        f"failures={group['preserved_failure_count']}, "
        f"total_bars_saved={group['total_bars_saved']}, mean_bars_saved={mean_text}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/phase-8-partitions/manifest.json",
        help="frozen phase-8 quality partition manifest",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/phase-9-quality-policy.pt",
        help="trained quality policy checkpoint",
    )
    parser.add_argument("--partition", default="validation", help="partition to evaluate")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--stall-patience", type=int, default=1)
    parser.add_argument("--verification-tolerance", type=float, default=1e-9)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
