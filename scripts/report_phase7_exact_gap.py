"""Compute the classical-baseline gap to exact references over the whole corpus."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    BenchmarkRunRecord,
    CorpusBaseline,
    SolverMode,
    SyntheticInstanceGenerator,
    build_exact_gap_report,
    collect_environment,
    generators_from_final_manifest,
    write_exact_gap_csv,
)
from neural_cutting_stock.visualization.phase4 import load_phase4_runs


def main() -> None:
    args = _parse_args()
    environment = collect_environment(Path.cwd())
    final_manifest = json.loads(args.final_manifest.read_text(encoding="utf-8"))
    phase3_manifest = json.loads(args.phase3_manifest.read_text(encoding="utf-8"))
    generators = generators_from_final_manifest(final_manifest, phase3_manifest)

    corpus: list[CorpusBaseline] = []
    exclusions: list[dict[str, str]] = []
    covered: set[str] = set()

    phase6_grouped: dict[str, list[BenchmarkRunRecord]] = defaultdict(list)
    for record in _classical_runs(load_phase4_runs(args.phase6_classical_runs)):
        phase6_grouped[record.instance_id].append(record)
    for entry, generator in zip(final_manifest["instances"], generators, strict=True):
        instance_id = entry["instance_id"]
        records = phase6_grouped.get(instance_id, ())
        if not records:
            exclusions.append(
                {
                    "instance_id": instance_id,
                    "source": str(args.final_manifest),
                    "reason": "no persisted classical baseline run",
                }
            )
            continue
        corpus.append(
            CorpusBaseline(
                instance_id=instance_id,
                instance=generator.generate(),
                source=str(args.final_manifest),
                size_class=entry["target_size_class"],
                family_id=entry["family_id"],
                classical_records=tuple(records),
            )
        )
        covered.add(instance_id)

    phase4_grouped: dict[str, list[BenchmarkRunRecord]] = defaultdict(list)
    for record in _classical_runs(load_phase4_runs(args.phase4_runs)):
        phase4_grouped[record.instance_id].append(record)
    for instance_id, records in sorted(phase4_grouped.items()):
        outcome = _phase4_corpus_baseline(
            instance_id, records, source=str(args.phase4_runs)
        )
        if isinstance(outcome, CorpusBaseline):
            corpus.append(outcome)
            covered.add(instance_id)
        else:
            exclusions.append(outcome)

    for trajectory in phase3_manifest.get("trajectories", []):
        instance_id = trajectory.get("instance_id") if isinstance(trajectory, dict) else None
        if isinstance(instance_id, str) and instance_id and instance_id not in covered:
            exclusions.append(
                {
                    "instance_id": instance_id,
                    "source": str(args.phase3_manifest),
                    "reason": "no persisted classical baseline run",
                }
            )

    excluded_ids = {item["instance_id"] for item in exclusions}
    profile = json.loads(args.phase2_profile.read_text(encoding="utf-8"))
    for run in profile.get("runs", []):
        instance_id = run.get("instance_id") if isinstance(run, dict) else None
        if (
            isinstance(instance_id, str)
            and instance_id
            and instance_id not in covered
            and instance_id not in excluded_ids
        ):
            exclusions.append(
                {
                    "instance_id": instance_id,
                    "source": str(args.phase2_profile),
                    "reason": "instance data was never persisted; "
                    "the exact reference cannot be bound to this baseline",
                }
            )

    report = build_exact_gap_report(
        corpus,
        environment=environment,
        integrality_tolerance=args.integrality_tolerance,
        feasibility_tolerance=args.feasibility_tolerance,
        cross_check_with_enumeration=args.cross_check_enumeration,
        exclusions=exclusions,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "exact-gap.json"
    csv_path = args.output_dir / "exact-gap.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_exact_gap_csv(report, csv_path)
    counts = report["counts"]
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(
        f"instances: {counts['instance_count']}, "
        f"optimal references: {counts['optimal_reference_count']}, "
        f"gaps available: {counts['gap_available_count']}, "
        f"zero gaps: {counts['zero_gap_count']}, "
        f"positive gaps: {counts['positive_gap_count']}, "
        f"excluded: {counts['excluded_instance_count']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-manifest", type=Path, default=Path("data/phase-6-final/manifest.json")
    )
    parser.add_argument(
        "--phase3-manifest", type=Path, default=Path("data/phase-3-corpus/manifest.json")
    )
    parser.add_argument(
        "--phase6-classical-runs",
        type=Path,
        default=Path("results/phase-6-classical-runs.csv"),
    )
    parser.add_argument(
        "--phase4-runs", type=Path, default=Path("results/phase-4-benchmark-runs.csv")
    )
    parser.add_argument(
        "--phase2-profile", type=Path, default=Path("results/phase-2-baseline-profile.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--integrality-tolerance", type=float, default=1e-9)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-9)
    parser.add_argument("--cross-check-enumeration", action="store_true")
    return parser.parse_args()


def _classical_runs(records: tuple[BenchmarkRunRecord, ...]) -> list[BenchmarkRunRecord]:
    return [record for record in records if record.solver_mode is SolverMode.CLASSICAL]


def _phase4_corpus_baseline(
    instance_id: str, records: list[BenchmarkRunRecord], *, source: str
) -> CorpusBaseline | dict[str, str]:
    """Rebuild one phase-4 corpus entry, refusing anything not hash-verifiable.

    The raw rows carry only the aggregate instance dimensions, so the
    deterministic generator is reconstructed from the recorded seed and type
    count under the documented uniform_integer_v1 defaults. The reconstruction
    is accepted only when its normalized instance reproduces the recorded
    instance_id exactly; otherwise the instance stays visible as an exclusion.
    """

    first = records[0]
    signature = (first.seed, first.number_of_piece_types, first.stock_length, first.kerf)
    for record in records:
        if (
            record.seed,
            record.number_of_piece_types,
            record.stock_length,
            record.kerf,
        ) != signature:
            return {
                "instance_id": instance_id,
                "source": source,
                "reason": "inconsistent generator parameters across baseline runs",
            }
    generator = SyntheticInstanceGenerator(
        seed=first.seed, number_of_types=first.number_of_piece_types
    )
    if generator.instance_id != instance_id:
        return {
            "instance_id": instance_id,
            "source": source,
            "reason": "generator reconstruction does not reproduce the recorded instance_id",
        }
    return CorpusBaseline(
        instance_id=instance_id,
        instance=generator.generate(),
        source=source,
        size_class=first.size_class,
        family_id=generator.family_id,
        classical_records=tuple(records),
    )


if __name__ == "__main__":
    main()
