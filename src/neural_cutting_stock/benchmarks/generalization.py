"""Out-of-distribution size generalization from persisted final campaign runs."""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .comparison import INSTANCE_IDENTITY_FIELDS, PairedRunComparison, build_paired_comparison
from .corpus import read_corpus_manifest, trajectory_sha256
from .partitions import DatasetPartition
from .schema import BenchmarkRunRecord, SolverMode
from .trajectory import read_trajectory

GENERALIZATION_SCHEMA_VERSION = "size-generalization-v1"


def training_size_frontier(corpus_manifest_path: str | Path) -> dict[str, Any]:
    """Derive the largest validated piece-type count of the learning corpus.

    Every trajectory referenced by the manifest is schema-read and hash-checked.
    The frontier is the maximum piece-type count over all partitions, so an
    instance only counts as above-training when no learning partition ever saw
    its size.
    """

    manifest = read_corpus_manifest(corpus_manifest_path)
    root = Path(corpus_manifest_path).parent.resolve()
    counts_by_partition: dict[str, int] = {}
    for entry in manifest["trajectories"]:
        if not isinstance(entry, dict):
            raise ValueError("trajectory manifest entries must be objects")
        try:
            trajectory_id = entry["trajectory_id"]
            relative_path = Path(entry["path"])
            partition = DatasetPartition(entry["partition"])
            expected_hash = entry["sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid trajectory manifest entry") from error
        if not isinstance(trajectory_id, str) or not isinstance(expected_hash, str):
            raise ValueError("trajectory manifest identity fields must be strings")
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise ValueError(f"trajectory path escapes corpus directory: {relative_path}")
        trajectory = read_trajectory(path)
        if trajectory.metadata.trajectory_id != trajectory_id:
            raise ValueError(f"trajectory identity differs for {relative_path}")
        if trajectory_sha256(trajectory) != expected_hash:
            raise ValueError(f"trajectory hash differs for {trajectory_id!r}")
        count = len(trajectory.metadata.piece_lengths)
        previous = counts_by_partition.get(partition.value)
        counts_by_partition[partition.value] = (
            count if previous is None else max(previous, count)
        )
    if not counts_by_partition:
        raise ValueError("learning corpus must contain trajectories")
    return {
        "source_schema_version": manifest["schema_version"],
        "maximum_training_piece_types": max(counts_by_partition.values()),
        "piece_types_by_partition": dict(sorted(counts_by_partition.items())),
    }


def pair_campaign_records(
    classical_records: Sequence[BenchmarkRunRecord],
    neural_records: Sequence[BenchmarkRunRecord],
) -> tuple[tuple[BenchmarkRunRecord, BenchmarkRunRecord], ...]:
    """Pair separately executed campaigns run-to-run on identical instances.

    Each mode was executed as its own frozen-protocol matrix, so campaign-level
    ``config_id`` values legitimately differ; identity therefore requires the
    same environment, seed and instance data instead of one shared config_id.
    """

    grouped: dict[tuple[str, int], dict[SolverMode, BenchmarkRunRecord]] = {}
    for mode, records in (
        (SolverMode.CLASSICAL, classical_records),
        (SolverMode.NEURAL, neural_records),
    ):
        for record in records:
            if record.solver_mode is not mode:
                raise ValueError(
                    f"{mode.value} pairing received a {record.solver_mode.value} run"
                )
            key = (record.instance_id, record.repetition)
            modes = grouped.setdefault(key, {})
            if mode in modes:
                raise ValueError(f"duplicate {mode.value} run for {key}")
            modes[mode] = record
    pairs = []
    for key, modes in sorted(grouped.items()):
        classical = modes.get(SolverMode.CLASSICAL)
        neural = modes.get(SolverMode.NEURAL)
        if classical is None or neural is None:
            raise ValueError(f"missing paired campaign run for {key}")
        _validate_campaign_pair_identity(classical, neural)
        pairs.append((classical, neural))
    return tuple(pairs)


def evaluate_size_generalization(
    classical_records: Sequence[BenchmarkRunRecord],
    neural_records: Sequence[BenchmarkRunRecord],
    frontier: dict[str, Any],
    quality_tolerance: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare both modes against the training size frontier from raw records.

    Objective differences and quality are recomputed per pair before any
    aggregation; failed or non-admissible repetitions remain visible in their
    instance rows and in the coverage counts.
    """

    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    maximum = frontier.get("maximum_training_piece_types")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError("frontier must declare a positive maximum_training_piece_types")

    grouped: dict[
        str, list[tuple[BenchmarkRunRecord, BenchmarkRunRecord, PairedRunComparison]]
    ] = {}
    for classical, neural in pair_campaign_records(classical_records, neural_records):
        comparison = build_paired_comparison(classical, neural, quality_tolerance)
        grouped.setdefault(classical.instance_id, []).append((classical, neural, comparison))

    instances = []
    for instance_id, rows in sorted(
        grouped.items(),
        key=lambda item: (item[1][0][0].number_of_piece_types, item[0]),
    ):
        type_count = rows[0][0].number_of_piece_types
        status_counts: dict[str, int] = {}
        for classical, neural, _ in rows:
            for record in (classical, neural):
                key = f"{record.solver_mode.value}:{record.run_status.value}"
                status_counts[key] = status_counts.get(key, 0) + 1
        instances.append(
            {
                "instance_id": instance_id,
                "number_of_piece_types": type_count,
                "above_training_size": type_count > maximum,
                "repetition_count": len(rows),
                "status_counts": dict(sorted(status_counts.items())),
                "objective_differences_vs_classical": [
                    comparison.objective_difference_vs_classical
                    for _, _, comparison in rows
                ],
                "admissible_repetition_count": sum(
                    comparison.quality_preserved and comparison.speedup_vs_classical is not None
                    for _, _, comparison in rows
                ),
            }
        )

    above = [item for item in instances if item["above_training_size"]]
    within = [item for item in instances if not item["above_training_size"]]
    report = {
        "schema_version": GENERALIZATION_SCHEMA_VERSION,
        "quality_tolerance_bars": quality_tolerance,
        "training_frontier": frontier,
        "campaign_config_ids": {
            "classical": sorted({record.config_id for record in classical_records}),
            "neural": sorted({record.config_id for record in neural_records}),
        },
        "run_count": len(classical_records) + len(neural_records),
        "pair_count": sum(item["repetition_count"] for item in instances),
        "instances": instances,
        "coverage": {
            "instance_count_above_training": len(above),
            "instance_count_within_training": len(within),
            "admissible_pair_count_above_training": sum(
                item["admissible_repetition_count"] for item in above
            ),
            "admissible_pair_count_within_training": sum(
                item["admissible_repetition_count"] for item in within
            ),
            "objective_differences_vs_classical_above_training": [
                difference
                for item in above
                for difference in item["objective_differences_vs_classical"]
            ],
        },
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _validate_campaign_pair_identity(
    classical: BenchmarkRunRecord, neural: BenchmarkRunRecord
) -> None:
    """Reject pairs that were not measured on the same instance and conditions."""

    if classical.environment != neural.environment:
        raise ValueError("paired campaign runs must use the same environment")
    if classical.seed != neural.seed:
        raise ValueError("paired campaign runs must use the same seed")
    for field in INSTANCE_IDENTITY_FIELDS:
        if getattr(classical, field) != getattr(neural, field):
            raise ValueError(f"paired campaign runs must use the same instance data: {field}")
