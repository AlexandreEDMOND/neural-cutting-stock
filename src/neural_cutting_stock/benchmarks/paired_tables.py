"""Paired quality, runtime, memory, iteration and column publication tables."""

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .comparison import build_paired_comparison
from .generalization import pair_campaign_records
from .schema import BenchmarkRunRecord

PAIRED_TABLES_SCHEMA_VERSION = "phase-6-paired-tables-v1"


def build_paired_tables(
    classical_records: Sequence[BenchmarkRunRecord],
    neural_records: Sequence[BenchmarkRunRecord],
    quality_tolerance: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build paired tables from raw campaign runs without filtering any pair.

    Quality, runtime, memory, iteration and column values are recomputed per
    run-to-run pair before aggregation. Instance medians aggregate only
    admissible repetitions (quality preserved and both runtimes measured);
    failed, timeouted or quality-violating pairs stay visible with their raw
    diagnostics in the ``pairs`` list and the status counts.
    """

    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs = []
    for classical, neural in pair_campaign_records(classical_records, neural_records):
        comparison = build_paired_comparison(classical, neural, quality_tolerance)
        row = _pair_row(classical, neural, comparison)
        pairs.append(row)
        grouped[classical.instance_id].append(row)
    instances = [
        _instance_row(instance_id, rows)
        for instance_id, rows in sorted(
            grouped.items(), key=lambda item: (item[1][0]["number_of_piece_types"], item[0])
        )
    ]
    report = {
        "schema_version": PAIRED_TABLES_SCHEMA_VERSION,
        "quality_tolerance_bars": quality_tolerance,
        "run_count": len(classical_records) + len(neural_records),
        "pair_count": len(pairs),
        "admissible_pair_count": sum(row["admissible"] for row in pairs),
        "instances": instances,
        "pairs": pairs,
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _pair_row(
    classical: BenchmarkRunRecord, neural: BenchmarkRunRecord, comparison: Any
) -> dict[str, Any]:
    return {
        "instance_id": comparison.instance_id,
        "repetition": comparison.repetition,
        "number_of_piece_types": classical.number_of_piece_types,
        "classical_run_id": comparison.classical_run_id,
        "neural_run_id": comparison.neural_run_id,
        "classical_run_status": classical.run_status.value,
        "neural_run_status": neural.run_status.value,
        "objective_classical_bars": classical.objective_value,
        "objective_neural_bars": neural.objective_value,
        "objective_difference_vs_classical": comparison.objective_difference_vs_classical,
        "quality_preserved": comparison.quality_preserved,
        "admissible": comparison.quality_preserved
        and comparison.speedup_vs_classical is not None,
        "classical_total_runtime_seconds": classical.total_runtime_seconds,
        "neural_total_runtime_seconds": neural.total_runtime_seconds,
        "speedup_vs_classical": comparison.speedup_vs_classical,
        "classical_peak_memory_bytes": classical.peak_memory_bytes,
        "neural_peak_memory_bytes": neural.peak_memory_bytes,
        "classical_cg_iterations": classical.number_of_cg_iterations,
        "neural_cg_iterations": neural.number_of_cg_iterations,
        "classical_generated_columns": classical.number_of_generated_columns,
        "neural_generated_columns": neural.number_of_generated_columns,
        "classical_added_columns": classical.number_of_columns_added,
        "neural_added_columns": neural.number_of_columns_added,
        "classical_final_columns": classical.final_column_count,
        "neural_final_columns": neural.final_column_count,
    }


def _instance_row(instance_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    admissible = [row for row in rows if row["admissible"]]
    status_counts: dict[str, int] = {}
    for row in rows:
        for prefix, field in (
            ("classical", "classical_run_status"),
            ("neural", "neural_run_status"),
        ):
            key = f"{prefix}:{row[field]}"
            status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "instance_id": instance_id,
        "number_of_piece_types": rows[0]["number_of_piece_types"],
        "repetition_count": len(rows),
        "admissible_repetition_count": len(admissible),
        "status_counts": dict(sorted(status_counts.items())),
        "quality_violation_pair_count": sum(not row["quality_preserved"] for row in rows),
        "objective_classical_bars_median": _median(
            row["objective_classical_bars"] for row in admissible
        ),
        "objective_neural_bars_median": _median(
            row["objective_neural_bars"] for row in admissible
        ),
        "objective_difference_vs_classical_median": _median(
            row["objective_difference_vs_classical"] for row in admissible
        ),
        "classical_runtime_seconds_median": _median(
            row["classical_total_runtime_seconds"] for row in admissible
        ),
        "neural_runtime_seconds_median": _median(
            row["neural_total_runtime_seconds"] for row in admissible
        ),
        "speedup_vs_classical_median": _median(row["speedup_vs_classical"] for row in admissible),
        "classical_peak_memory_bytes_median": _median(
            row["classical_peak_memory_bytes"] for row in admissible
        ),
        "neural_peak_memory_bytes_median": _median(
            row["neural_peak_memory_bytes"] for row in admissible
        ),
        "classical_cg_iterations_median": _median(
            row["classical_cg_iterations"] for row in admissible
        ),
        "neural_cg_iterations_median": _median(
            row["neural_cg_iterations"] for row in admissible
        ),
        "classical_generated_columns_median": _median(
            row["classical_generated_columns"] for row in admissible
        ),
        "neural_generated_columns_median": _median(
            row["neural_generated_columns"] for row in admissible
        ),
        "classical_added_columns_median": _median(
            row["classical_added_columns"] for row in admissible
        ),
        "neural_added_columns_median": _median(row["neural_added_columns"] for row in admissible),
        "classical_final_columns_median": _median(
            row["classical_final_columns"] for row in admissible
        ),
        "neural_final_columns_median": _median(row["neural_final_columns"] for row in admissible),
    }


def _median(values: Any) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return float(numbers[middle])
    return (numbers[middle - 1] + numbers[middle]) / 2
