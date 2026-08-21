"""Uncertainty summaries for repeated paired benchmark measurements."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .comparison import compare_paired_runs
from .schema import BenchmarkRunRecord, SolverMode

UNCERTAINTY_SCHEMA_VERSION = "paired-uncertainty-v1"
RUNTIME_UNCERTAINTY_SCHEMA_VERSION = "runtime-uncertainty-v1"


def summarize_repeated_runs(
    records: tuple[BenchmarkRunRecord, ...],
    quality_tolerance: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize repeated paired runs without hiding failed repetitions.

    Statistics are computed per instance before the optional size-class summary.
    The interval is a normal-approximation 95% confidence interval for the mean;
    it is reported only when at least two admissible repetitions are available.
    """

    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    comparisons = compare_paired_runs(records, quality_tolerance)
    by_id = {record.run_id: record for record in records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comparison in comparisons:
        classical = by_id[comparison.classical_run_id]
        neural = by_id[comparison.neural_run_id]
        grouped[comparison.instance_id].append(
            {
                "comparison": comparison,
                "classical": classical,
                "neural": neural,
            }
        )

    instances = []
    for instance_id, pairs in sorted(grouped.items()):
        admissible = [
            pair
            for pair in pairs
            if pair["comparison"].quality_preserved
            and pair["comparison"].speedup_vs_classical is not None
        ]
        classical_values = [pair["classical"].total_runtime_seconds for pair in admissible]
        neural_values = [pair["neural"].total_runtime_seconds for pair in admissible]
        speedups = [pair["comparison"].speedup_vs_classical for pair in admissible]
        instances.append(
            {
                "instance_id": instance_id,
                "size_class": _size_class(pairs),
                "repetition_count": len(pairs),
                "admissible_repetition_count": len(admissible),
                "status_counts": _status_counts(pairs),
                "classical_runtime_seconds": _statistics(classical_values),
                "neural_runtime_seconds": _statistics(neural_values),
                "speedup_vs_classical": _statistics(speedups),
            }
        )

    report = {
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "quality_tolerance_bars": quality_tolerance,
        "run_count": len(records),
        "pair_count": len(comparisons),
        "admissible_pair_count": sum(
            item["admissible_repetition_count"] for item in instances
        ),
        "instances": instances,
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def summarize_runtime_variability(
    records: tuple[BenchmarkRunRecord, ...],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize repeated runtimes for one solver mode, retaining failures."""

    modes = {record.solver_mode for record in records}
    if len(modes) > 1:
        raise ValueError("runtime variability accepts one solver mode only")
    grouped: dict[str, list[BenchmarkRunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.instance_id].append(record)
    instances = [
        {
            "instance_id": instance_id,
            "size_class": _single_size_class(runs),
            "repetition_count": len(runs),
            "status_counts": dict(sorted(_count_statuses(runs).items())),
            "runtime_seconds": _statistics(
                [
                    run.total_runtime_seconds
                    for run in runs
                    if run.run_status.value == "optimal_lp_restricted_ip"
                ]
            ),
        }
        for instance_id, runs in sorted(grouped.items())
    ]
    report = {
        "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
        "solver_mode": next(iter(modes)).value if modes else None,
        "run_count": len(records),
        "instances": instances,
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _statistics(values: list[float | None]) -> dict[str, Any]:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "sample_stddev": None, "ci95": None}
    mean = sum(numbers) / len(numbers)
    middle = len(numbers) // 2
    median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2
    stddev = (
        math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))
        if len(numbers) > 1
        else None
    )
    half_width = 1.96 * stddev / math.sqrt(len(numbers)) if stddev is not None else None
    return {
        "count": len(numbers),
        "mean": mean,
        "median": median,
        "sample_stddev": stddev,
        "ci95": None if half_width is None else [mean - half_width, mean + half_width],
    }


def _size_class(pairs: list[dict[str, Any]]) -> str | None:
    values = {pair["classical"].size_class for pair in pairs}
    return values.pop() if len(values) == 1 else None


def _single_size_class(records: list[BenchmarkRunRecord]) -> str | None:
    values = {record.size_class for record in records}
    return values.pop() if len(values) == 1 else None


def _count_statuses(records: list[BenchmarkRunRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.run_status.value] = counts.get(record.run_status.value, 0) + 1
    return counts


def _status_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        for mode in (SolverMode.CLASSICAL, SolverMode.NEURAL):
            status = pair[mode.value].run_status.value
            key = f"{mode.value}:{status}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
