"""Aggregation and persistence of measured benchmark profiles."""

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from .comparison import compare_paired_runs
from .schema import BenchmarkRunRecord, RunStatus, SolverMode

PROFILE_SCHEMA_VERSION = "baseline-profile-v1"
NEURAL_PROFILE_SCHEMA_VERSION = "neural-profile-v1"
PAIRED_PROFILE_SCHEMA_VERSION = "paired-profile-v1"
SIZE_CLASS_SCHEMA_VERSION = "size-class-v1"
PROFILE_COMPONENTS = (
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
)
NEURAL_PROFILE_COMPONENTS = (
    "total_runtime_seconds",
    "feature_preparation_runtime",
    "neural_inference_runtime",
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
)
PAIRED_PROFILE_COMPONENTS = (
    "total_runtime_seconds",
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
)
NEURAL_ONLY_PROFILE_COMPONENTS = (
    "feature_preparation_runtime",
    "neural_inference_runtime",
)
SIZE_CLASS_RUNTIME_THRESHOLDS_SECONDS = (0.015997, 0.06385, 0.1433)


class SizeClass(StrEnum):
    """Ordered difficulty categories used by benchmark reports."""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    XL = "XL"


def classify_runtime(runtime_seconds: float) -> SizeClass:
    """Classify a completed run using the frozen phase-2 runtime cut points."""

    if not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("runtime_seconds must be finite and non-negative")
    for size_class, threshold in zip(
        (SizeClass.SMALL, SizeClass.MEDIUM, SizeClass.LARGE),
        SIZE_CLASS_RUNTIME_THRESHOLDS_SECONDS,
        strict=True,
    ):
        if runtime_seconds < threshold:
            return size_class
    return SizeClass.XL


def profile_classical_runs(
    records: tuple[BenchmarkRunRecord, ...],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate measured runs and optionally persist the complete profile.

    Failed runs remain in ``runs`` and in the status counts. Only successful runs
    with a complete timing decomposition contribute to bottleneck totals.
    """

    if any(record.solver_mode is not SolverMode.CLASSICAL for record in records):
        raise ValueError("baseline profiling accepts classical runs only")

    status_counts: dict[str, int] = {}
    successful = []
    for record in sorted(records, key=lambda item: item.run_id):
        status = record.run_status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        if (
            record.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP
            and all(getattr(record, component) is not None for component in PROFILE_COMPONENTS)
        ):
            successful.append(record)

    component_totals = {
        component: _finite_sum(getattr(record, component) for record in successful)
        for component in PROFILE_COMPONENTS
    }
    measured_total = sum(component_totals.values())
    component_shares = {
        component: (value / measured_total if measured_total else None)
        for component, value in component_totals.items()
    }
    dominant_component = (
        max(component_totals, key=component_totals.__getitem__) if successful else None
    )
    profile = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "size_class_schema_version": SIZE_CLASS_SCHEMA_VERSION,
        "run_count": len(records),
        "successful_run_count": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "dominant_component": dominant_component,
        "component_totals_seconds": component_totals,
        "component_shares": component_shares,
        "size_class_counts": _size_class_counts(successful),
        "runs": [record.to_dict() for record in sorted(records, key=lambda item: item.run_id)],
    }
    if output_path is not None:
        _write_profile(output_path, profile)
    return profile


def profile_neural_runs(
    records: tuple[BenchmarkRunRecord, ...],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate end-to-end neural timings while retaining every raw run.

    Successful runs contribute only when all neural and solver timing fields are
    present. Incomplete or failed runs remain visible in ``runs`` and counts.
    """

    if any(record.solver_mode is not SolverMode.NEURAL for record in records):
        raise ValueError("neural profiling accepts neural runs only")

    status_counts: dict[str, int] = {}
    successful = []
    required = NEURAL_PROFILE_COMPONENTS + (
        "number_of_candidates",
        "number_of_selected_columns",
        "exact_fallback_calls",
    )
    for record in sorted(records, key=lambda item: item.run_id):
        status = record.run_status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        if record.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP and all(
            getattr(record, field) is not None for field in required
        ):
            successful.append(record)

    component_totals = {
        component: _finite_sum(getattr(record, component) for record in successful)
        for component in NEURAL_PROFILE_COMPONENTS
    }
    profile = {
        "profile_schema_version": NEURAL_PROFILE_SCHEMA_VERSION,
        "size_class_schema_version": SIZE_CLASS_SCHEMA_VERSION,
        "run_count": len(records),
        "successful_run_count": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "component_totals_seconds": component_totals,
        "size_class_counts": _size_class_counts(successful),
        "candidate_totals": {
            field: sum(getattr(record, field) for record in successful)
            for field in (
                "number_of_candidates",
                "number_of_selected_columns",
                "exact_fallback_calls",
            )
        },
        "runs": [record.to_dict() for record in sorted(records, key=lambda item: item.run_id)],
    }
    if output_path is not None:
        _write_profile(output_path, profile)
    return profile


def compare_paired_profiles(
    records: tuple[BenchmarkRunRecord, ...],
    quality_tolerance: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare measured components on identical instances and resources.

    Raw failures and pairs excluded from aggregation remain in ``paired_runs``.
    Only quality-preserved pairs with complete timing decompositions contribute to
    the component medians.
    """

    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    comparisons = compare_paired_runs(records, quality_tolerance)
    by_id = {record.run_id: record for record in records}
    paired_runs = []
    eligible: list[tuple[BenchmarkRunRecord, BenchmarkRunRecord]] = []
    for comparison in comparisons:
        classical = by_id[comparison.classical_run_id]
        neural = by_id[comparison.neural_run_id]
        _validate_pair_resources(classical, neural)
        complete = all(
            getattr(classical, component) is not None
            and getattr(neural, component) is not None
            for component in PAIRED_PROFILE_COMPONENTS
        ) and all(
            getattr(neural, component) is not None for component in NEURAL_ONLY_PROFILE_COMPONENTS
        )
        included = comparison.quality_preserved and complete
        if included:
            eligible.append((classical, neural))
        paired_runs.append(
            {
                "instance_id": comparison.instance_id,
                "repetition": comparison.repetition,
                "classical_run_id": comparison.classical_run_id,
                "neural_run_id": comparison.neural_run_id,
                "objective_difference_vs_classical": comparison.objective_difference_vs_classical,
                "speedup_vs_classical": comparison.speedup_vs_classical,
                "quality_preserved": comparison.quality_preserved,
                "comparable": comparison.comparable,
                "profile_complete": complete,
                "included_in_profile": included,
            }
        )

    component_medians = {
        "classical": {
            component: _median([getattr(record, component) for record, _ in eligible])
            for component in PAIRED_PROFILE_COMPONENTS
        },
        "neural": {
            component: _median([getattr(record, component) for _, record in eligible])
            for component in PAIRED_PROFILE_COMPONENTS + NEURAL_ONLY_PROFILE_COMPONENTS
        },
    }
    profile = {
        "profile_schema_version": PAIRED_PROFILE_SCHEMA_VERSION,
        "quality_tolerance": quality_tolerance,
        "run_count": len(records),
        "pair_count": len(comparisons),
        "comparable_pair_count": sum(item.comparable for item in comparisons),
        "quality_preserved_pair_count": sum(item.quality_preserved for item in comparisons),
        "profile_eligible_pair_count": len(eligible),
        "incomplete_profile_pair_count": sum(
            not item["profile_complete"] for item in paired_runs
        ),
        "component_medians_seconds": component_medians,
        "paired_runs": paired_runs,
    }
    if output_path is not None:
        _write_profile(output_path, profile)
    return profile


def _finite_sum(values: Any) -> float:
    total = sum(value for value in values if value is not None)
    if not math.isfinite(total):
        raise ValueError("profile timing values must be finite")
    return total


def _size_class_counts(records: list[BenchmarkRunRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.size_class is not None:
            counts[record.size_class] = counts.get(record.size_class, 0) + 1
    return dict(sorted(counts.items()))


def _validate_pair_resources(
    classical: BenchmarkRunRecord, neural: BenchmarkRunRecord
) -> None:
    if classical.config_id != neural.config_id:
        raise ValueError("paired runs must use the same config_id")
    if classical.environment != neural.environment:
        raise ValueError("paired runs must use the same environment")
    for field in (
        "seed",
        "stock_length",
        "kerf",
        "number_of_piece_types",
        "total_demand",
        "requested_length",
        "length_distribution",
        "demand_distribution",
    ):
        if getattr(classical, field) != getattr(neural, field):
            raise ValueError(f"paired runs must share instance field {field}")


def _median(values: list[float | None]) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return float(numbers[middle])
    return float((numbers[middle - 1] + numbers[middle]) / 2)


def _write_profile(path: str | Path, profile: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, sort_keys=True)
        stream.write("\n")
