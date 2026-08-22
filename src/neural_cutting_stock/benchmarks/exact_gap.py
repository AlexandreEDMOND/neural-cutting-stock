"""Integer-quality gap of the classical baseline against exact references.

Every corpus instance carries its persisted classical baseline runs, whose
integer guarantee is `optimal_over_generated_columns_only`, and is compared
to a freshly computed and independently verified `exact-reference-v1` record.
The baseline never depends on this computation; failures, unverified
references and unavailable baselines keep their diagnosis in the report with
a null gap instead of being filtered or repaired silently.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver.maximal_patterns import MaximalPatternLimits

from .exact_reference import (
    ExactReferenceRecord,
    ExactReferenceStatus,
    solve_milp_exact_reference,
)
from .exact_reference_verification import verify_milp_exact_reference
from .schema import BenchmarkRunRecord, EnvironmentMetadata, RunStatus, SolverMode
from .stats import median

EXACT_GAP_SCHEMA_VERSION = "exact-gap-v1"


@dataclass(frozen=True, slots=True)
class CorpusBaseline:
    """One corpus instance together with its persisted classical baseline runs."""

    instance_id: str
    instance: CuttingStockInstance
    source: str
    size_class: str | None
    family_id: str | None
    classical_records: tuple[BenchmarkRunRecord, ...]

    def __post_init__(self) -> None:
        _require_text("instance_id", self.instance_id)
        _require_text("source", self.source)
        if not self.classical_records:
            raise ValueError("a corpus baseline requires at least one classical run")
        for record in self.classical_records:
            if record.solver_mode is not SolverMode.CLASSICAL:
                raise ValueError("baseline records must all be classical runs")
            if record.instance_id != self.instance_id:
                raise ValueError("baseline records must belong to this instance_id")
            _require_same_instance_data(self.instance, record)


def build_exact_gap_report(
    corpus: Sequence[CorpusBaseline],
    *,
    environment: EnvironmentMetadata,
    integrality_tolerance: float = 1e-9,
    feasibility_tolerance: float = 1e-9,
    limits: MaximalPatternLimits | None = None,
    cross_check_with_enumeration: bool = False,
    exclusions: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    """Compute the verified exact reference per instance and the baseline gap.

    The gap of every successful baseline repetition is ``number_of_stock_bars
    minus the certified integer optimum``; it stays null whenever the
    reference is not optimal, fails independent verification, or no baseline
    repetition succeeded. Nothing time-dependent enters the report, so two
    builds over the same inputs are identical.
    """

    if not isinstance(environment, EnvironmentMetadata):
        raise ValueError("environment must be EnvironmentMetadata")
    for name, value in (
        ("integrality_tolerance", integrality_tolerance),
        ("feasibility_tolerance", feasibility_tolerance),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    instance_ids = [spec.instance_id for spec in corpus]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("corpus entries must have unique instance_id values")
    validated_exclusions = [_validated_exclusion(item) for item in exclusions]

    entries = [
        _entry(
            spec,
            environment=environment,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            limits=limits,
            cross_check_with_enumeration=cross_check_with_enumeration,
        )
        for spec in sorted(corpus, key=lambda spec: spec.instance_id)
    ]
    report = {
        "schema_version": EXACT_GAP_SCHEMA_VERSION,
        "integrality_tolerance": integrality_tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "cross_check_with_enumeration": cross_check_with_enumeration,
        "environment": {
            "code_commit": environment.code_commit,
            "python_version": environment.python_version,
            "dependency_versions": environment.dependency_versions,
            "hardware_id": environment.hardware_id,
        },
        "counts": {
            "instance_count": len(entries),
            "excluded_instance_count": len(validated_exclusions),
            "optimal_reference_count": sum(
                entry["reference_status"] == ExactReferenceStatus.OPTIMAL.value
                for entry in entries
            ),
            "lower_bound_only_reference_count": sum(
                entry["reference_status"] == ExactReferenceStatus.LOWER_BOUND_ONLY.value
                for entry in entries
            ),
            "failed_reference_count": sum(
                entry["reference_status"] == ExactReferenceStatus.FAILED.value
                for entry in entries
            ),
            "verification_failure_count": sum(
                entry["verification_passed"] is False for entry in entries
            ),
            "gap_available_count": sum(entry["gap_available"] for entry in entries),
            "zero_gap_count": sum(bool(entry["zero_gap"]) for entry in entries),
            "positive_gap_count": sum(
                entry["gap_available"] and entry["gap_bars_median"] > 0 for entry in entries
            ),
        },
        "instances": entries,
        "excluded": sorted(validated_exclusions, key=lambda item: item["instance_id"]),
    }
    return report


def write_exact_gap_csv(report: dict[str, Any], path: str | Path) -> None:
    """Write one flat row per corpus instance; every failure stays explicit."""

    fieldnames = (
        "instance_id",
        "source",
        "size_class",
        "family_id",
        "stock_length",
        "kerf",
        "number_of_piece_types",
        "total_demand",
        "reference_method",
        "reference_status",
        "pattern_count",
        "integer_optimum_bars",
        "certified_lower_bound_bars",
        "lp_bound_bars",
        "exhaustive_optimum_bars",
        "verification_passed",
        "reference_error_message",
        "baseline_run_count",
        "baseline_optimal_run_count",
        "baseline_objective_bars_median",
        "gap_available",
        "gap_unavailable_reason",
        "gap_bars_per_repetition",
        "gap_bars_median",
        "zero_gap",
        "detail",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for entry in report["instances"]:
            writer.writerow({name: _csv_cell(entry, name) for name in fieldnames})


def _entry(
    spec: CorpusBaseline,
    *,
    environment: EnvironmentMetadata,
    integrality_tolerance: float,
    feasibility_tolerance: float,
    limits: MaximalPatternLimits | None,
    cross_check_with_enumeration: bool,
) -> dict[str, Any]:
    ordered = sorted(spec.classical_records, key=lambda record: (record.repetition, record.run_id))
    status_counts: dict[str, int] = {}
    for record in ordered:
        key = record.run_status.value
        status_counts[key] = status_counts.get(key, 0) + 1
    successful = [
        record for record in ordered if record.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP
    ]
    missing_objective_run_ids = [
        record.run_id
        for record in successful
        if record.number_of_stock_bars is None
    ]
    successful_run_ids = {record.run_id for record in successful}

    outcome, reference = solve_milp_exact_reference(
        spec.instance_id,
        spec.instance,
        environment=environment,
        integrality_tolerance=integrality_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        limits=limits,
    )
    verification = None
    verification_errors: list[str] = []
    if outcome is not None and reference.status is ExactReferenceStatus.OPTIMAL:
        verification = verify_milp_exact_reference(
            spec.instance_id,
            spec.instance,
            outcome,
            reference,
            limits=limits,
            cross_check_with_enumeration=cross_check_with_enumeration,
        )
        verification_errors = list(verification.errors)

    if reference.integer_optimum_bars is not None:
        gaps = [
            record.number_of_stock_bars - reference.integer_optimum_bars
            for record in successful
            if record.number_of_stock_bars is not None
        ]
    else:
        gaps = []
    reason = _unavailability_reason(
        reference, verification_errors, successful, missing_objective_run_ids
    )
    gap_available = reason is None
    return {
        "instance_id": spec.instance_id,
        "source": spec.source,
        "size_class": spec.size_class,
        "family_id": spec.family_id,
        "stock_length": spec.instance.stock_length,
        "kerf": spec.instance.kerf,
        "number_of_piece_types": spec.instance.number_of_types,
        "total_demand": sum(spec.instance.demands),
        "reference_method": reference.reference_method.value,
        "reference_status": reference.status.value,
        "reference_method_limits": reference.method_limits,
        "pattern_count": outcome.number_of_patterns if outcome is not None else None,
        "integer_optimum_bars": reference.integer_optimum_bars,
        "certified_lower_bound_bars": reference.certified_lower_bound_bars,
        "lp_bound_bars": verification.lp_bound_bars if verification else None,
        "exhaustive_optimum_bars": (
            verification.exhaustive_optimum_bars if verification else None
        ),
        "verification_passed": verification.passed if verification else None,
        "verification_errors": verification_errors,
        "reference_error_message": reference.error_message,
        "baseline_run_count": len(ordered),
        "baseline_optimal_run_count": len(successful),
        "baseline_status_counts": dict(sorted(status_counts.items())),
        "baseline_non_optimal_run_ids": [
            record.run_id for record in ordered if record.run_id not in successful_run_ids
        ],
        "baseline_missing_objective_run_ids": missing_objective_run_ids,
        "baseline_objective_bars": [
            record.number_of_stock_bars
            for record in successful
            if record.number_of_stock_bars is not None
        ],
        "baseline_objective_bars_median": median(
            record.number_of_stock_bars for record in successful
        ),
        "gap_available": gap_available,
        "gap_unavailable_reason": reason,
        "gap_bars_per_repetition": gaps if gap_available else None,
        "gap_bars_median": median(gaps) if gap_available else None,
        "zero_gap": bool(gaps) and all(gap == 0 for gap in gaps) if gap_available else None,
    }


def _unavailability_reason(
    reference: ExactReferenceRecord,
    verification_errors: Sequence[str],
    successful: Sequence[BenchmarkRunRecord],
    missing_objective_run_ids: Sequence[str],
) -> str | None:
    if reference.status is not ExactReferenceStatus.OPTIMAL:
        return f"reference_not_{reference.status.value}"
    if verification_errors:
        return "reference_verification_failed"
    if not successful:
        return "no_successful_baseline_run"
    if missing_objective_run_ids:
        return "successful_baseline_runs_without_bar_count"
    return None


def _validated_exclusion(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError("each exclusion must be a mapping")
    for name in ("instance_id", "reason"):
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"each exclusion requires a non-empty {name}")
    source = item.get("source", "")
    if not isinstance(source, str):
        raise ValueError("exclusion source must be a string when present")
    return {"instance_id": item["instance_id"], "source": source, "reason": item["reason"]}


def _require_same_instance_data(instance: CuttingStockInstance, record: BenchmarkRunRecord) -> None:
    materialized_requested_length = sum(
        length * demand
        for length, demand in zip(instance.piece_lengths, instance.demands, strict=True)
    )
    mismatch = (
        record.stock_length != instance.stock_length
        or record.kerf != instance.kerf
        or record.number_of_piece_types != instance.number_of_types
        or record.total_demand != sum(instance.demands)
        or record.requested_length != materialized_requested_length
    )
    if mismatch:
        raise ValueError("baseline record data does not match the materialized instance")


def _csv_cell(entry: dict[str, Any], name: str) -> Any:
    if name == "detail":
        parts = list(entry["verification_errors"])
        if entry["reference_error_message"]:
            parts.append(entry["reference_error_message"])
        return "; ".join(parts)
    value = entry[name]
    if name == "gap_bars_per_repetition":
        return "" if value is None else ";".join(str(gap) for gap in value)
    return "" if value is None else value


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


__all__ = [
    "EXACT_GAP_SCHEMA_VERSION",
    "CorpusBaseline",
    "build_exact_gap_report",
    "write_exact_gap_csv",
]
