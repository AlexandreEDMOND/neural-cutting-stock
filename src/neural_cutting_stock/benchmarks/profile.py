"""Aggregation and persistence of measured classical baseline profiles."""

import json
import math
from pathlib import Path
from typing import Any

from .schema import BenchmarkRunRecord, RunStatus, SolverMode

PROFILE_SCHEMA_VERSION = "baseline-profile-v1"
PROFILE_COMPONENTS = (
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
)


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
        "run_count": len(records),
        "successful_run_count": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "dominant_component": dominant_component,
        "component_totals_seconds": component_totals,
        "component_shares": component_shares,
        "runs": [record.to_dict() for record in sorted(records, key=lambda item: item.run_id)],
    }
    if output_path is not None:
        _write_profile(output_path, profile)
    return profile


def _finite_sum(values: Any) -> float:
    total = sum(value for value in values if value is not None)
    if not math.isfinite(total):
        raise ValueError("profile timing values must be finite")
    return total


def _write_profile(path: str | Path, profile: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, sort_keys=True)
        stream.write("\n")
