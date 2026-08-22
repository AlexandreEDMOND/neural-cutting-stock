"""Family and size-class breakdown of the persisted exact-gap report.

The breakdown consumes a validated `exact-gap-v1` report and aggregates its
per-instance rows by `size_class` and `family_id`. Instances whose gap stays
unavailable keep their diagnosis visible inside their groups and never feed
the margin counts; excluded instances are reported as totals only, since no
reference was computed for them.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .exact_gap import EXACT_GAP_SCHEMA_VERSION
from .final_manifest import SIZE_CLASSES
from .stats import median

EXACT_GAP_BREAKDOWN_SCHEMA_VERSION = "exact-gap-breakdown-v1"


def build_exact_gap_breakdown(report: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate a persisted exact-gap report into family and size tables.

    Every group reports where the baseline loses bars against the certified
    integer optimum (positive margin) and where it does not (zero margin).
    The function is pure: identical inputs produce an identical breakdown.
    """

    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    if report.get("schema_version") != EXACT_GAP_SCHEMA_VERSION:
        raise ValueError("breakdown requires an exact-gap-v1 report")
    instances = report.get("instances")
    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
        raise ValueError("report must carry an instance sequence")

    return {
        "schema_version": EXACT_GAP_BREAKDOWN_SCHEMA_VERSION,
        "source_schema_version": report["schema_version"],
        "totals": _totals(report),
        "by_size_class": _groups(instances, lambda entry: entry["size_class"]),
        "by_family": _groups(instances, lambda entry: entry["family_id"]),
    }


def _totals(report: Mapping[str, Any]) -> dict[str, Any]:
    counts = report["counts"]
    return {
        "instance_count": counts["instance_count"],
        "excluded_instance_count": counts["excluded_instance_count"],
        "optimal_reference_count": counts["optimal_reference_count"],
        "lower_bound_only_reference_count": counts["lower_bound_only_reference_count"],
        "failed_reference_count": counts["failed_reference_count"],
        "verification_failure_count": counts["verification_failure_count"],
        "gap_available_count": counts["gap_available_count"],
        "zero_gap_count": counts["zero_gap_count"],
        "positive_gap_count": counts["positive_gap_count"],
    }


def _group(key: str | None, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [entry for entry in entries if entry["gap_available"]]
    positive = sorted(
        (entry for entry in available if entry["gap_bars_median"] > 0),
        key=lambda entry: (-entry["gap_bars_median"], entry["instance_id"]),
    )
    reasons: dict[str, int] = defaultdict(int)
    for entry in entries:
        if not entry["gap_available"]:
            reasons[entry["gap_unavailable_reason"]] += 1
    return {
        "key": key,
        "instance_count": len(entries),
        "gap_available_count": len(available),
        "zero_gap_count": sum(bool(entry["zero_gap"]) for entry in available),
        "positive_gap_count": len(positive),
        "gap_unavailable_reasons": dict(sorted(reasons.items())),
        "max_gap_bars_median": max((entry["gap_bars_median"] for entry in available), default=None),
        "median_integer_optimum_bars": median(entry["integer_optimum_bars"] for entry in available),
        "piece_type_counts": dict(
            sorted(
                {
                    entry["number_of_piece_types"]: sum(
                        1
                        for other in entries
                        if other["number_of_piece_types"] == entry["number_of_piece_types"]
                    )
                    for entry in entries
                }.items()
            )
        ),
        "positive_instances": [
            {
                "instance_id": entry["instance_id"],
                "size_class": entry["size_class"],
                "family_id": entry["family_id"],
                "number_of_piece_types": entry["number_of_piece_types"],
                "integer_optimum_bars": entry["integer_optimum_bars"],
                "baseline_objective_bars_median": entry["baseline_objective_bars_median"],
                "gap_bars_median": entry["gap_bars_median"],
                "gap_bars_per_repetition": list(entry["gap_bars_per_repetition"]),
            }
            for entry in positive
        ],
    }


def _groups(instances: Sequence[Mapping[str, Any]], key_of: Any) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in instances:
        grouped[key_of(entry)].append(entry)
    known = [key for key in SIZE_CLASSES if key in grouped]
    unknown = sorted(
        (key for key in grouped if key not in SIZE_CLASSES),
        key=lambda key: (key is None, str(key)),
    )
    return [_group(key, grouped[key]) for key in [*known, *unknown]]


__all__ = [
    "EXACT_GAP_BREAKDOWN_SCHEMA_VERSION",
    "build_exact_gap_breakdown",
]
