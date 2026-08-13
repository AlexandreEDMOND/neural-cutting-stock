"""Strict paired comparison of Classical CG and Neural CG raw runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from .schema import BenchmarkRunRecord, RunStatus, SolverMode

OPTIMIZATION_METRIC_SCHEMA_VERSION = "quality-gated-speedup-v1"
FREEZE_DECISION_SCHEMA_VERSION = "validation-freeze-decision-v1"


@dataclass(frozen=True, slots=True)
class PairedRunComparison:
    """Derived metrics for one instance and repetition, including failures."""

    instance_id: str
    repetition: int
    classical_run_id: str
    neural_run_id: str
    objective_difference_vs_classical: float | None
    speedup_vs_classical: float | None
    quality_preserved: bool
    comparable: bool


@dataclass(frozen=True, slots=True)
class OptimizationMetric:
    """End-to-end optimization score with quality as a hard gate."""

    schema_version: str
    score: float
    speedup_vs_classical: float | None
    objective_difference_vs_classical: float | None
    quality_preserved: bool
    comparable: bool


@dataclass(frozen=True, slots=True)
class CandidateFreezeDecision:
    """Validation decision for retaining one end-to-end candidate."""

    schema_version: str
    candidate_id: str
    frozen: bool
    reason: str
    pair_count: int
    classical_total_runtime_seconds: float | None
    candidate_total_runtime_seconds: float | None


def freeze_candidate_on_validation(
    records: Sequence[BenchmarkRunRecord],
    candidate_id: str,
    quality_tolerance: float = 0.0,
) -> CandidateFreezeDecision:
    """Freeze a candidate only after a quality-preserving total-runtime gain.

    Runtime is aggregated across the paired validation records, rather than
    inferred from per-pair speedups. A missing, failed, infeasible, or
    lower-quality pair therefore prevents freezing and remains diagnosable in
    the returned decision.
    """

    if not candidate_id.strip():
        raise ValueError("candidate_id must not be empty")
    comparisons = compare_paired_runs(records, quality_tolerance)
    by_run_id = {record.run_id: record for record in records}
    classical_runtimes = [
        by_run_id[comparison.classical_run_id].total_runtime_seconds
        for comparison in comparisons
    ]
    candidate_runtimes = [
        by_run_id[comparison.neural_run_id].total_runtime_seconds for comparison in comparisons
    ]
    runtimes_available = all(
        runtime is not None and isfinite(runtime) and runtime > 0
        for runtime in (*classical_runtimes, *candidate_runtimes)
    )
    classical_total = sum(classical_runtimes) if runtimes_available else None
    candidate_total = sum(candidate_runtimes) if runtimes_available else None
    quality_preserved = all(comparison.quality_preserved for comparison in comparisons)
    frozen = (
        bool(comparisons)
        and quality_preserved
        and classical_total is not None
        and candidate_total is not None
        and candidate_total < classical_total
    )
    if not quality_preserved:
        reason = "quality_not_preserved"
    elif not runtimes_available:
        reason = "total_runtime_missing_or_invalid"
    elif candidate_total >= classical_total:
        reason = "no_total_runtime_improvement"
    else:
        reason = "strict_total_runtime_improvement"
    return CandidateFreezeDecision(
        schema_version=FREEZE_DECISION_SCHEMA_VERSION,
        candidate_id=candidate_id,
        frozen=frozen,
        reason=reason,
        pair_count=len(comparisons),
        classical_total_runtime_seconds=classical_total,
        candidate_total_runtime_seconds=candidate_total,
    )


def quality_gated_speedup(comparison: PairedRunComparison) -> OptimizationMetric:
    """Return a score that cannot reward faster but lower-quality runs.

    The score is the paired wall-clock speedup only for a comparable pair whose
    independently verified quality is preserved. Missing measurements and
    quality violations score zero while their diagnostics remain available.
    """

    eligible = comparison.comparable and comparison.quality_preserved
    return OptimizationMetric(
        schema_version=OPTIMIZATION_METRIC_SCHEMA_VERSION,
        score=comparison.speedup_vs_classical if eligible else 0.0,
        speedup_vs_classical=comparison.speedup_vs_classical,
        objective_difference_vs_classical=comparison.objective_difference_vs_classical,
        quality_preserved=comparison.quality_preserved,
        comparable=comparison.comparable,
    )


def compare_paired_runs(
    records: Sequence[BenchmarkRunRecord], quality_tolerance: float = 0.0
) -> tuple[PairedRunComparison, ...]:
    """Recompute quality and speedup only from uniquely matched raw records."""

    if not isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    grouped: dict[tuple[str, int], dict[SolverMode, BenchmarkRunRecord]] = {}
    for record in records:
        key = (record.instance_id, record.repetition)
        modes = grouped.setdefault(key, {})
        if record.solver_mode in modes:
            raise ValueError(f"duplicate {record.solver_mode.value} run for {key}")
        modes[record.solver_mode] = record

    comparisons = []
    for (instance_id, repetition), modes in sorted(grouped.items()):
        classical = modes.get(SolverMode.CLASSICAL)
        neural = modes.get(SolverMode.NEURAL)
        if classical is None or neural is None:
            raise ValueError(f"missing paired run for {(instance_id, repetition)}")
        _validate_pair_identity(classical, neural)
        objectives = (
            neural.objective_value - classical.objective_value
            if neural.objective_value is not None and classical.objective_value is not None
            else None
        )
        speedup = (
            classical.total_runtime_seconds / neural.total_runtime_seconds
            if classical.total_runtime_seconds is not None
            and neural.total_runtime_seconds is not None
            and classical.total_runtime_seconds > 0
            and neural.total_runtime_seconds > 0
            else None
        )
        quality_preserved = (
            objectives is not None
            and abs(objectives) <= quality_tolerance
            and classical.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP
            and neural.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP
            and classical.plan_feasible is True
            and neural.plan_feasible is True
        )
        comparisons.append(
            PairedRunComparison(
                instance_id,
                repetition,
                classical.run_id,
                neural.run_id,
                objectives,
                speedup,
                quality_preserved,
                objectives is not None and speedup is not None,
            )
        )
    return tuple(comparisons)


def _validate_pair_identity(
    classical: BenchmarkRunRecord, neural: BenchmarkRunRecord
) -> None:
    """Reject pairs that were not measured on the same instance and resources."""

    if classical.config_id != neural.config_id:
        raise ValueError("paired runs must use the same config_id")
    if classical.environment != neural.environment:
        raise ValueError("paired runs must use the same environment")
    if classical.seed != neural.seed:
        raise ValueError("paired runs must use the same seed")
    instance_fields = (
        "stock_length",
        "kerf",
        "number_of_piece_types",
        "total_demand",
        "requested_length",
        "length_distribution",
        "demand_distribution",
    )
    for field in instance_fields:
        if getattr(classical, field) != getattr(neural, field):
            raise ValueError(f"paired runs must use the same instance data: {field}")
