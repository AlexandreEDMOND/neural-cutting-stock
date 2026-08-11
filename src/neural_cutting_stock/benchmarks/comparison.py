"""Strict paired comparison of Classical CG and Neural CG raw runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from .schema import BenchmarkRunRecord, RunStatus, SolverMode

OPTIMIZATION_METRIC_SCHEMA_VERSION = "quality-gated-speedup-v1"


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
