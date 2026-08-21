"""Failure, violation, fallback and timeout analysis over paired campaigns."""

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .comparison import build_paired_comparison
from .generalization import pair_campaign_records
from .schema import BenchmarkRunRecord, RunStatus, SolverMode

FAILURE_ANALYSIS_SCHEMA_VERSION = "campaign-failure-analysis-v1"


def analyze_campaign_failures(
    classical_records: Sequence[BenchmarkRunRecord],
    neural_records: Sequence[BenchmarkRunRecord],
    quality_tolerance: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze failures, violations, fallbacks and timeouts without filtering.

    Every non-successful run, infeasible plan and paired objective beyond the
    declared tolerance stays visible with its raw diagnostics. Fallback counts
    summarize how often the exact pricing guard had to rescue the learned
    selection instead of trusting it.
    """

    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    pairs = pair_campaign_records(classical_records, neural_records)
    report = {
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "quality_tolerance_bars": quality_tolerance,
        "run_count": len(classical_records) + len(neural_records),
        "pair_count": len(pairs),
        "modes": {
            SolverMode.CLASSICAL.value: _mode_analysis(SolverMode.CLASSICAL, classical_records),
            SolverMode.NEURAL.value: _mode_analysis(SolverMode.NEURAL, neural_records),
        },
        "pairs": _pair_analysis(pairs, quality_tolerance),
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _mode_analysis(mode: SolverMode, records: Sequence[BenchmarkRunRecord]) -> dict[str, Any]:
    failure_runs = [
        _diagnostic(record)
        for record in records
        if record.run_status is not RunStatus.OPTIMAL_LP_RESTRICTED_IP
    ]
    plan_violation_runs = [
        _diagnostic(record)
        for record in records
        if record.plan_feasible is False or record.run_status is RunStatus.INVALID_PLAN
    ]
    analysis: dict[str, Any] = {
        "run_count": len(records),
        "status_counts": {
            status.value: sum(record.run_status is status for record in records)
            for status in RunStatus
        },
        "failure_count": len(failure_runs),
        "timeout_count": sum(record.run_status is RunStatus.TIMEOUT for record in records),
        "plan_violation_count": len(plan_violation_runs),
        "exact_pricing_calls_total": _total(record.exact_pricing_calls for record in records),
        "failure_runs": failure_runs,
        "plan_violation_runs": plan_violation_runs,
    }
    if mode is SolverMode.NEURAL:
        analysis["exact_fallback_calls_total"] = _total(
            record.exact_fallback_calls for record in records
        )
        analysis["runs_with_exact_fallback"] = sum(
            (record.exact_fallback_calls or 0) > 0 for record in records
        )
    return analysis


def _pair_analysis(
    pairs: Sequence[tuple[BenchmarkRunRecord, BenchmarkRunRecord]],
    quality_tolerance: float,
) -> dict[str, Any]:
    comparisons = [
        build_paired_comparison(classical, neural, quality_tolerance) for classical, neural in pairs
    ]
    violations = [
        {
            "instance_id": comparison.instance_id,
            "repetition": comparison.repetition,
            "objective_difference_vs_classical": comparison.objective_difference_vs_classical,
        }
        for comparison in comparisons
        if comparison.objective_difference_vs_classical is not None
        and not comparison.quality_preserved
    ]
    return {
        "pair_count": len(comparisons),
        "admissible_pair_count": sum(
            comparison.quality_preserved and comparison.speedup_vs_classical is not None
            for comparison in comparisons
        ),
        "quality_violation_pair_count": len(violations),
        "quality_violation_pairs": violations,
    }


def _diagnostic(record: BenchmarkRunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "instance_id": record.instance_id,
        "solver_mode": record.solver_mode.value,
        "repetition": record.repetition,
        "run_status": record.run_status.value,
        "termination_reason": record.termination_reason,
        "error_message": record.error_message,
    }


def _total(values: Iterable[int | None]) -> int:
    return sum(value for value in values if value is not None)
