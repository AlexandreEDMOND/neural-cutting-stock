import json
from pathlib import Path

import pytest
from test_benchmark_comparison import _record as base_record

from neural_cutting_stock.benchmarks import (
    FAILURE_ANALYSIS_SCHEMA_VERSION,
    RunStatus,
    SolverMode,
    analyze_campaign_failures,
)
from neural_cutting_stock.visualization.phase4 import load_phase4_runs

ROOT = Path(__file__).parents[1]


def _campaign_record(mode: SolverMode, instance: str, repetition: int, **changes: object):
    return base_record(
        mode,
        f"{mode.value}-{instance}-{repetition}",
        instance_id=instance,
        repetition=repetition,
        config_id=f"config-{mode.value}",
        **changes,
    )


def test_failure_analysis_reports_clean_campaigns_without_filtering() -> None:
    records = [
        _campaign_record(mode, "i-1", repetition)
        for mode in (SolverMode.CLASSICAL, SolverMode.NEURAL)
        for repetition in (0, 1)
    ]

    report = analyze_campaign_failures(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL],
        [r for r in records if r.solver_mode is SolverMode.NEURAL],
    )

    assert (
        report["schema_version"]
        == FAILURE_ANALYSIS_SCHEMA_VERSION
        == ("campaign-failure-analysis-v1")
    )
    assert report["run_count"] == 4 and report["pair_count"] == 2
    for mode, count in (("classical", 2), ("neural", 2)):
        item = report["modes"][mode]
        assert item["run_count"] == count
        assert item["status_counts"] == {
            "optimal_lp_restricted_ip": count,
            "timeout": 0,
            "infeasible": 0,
            "solver_error": 0,
            "invalid_plan": 0,
        }
        assert item["failure_count"] == 0
        assert item["timeout_count"] == 0
        assert item["plan_violation_count"] == 0
        assert item["failure_runs"] == [] and item["plan_violation_runs"] == []
    assert report["modes"]["neural"]["runs_with_exact_fallback"] >= 0
    assert "exact_fallback_calls_total" not in report["modes"]["classical"]
    assert report["pairs"]["admissible_pair_count"] == 2
    assert report["pairs"]["quality_violation_pair_count"] == 0


def test_failure_analysis_keeps_timeout_diagnostics_visible() -> None:
    classical = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1),
    ]
    neural = [
        _campaign_record(SolverMode.NEURAL, "i-1", 0),
        _campaign_record(
            SolverMode.NEURAL,
            "i-1",
            1,
            run_status=RunStatus.TIMEOUT,
            termination_reason="resource_limit",
            error_message="max runtime reached",
            objective_value=None,
            plan_feasible=None,
            total_runtime_seconds=None,
        ),
    ]

    report = analyze_campaign_failures(classical, neural)

    item = report["modes"]["neural"]
    assert item["failure_count"] == 1 and item["timeout_count"] == 1
    assert item["status_counts"]["timeout"] == 1
    failure = item["failure_runs"][0]
    assert failure["run_id"] == "neural-i-1-1"
    assert failure["instance_id"] == "i-1"
    assert failure["repetition"] == 1
    assert failure["run_status"] == "timeout"
    assert failure["termination_reason"] == "resource_limit"
    assert failure["error_message"] == "max runtime reached"
    assert report["pairs"]["pair_count"] == 2
    assert report["pairs"]["admissible_pair_count"] == 1


def test_failure_analysis_flags_infeasible_plans_as_violations() -> None:
    records = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
        _campaign_record(SolverMode.CLASSICAL, "i-1", 1),
        _campaign_record(SolverMode.NEURAL, "i-1", 0),
        _campaign_record(SolverMode.NEURAL, "i-1", 1, plan_feasible=False),
    ]
    neural = [r for r in records if r.solver_mode is SolverMode.NEURAL]

    report = analyze_campaign_failures(
        [r for r in records if r.solver_mode is SolverMode.CLASSICAL], neural
    )

    item = report["modes"]["neural"]
    assert item["plan_violation_count"] == 1
    violation = item["plan_violation_runs"][0]
    assert violation["repetition"] == 1
    assert violation["run_status"] == "optimal_lp_restricted_ip"
    assert report["pairs"]["admissible_pair_count"] == 1


def test_failure_analysis_lists_pairs_beyond_quality_tolerance() -> None:
    classical = [
        _campaign_record(SolverMode.CLASSICAL, "i-1", 0),
        _campaign_record(SolverMode.CLASSICAL, "i-2", 0),
    ]
    neural = [
        _campaign_record(SolverMode.NEURAL, "i-1", 0, objective_value=6.0),
        _campaign_record(SolverMode.NEURAL, "i-2", 0),
    ]

    report = analyze_campaign_failures(classical, neural, quality_tolerance=0.0)

    assert report["pairs"]["quality_violation_pair_count"] == 1
    violation = report["pairs"]["quality_violation_pairs"][0]
    assert violation["instance_id"] == "i-1"
    assert violation["objective_difference_vs_classical"] == 1.0
    assert report["pairs"]["admissible_pair_count"] == 1


def test_failure_analysis_rejects_invalid_tolerance() -> None:
    classical = [_campaign_record(SolverMode.CLASSICAL, "i-1", 0)]
    neural = [_campaign_record(SolverMode.NEURAL, "i-1", 0)]

    with pytest.raises(ValueError, match="quality_tolerance"):
        analyze_campaign_failures(classical, neural, quality_tolerance=-1.0)


def test_phase6_final_campaigns_contain_no_hidden_failures() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    classical = load_phase4_runs(ROOT / "results/phase-6-classical-runs.csv")
    neural = load_phase4_runs(ROOT / "results/phase-6-neural-runs.csv")

    report = analyze_campaign_failures(
        classical, neural, quality_tolerance=config["protocol"]["quality_tolerance_bars"]
    )

    assert report["run_count"] == 72 and report["pair_count"] == 36
    for mode in ("classical", "neural"):
        item = report["modes"][mode]
        assert item["run_count"] == 36
        assert item["status_counts"]["optimal_lp_restricted_ip"] == 36
        assert item["failure_count"] == 0
        assert item["timeout_count"] == 0
        assert item["plan_violation_count"] == 0
    neural_item = report["modes"]["neural"]
    assert neural_item["exact_fallback_calls_total"] >= 36
    assert neural_item["runs_with_exact_fallback"] >= 1
    assert report["modes"]["classical"]["exact_pricing_calls_total"] > 0
    assert report["pairs"]["admissible_pair_count"] == 36
    assert report["pairs"]["quality_violation_pair_count"] == 0
