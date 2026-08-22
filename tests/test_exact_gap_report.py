import csv
import json

import pytest

from neural_cutting_stock.benchmarks import (
    EXACT_GAP_SCHEMA_VERSION,
    BenchmarkRunRecord,
    CorpusBaseline,
    EnvironmentMetadata,
    ExactReferenceStatus,
    ExactReferenceVerification,
    RunStatus,
    SolverMode,
    build_exact_gap_report,
    exact_gap,
    solve_milp_exact_reference,
    write_exact_gap_csv,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import MaximalPatternLimits

_ENVIRONMENT = EnvironmentMetadata("commit-exact-gap", "3.11", "numpy/scipy", "test-machine")
_INSTANCE_ID = "gap-instance"


def _instance() -> CuttingStockInstance:
    return CuttingStockInstance(10.0, 0.0, (3.0, 4.0), (4, 4))


def _record(
    run_id: str, repetition: int = 0, bars: int | None = None, **changes: object
) -> BenchmarkRunRecord:
    values: dict[str, object] = {
        "run_id": run_id,
        "instance_id": _INSTANCE_ID,
        "solver_mode": SolverMode.CLASSICAL,
        "solver_version": "classical-cg-v1",
        "seed": 11,
        "config_id": "config-gap",
        "repetition": repetition,
        "environment": _ENVIRONMENT,
        "stock_length": 10.0,
        "kerf": 0.0,
        "number_of_piece_types": 2,
        "total_demand": 8,
        "requested_length": 28.0,
        "length_distribution": "uniform_integer_v1",
        "demand_distribution": "uniform_integer_v1",
        "run_status": RunStatus.OPTIMAL_LP_RESTRICTED_IP,
        "master_status": "optimal",
        "pricing_status": "optimal",
        "integer_master_status": "optimal",
        "termination_reason": "no_improving_column",
        "objective_value": float(bars) if bars is not None else None,
        "number_of_stock_bars": bars,
        "plan_feasible": bars is not None,
    }
    values.update(changes)
    return BenchmarkRunRecord(**values)


def _corpus(*records: BenchmarkRunRecord) -> list[CorpusBaseline]:
    return [
        CorpusBaseline(
            instance_id=_INSTANCE_ID,
            instance=_instance(),
            source="tests",
            size_class="SMALL",
            family_id=None,
            classical_records=records,
        )
    ]


def _report(corpus: list[CorpusBaseline], **changes: object) -> dict[str, object]:
    values: dict[str, object] = {"environment": _ENVIRONMENT}
    values.update(changes)
    return build_exact_gap_report(corpus, **values)


def test_solve_milp_exact_reference_exposes_the_outcome_for_verification() -> None:
    outcome, record = solve_milp_exact_reference(
        _INSTANCE_ID,
        _instance(),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
    )

    assert outcome is not None
    assert outcome.status == 0
    assert record.status is ExactReferenceStatus.OPTIMAL
    assert record.integer_optimum_bars == 3
    assert outcome.objective_value == record.integer_optimum_bars

    guarded_outcome, guarded_record = solve_milp_exact_reference(
        _INSTANCE_ID,
        _instance(),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        limits=MaximalPatternLimits(max_search_space_size=1),
    )
    assert guarded_outcome is None
    assert guarded_record.status is ExactReferenceStatus.FAILED
    assert guarded_record.error_message


def test_zero_gap_when_baseline_meets_the_exact_optimum() -> None:
    corpus = _corpus(_record("run-0", 0, 3), _record("run-1", 1, 3), _record("run-2", 2, 3))
    report = _report(corpus)
    entry = report["instances"][0]

    assert report["schema_version"] == EXACT_GAP_SCHEMA_VERSION
    assert entry["reference_status"] == "optimal"
    assert entry["integer_optimum_bars"] == 3
    assert entry["verification_passed"] is True
    assert entry["verification_errors"] == []
    assert entry["lp_bound_bars"] <= entry["integer_optimum_bars"] + 1e-9
    assert entry["gap_available"] is True
    assert entry["gap_unavailable_reason"] is None
    assert entry["gap_bars_per_repetition"] == [0, 0, 0]
    assert entry["gap_bars_median"] == 0.0
    assert entry["zero_gap"] is True
    counts = report["counts"]
    assert counts["instance_count"] == 1
    assert counts["optimal_reference_count"] == 1
    assert counts["gap_available_count"] == 1
    assert counts["zero_gap_count"] == 1
    assert counts["positive_gap_count"] == 0


def test_positive_gap_is_measured_per_repetition_with_median() -> None:
    corpus = _corpus(_record("run-0", 0, 4), _record("run-1", 1, 3), _record("run-2", 2, 5))
    report = _report(corpus)
    entry = report["instances"][0]

    assert entry["gap_bars_per_repetition"] == [1, 0, 2]
    assert entry["gap_bars_median"] == 1.0
    assert entry["zero_gap"] is False
    assert entry["baseline_objective_bars_median"] == 4.0
    assert report["counts"]["positive_gap_count"] == 1


def test_non_successful_baseline_runs_stay_visible_without_contributing() -> None:
    timeout = _record(
        "run-timeout",
        0,
        run_status=RunStatus.TIMEOUT,
        error_message="time limit reached",
    )
    report = _report(_corpus(timeout, _record("run-1", 1, 3), _record("run-2", 2, 4)))
    entry = report["instances"][0]

    assert entry["baseline_run_count"] == 3
    assert entry["baseline_optimal_run_count"] == 2
    assert entry["baseline_status_counts"] == {
        "optimal_lp_restricted_ip": 2,
        "timeout": 1,
    }
    assert entry["baseline_non_optimal_run_ids"] == ["run-timeout"]
    assert entry["gap_bars_per_repetition"] == [0, 1]
    assert entry["gap_available"] is True


def test_enumeration_guard_produces_failed_reference_and_nulls_the_gap() -> None:
    report = _report(
        _corpus(_record("run-0", 0, 9)),
        limits=MaximalPatternLimits(max_search_space_size=1),
    )
    entry = report["instances"][0]

    assert entry["reference_status"] == "failed"
    assert entry["reference_method_limits"].endswith("max_search_space_size=1,max_patterns=100000")
    assert entry["pattern_count"] is None
    assert entry["integer_optimum_bars"] is None
    assert entry["verification_passed"] is None
    assert entry["gap_available"] is False
    assert entry["gap_unavailable_reason"] == "reference_not_failed"
    assert entry["gap_bars_median"] is None
    assert entry["zero_gap"] is None
    assert report["counts"]["failed_reference_count"] == 1
    assert report["counts"]["gap_available_count"] == 0


def test_verification_failure_keeps_diagnosis_and_nulls_the_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_verification(*args: object, **kwargs: object) -> ExactReferenceVerification:
        return ExactReferenceVerification(
            instance_id=args[0],
            plan_verification=None,
            lp_bound_bars=None,
            exhaustive_optimum_bars=None,
            errors=("plan check failed: synthetic mismatch",),
            passed=False,
        )

    monkeypatch.setattr(exact_gap, "verify_milp_exact_reference", failing_verification)

    report = _report(_corpus(_record("run-0", 0, 3)))
    entry = report["instances"][0]

    assert entry["verification_passed"] is False
    assert entry["verification_errors"] == ["plan check failed: synthetic mismatch"]
    assert entry["gap_available"] is False
    assert entry["gap_unavailable_reason"] == "reference_verification_failed"
    assert entry["gap_bars_median"] is None
    assert report["counts"]["verification_failure_count"] == 1
    assert report["counts"]["gap_available_count"] == 0


def test_report_is_deterministic_and_time_free() -> None:
    corpus = _corpus(_record("run-0", 0, 3))

    first = _report(corpus)
    second = _report(corpus)

    assert first == second
    dumped = json.dumps(first)
    for forbidden in ("runtime", "_seconds", "timestamp"):
        assert forbidden not in dumped


def test_csv_round_trip_keeps_every_column_explicit(tmp_path) -> None:
    report = _report(
        _corpus(_record("run-0", 0, 4), _record("run-1", 1, 3)),
        exclusions=[{"instance_id": "excluded-b", "reason": "no baseline"}],
    )
    path = tmp_path / "exact-gap.csv"
    write_exact_gap_csv(report, path)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["instance_id"] == _INSTANCE_ID
    assert row["reference_status"] == "optimal"
    assert row["integer_optimum_bars"] == "3"
    assert row["verification_passed"] == "True"
    assert row["gap_bars_per_repetition"] == "1;0"
    assert row["gap_bars_median"] == "0.5"
    assert row["zero_gap"] == "False"
    assert row["reference_error_message"] == ""
    assert {item["instance_id"] for item in report["excluded"]} == {"excluded-b"}
    assert report["counts"]["excluded_instance_count"] == 1


def test_duplicate_instance_ids_are_refused() -> None:
    corpus = _corpus(_record("run-0")) + _corpus(_record("run-1"))

    with pytest.raises(ValueError, match="unique instance_id"):
        build_exact_gap_report(corpus, environment=_ENVIRONMENT)


def test_corpus_baseline_refuses_foreign_or_neural_records() -> None:
    with pytest.raises(ValueError, match="must all be classical runs"):
        _corpus(_record("run-0", solver_mode=SolverMode.NEURAL, model_id="model-v1"))
    with pytest.raises(ValueError, match="must belong to this instance_id"):
        _corpus(_record("run-0", instance_id="other-instance"))
    with pytest.raises(ValueError, match="does not match the materialized instance"):
        _corpus(_record("run-0", stock_length=11.0))
    with pytest.raises(ValueError, match="at least one classical run"):
        _corpus()
