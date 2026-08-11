import json
from dataclasses import fields

import pytest

from neural_cutting_stock.benchmarks import (
    NEURAL_PROFILE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    BenchmarkRunRecord,
    ClassicalBenchmarkConfig,
    ClassicalBenchmarkRunner,
    EnvironmentMetadata,
    RunStatus,
    SizeClass,
    SolverMode,
    SyntheticInstanceGenerator,
    classify_runtime,
    compare_paired_profiles,
    profile_classical_runs,
    profile_neural_runs,
)


def _record(run_id: str, **changes: object) -> BenchmarkRunRecord:
    values: dict[str, object] = {
        "run_id": run_id,
        "instance_id": "instance-1",
        "solver_mode": SolverMode.CLASSICAL,
        "solver_version": "classical-cg-v1",
        "seed": 17,
        "config_id": "config-1",
        "repetition": 0,
        "environment": EnvironmentMetadata("abc", "3.11", "deps", "machine"),
        "stock_length": 100.0,
        "kerf": 0.0,
        "number_of_piece_types": 3,
        "total_demand": 12,
        "requested_length": 420.0,
        "length_distribution": "uniform",
        "demand_distribution": "uniform",
        "run_status": RunStatus.OPTIMAL_LP_RESTRICTED_IP,
        "master_status": "optimal",
        "pricing_status": "optimal",
        "integer_master_status": "optimal",
        "termination_reason": "no_improving_column",
        "master_problem_runtime": 2.0,
        "pricing_runtime": 5.0,
        "integer_master_runtime": 1.0,
        "column_management_runtime": 1.0,
        "verification_runtime": 0.5,
        "unattributed_runtime": 0.5,
    }
    values.update(changes)
    return BenchmarkRunRecord(**values)


def test_profile_identifies_dominant_component_and_persists_all_runs(tmp_path) -> None:
    output_path = tmp_path / "baseline-profile.json"
    records = (
        _record("run-b"),
        _record("run-a", pricing_runtime=7.0),
        _record(
            "run-c",
            run_status=RunStatus.SOLVER_ERROR,
            error_message="pricing unavailable",
            master_problem_runtime=None,
            pricing_runtime=None,
            integer_master_runtime=None,
            column_management_runtime=None,
            verification_runtime=None,
            unattributed_runtime=None,
        ),
    )

    profile = profile_classical_runs(records, output_path)

    assert profile["profile_schema_version"] == PROFILE_SCHEMA_VERSION
    assert profile["dominant_component"] == "pricing_runtime"
    assert profile["run_count"] == 3
    assert profile["successful_run_count"] == 2
    assert profile["status_counts"] == {"optimal_lp_restricted_ip": 2, "solver_error": 1}
    assert [run["run_id"] for run in profile["runs"]] == ["run-a", "run-b", "run-c"]
    assert json.loads(output_path.read_text()) == profile


def test_profile_rejects_neural_runs() -> None:
    neural = _record("run-neural", solver_mode=SolverMode.NEURAL, model_id="model-v1")

    with pytest.raises(ValueError, match="classical runs only"):
        profile_classical_runs((neural,))


def test_neural_profile_aggregates_end_to_end_and_neural_components(tmp_path) -> None:
    output_path = tmp_path / "neural-profile.json"
    neural = _record(
        "run-neural",
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        total_runtime_seconds=10.0,
        feature_preparation_runtime=1.0,
        neural_inference_runtime=2.0,
        number_of_candidates=12,
        number_of_selected_columns=3,
        exact_fallback_calls=2,
    )
    failed = _record(
        "run-failed",
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        run_status=RunStatus.SOLVER_ERROR,
        error_message="model unavailable",
    )

    profile = profile_neural_runs((neural, failed), output_path)

    assert profile["profile_schema_version"] == NEURAL_PROFILE_SCHEMA_VERSION
    assert profile["successful_run_count"] == 1
    assert profile["status_counts"] == {"optimal_lp_restricted_ip": 1, "solver_error": 1}
    assert profile["component_totals_seconds"]["total_runtime_seconds"] == 10.0
    assert profile["candidate_totals"] == {
        "number_of_candidates": 12,
        "number_of_selected_columns": 3,
        "exact_fallback_calls": 2,
    }
    assert json.loads(output_path.read_text()) == profile


def test_neural_profile_rejects_classical_runs() -> None:
    with pytest.raises(ValueError, match="neural runs only"):
        profile_neural_runs((_record("run-classical"),))


def test_runner_profile_is_reproducible_in_shape_and_status() -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    first = profile_classical_runs(ClassicalBenchmarkRunner(configuration).run())
    second = profile_classical_runs(ClassicalBenchmarkRunner(configuration).run())

    assert first["run_count"] == second["run_count"] == 1
    assert first["status_counts"] == second["status_counts"]
    assert first["runs"][0]["run_id"] == second["runs"][0]["run_id"]
    assert first["dominant_component"] == second["dominant_component"]


def test_size_class_uses_frozen_runtime_boundaries() -> None:
    assert classify_runtime(0.01) is SizeClass.SMALL
    assert classify_runtime(0.015997) is SizeClass.MEDIUM
    assert classify_runtime(0.06385) is SizeClass.LARGE
    assert classify_runtime(0.1433) is SizeClass.XL


def test_runner_assigns_size_class_from_measured_total_runtime() -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.total_runtime_seconds is not None
    assert record.size_class is not None
    assert record.size_class == classify_runtime(record.total_runtime_seconds).value


def test_compare_paired_profiles_aggregates_only_quality_preserved_complete_pairs(tmp_path) -> None:
    classical = _record(
        "classical",
        config_id="shared",
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        total_runtime_seconds=8.0,
        plan_feasible=True,
        objective_value=5.0,
    )
    neural = _record(
        "neural",
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        config_id="shared",
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        total_runtime_seconds=10.0,
        feature_preparation_runtime=1.0,
        neural_inference_runtime=2.0,
        number_of_candidates=2,
        number_of_selected_columns=1,
        exact_fallback_calls=1,
        plan_feasible=True,
        objective_value=5.0,
    )
    failed = _record(
        "failed-classical",
        run_status=RunStatus.SOLVER_ERROR,
        error_message="timeout",
        master_problem_runtime=None,
        pricing_runtime=None,
        integer_master_runtime=None,
        column_management_runtime=None,
        verification_runtime=None,
        unattributed_runtime=None,
        total_runtime_seconds=None,
    )
    failed_neural = _record(
        "failed-neural",
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        run_status=RunStatus.SOLVER_ERROR,
        error_message="timeout",
        master_problem_runtime=None,
        pricing_runtime=None,
        integer_master_runtime=None,
        column_management_runtime=None,
        verification_runtime=None,
        unattributed_runtime=None,
        total_runtime_seconds=None,
    )
    # The helper records use one fixed instance/repetition, so make the second pair unique.
    failed = replace_record(failed, instance_id="instance-2")
    failed_neural = replace_record(failed_neural, instance_id="instance-2")
    output = tmp_path / "paired-profile.json"

    profile = compare_paired_profiles(
        (classical, neural, failed, failed_neural), output_path=output
    )

    assert profile["profile_schema_version"] == "paired-profile-v1"
    assert profile["pair_count"] == 2
    assert profile["quality_preserved_pair_count"] == 1
    assert profile["profile_eligible_pair_count"] == 1
    assert profile["component_medians_seconds"]["classical"]["total_runtime_seconds"] == 8.0
    assert profile["component_medians_seconds"]["neural"]["total_runtime_seconds"] == 10.0
    assert profile["component_medians_seconds"]["neural"]["feature_preparation_runtime"] == 1.0
    assert profile["component_medians_seconds"]["neural"]["neural_inference_runtime"] == 2.0
    assert json.loads(output.read_text()) == profile


def test_compare_paired_profiles_rejects_different_resources() -> None:
    classical = _record("classical", plan_feasible=True, objective_value=5.0)
    neural = _record(
        "neural",
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        plan_feasible=True,
        objective_value=5.0,
        feature_preparation_runtime=1.0,
        neural_inference_runtime=2.0,
        number_of_candidates=2,
        number_of_selected_columns=1,
        exact_fallback_calls=1,
        environment=EnvironmentMetadata("other-commit", "3.11", "deps", "machine"),
    )

    with pytest.raises(ValueError, match="same environment"):
        compare_paired_profiles((classical, neural))


def replace_record(record: BenchmarkRunRecord, **changes: object) -> BenchmarkRunRecord:
    values = {field.name: getattr(record, field.name) for field in fields(record)}
    values.update(changes)
    return BenchmarkRunRecord(**values)
