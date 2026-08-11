import json

import pytest

from neural_cutting_stock.benchmarks import (
    PROFILE_SCHEMA_VERSION,
    BenchmarkRunRecord,
    ClassicalBenchmarkConfig,
    ClassicalBenchmarkRunner,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
    SyntheticInstanceGenerator,
    profile_classical_runs,
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
