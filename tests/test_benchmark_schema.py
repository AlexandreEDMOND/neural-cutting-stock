import pytest

from neural_cutting_stock.benchmarks import (
    DUAL_SIGN_CONVENTION,
    SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    BenchmarkRunRecord,
    ColumnGenerationTrajectory,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
    TrajectoryIteration,
    TrajectoryMetadata,
    TrajectoryStatus,
)


def _trajectory_metadata(**changes: object) -> TrajectoryMetadata:
    values: dict[str, object] = {
        "trajectory_id": "trajectory-1",
        "instance_id": "instance-1",
        "solver_version": "classical-cg-v1",
        "seed": 17,
        "config_id": "config-1",
        "environment": EnvironmentMetadata("abc123", "3.11.9", "numpy=1.26", "cpu/os/threads"),
        "stock_length": 100.0,
        "kerf": 0.0,
        "piece_lengths": (20.0, 40.0),
        "demands": (3, 2),
        "reduced_cost_tolerance": 1e-9,
        "integrality_tolerance": 1e-9,
        "feasibility_tolerance": 1e-9,
        "dual_type_order": (20.0, 40.0),
        "dual_tolerance": 1e-9,
    }
    values.update(changes)
    return TrajectoryMetadata(**values)


def _record(**changes: object) -> BenchmarkRunRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "instance_id": "instance-1",
        "solver_mode": SolverMode.CLASSICAL,
        "solver_version": "classical-cg-v1",
        "seed": 17,
        "config_id": "config-1",
        "repetition": 0,
        "environment": EnvironmentMetadata("abc123", "3.11.9", "numpy=1.26", "cpu/os/threads"),
        "stock_length": 100.0,
        "kerf": 0.0,
        "number_of_piece_types": 3,
        "total_demand": 12,
        "requested_length": 420.0,
        "length_distribution": "uniform_integer_v1",
        "demand_distribution": "uniform_integer_v1",
        "run_status": RunStatus.OPTIMAL_LP_RESTRICTED_IP,
        "master_status": "optimal",
        "pricing_status": "optimal",
        "integer_master_status": "optimal",
        "termination_reason": "no_improving_column",
    }
    values.update(changes)
    return BenchmarkRunRecord(**values)


def test_record_has_versioned_flat_json_ready_schema() -> None:
    record = _record()

    output = record.to_dict()

    assert output["schema_version"] == SCHEMA_VERSION
    assert output["solver_mode"] == "classical"
    assert output["run_status"] == "optimal_lp_restricted_ip"
    assert output["python_version"] == "3.11.9"
    assert "environment" not in output


def test_neural_fields_are_required_only_for_neural_runs() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _record(solver_mode=SolverMode.NEURAL)

    neural = _record(
        solver_mode=SolverMode.NEURAL,
        model_id="model-v1",
        neural_inference_runtime=0.01,
    )

    assert neural.to_dict()["model_id"] == "model-v1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "benchmark-run-v2"}, "unsupported schema_version"),
        ({"run_status": RunStatus.TIMEOUT}, "error_message"),
        ({"solver_mode": "unknown"}, "solver_mode"),
        ({"stock_length": float("nan")}, "stock_length"),
        ({"solver_mode": SolverMode.CLASSICAL, "model_id": "model-v1"}, "neural-only"),
    ],
)
def test_record_rejects_invalid_contract(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _record(**changes)


def test_trajectory_schema_is_versioned_and_json_ready() -> None:
    trajectory = ColumnGenerationTrajectory(
        _trajectory_metadata(),
        (TrajectoryIteration(1, "optimal", "optimal", 2.0, (0.5, 0.25), 2, 2, 0),),
        TrajectoryStatus.CONVERGED,
        "no_improving_column",
    )

    output = trajectory.to_dict()

    assert output["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert output["metadata"]["instance_id"] == "instance-1"
    assert output["metadata"]["code_commit"] == "abc123"
    assert output["iterations"][0]["iteration_index"] == 1
    assert output["status"] == "converged"
    assert output["metadata"]["dual_type_order"] == (20.0, 40.0)
    assert output["metadata"]["dual_tolerance"] == 1e-9
    assert output["metadata"]["dual_sign_convention"] == DUAL_SIGN_CONVENTION


def test_trajectory_records_duals_in_the_declared_type_order() -> None:
    trajectory = ColumnGenerationTrajectory(
        _trajectory_metadata(),
        (TrajectoryIteration(1, "optimal", dual_values=(0.5, 0.25)),),
        TrajectoryStatus.CONVERGED,
        "no_improving_column",
    )

    assert trajectory.to_dict()["iterations"][0]["dual_values"] == (0.5, 0.25)


def test_trajectory_records_candidate_patterns_and_exact_pricing_result() -> None:
    iteration = TrajectoryIteration(
        1,
        "optimal",
        pricing_status="optimal",
        candidate_patterns=((2, 0), (1, 1)),
        candidate_reduced_costs=(0.25, -0.1),
        best_reduced_cost=-0.1,
    )

    trajectory = ColumnGenerationTrajectory(
        _trajectory_metadata(), (iteration,), TrajectoryStatus.CONVERGED, "no_improving_column"
    )

    output = trajectory.to_dict()["iterations"][0]
    assert output["candidate_patterns"] == ((2, 0), (1, 1))
    assert output["candidate_reduced_costs"] == (0.25, -0.1)
    assert output["pricing_status"] == "optimal"
    assert output["best_reduced_cost"] == -0.1


def test_trajectory_iteration_can_persist_instance_and_rmp_state() -> None:
    iteration = TrajectoryIteration(
        1,
        "optimal",
        instance_id="instance-1",
        rmp_column_values=(1.0, 0.0),
        rmp_pattern_count=2,
    )

    trajectory = ColumnGenerationTrajectory(
        _trajectory_metadata(), (iteration,), TrajectoryStatus.CONVERGED, "no_improving_column"
    )
    output = trajectory.to_dict()["iterations"][0]

    assert output["instance_id"] == "instance-1"
    assert output["rmp_column_values"] == (1.0, 0.0)
    assert output["rmp_pattern_count"] == 2


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _trajectory_metadata(schema_version="cg-trajectory-v1"),
            "unsupported schema_version",
        ),
        (lambda: _trajectory_metadata(piece_lengths=(20.0,)), "same non-zero length"),
        (
            lambda: _trajectory_metadata(dual_type_order=(40.0, 20.0)),
            "dual_type_order",
        ),
        (
            lambda: ColumnGenerationTrajectory(
                _trajectory_metadata(),
                (TrajectoryIteration(1, "optimal", dual_values=(0.5,)),),
                TrajectoryStatus.CONVERGED,
                "no_improving_column",
            ),
            "dual_values must follow",
        ),
        (
            lambda: TrajectoryIteration(1, "optimal", dual_values=(0.5, -0.1)),
            "non-negative",
        ),
        (lambda: TrajectoryIteration(0, "optimal"), "start at 1"),
        (
            lambda: ColumnGenerationTrajectory(
                _trajectory_metadata(),
                (TrajectoryIteration(1, "failed"),),
                "failed",
                "pricing_failed",
            ),
            "error_message",
        ),
        (
            lambda: TrajectoryIteration(
                1, "optimal", candidate_patterns=((1,),), candidate_reduced_costs=None
            ),
            "recorded together",
        ),
        (
            lambda: TrajectoryIteration(
                1,
                "optimal",
                candidate_patterns=((1,),),
                candidate_reduced_costs=(float("nan"),),
            ),
            "candidate_reduced_costs",
        ),
        (
            lambda: TrajectoryIteration(
                1,
                "optimal",
                candidate_patterns=((1.0,),),
                candidate_reduced_costs=(0.0,),
            ),
            "non-negative integers",
        ),
    ],
)
def test_trajectory_schema_rejects_invalid_contract(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
