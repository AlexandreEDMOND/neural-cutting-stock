import pytest

from neural_cutting_stock.benchmarks import (
    SCHEMA_VERSION,
    BenchmarkRunRecord,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
)


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
