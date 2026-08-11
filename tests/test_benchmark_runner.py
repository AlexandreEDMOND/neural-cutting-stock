import csv

import pytest

from neural_cutting_stock.benchmarks import (
    ClassicalBenchmarkConfig,
    ClassicalBenchmarkRunner,
    EnvironmentMetadata,
    RunStatus,
    SyntheticInstanceGenerator,
    write_raw_runs,
)
from neural_cutting_stock.solver import ColumnGenerationResult


def test_classical_runner_executes_configured_matrix_in_stable_order() -> None:
    first = SyntheticInstanceGenerator(seed=11, number_of_types=2)
    second = SyntheticInstanceGenerator(seed=12, number_of_types=2)
    configuration = ClassicalBenchmarkConfig(
        generators=(first, second),
        repetitions=2,
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    records = ClassicalBenchmarkRunner(configuration).run()

    assert len(records) == 4
    assert [(record.instance_id, record.repetition) for record in records] == [
        (first.instance_id, 0),
        (first.instance_id, 1),
        (second.instance_id, 0),
        (second.instance_id, 1),
    ]
    assert all(record.run_status is RunStatus.OPTIMAL_LP_RESTRICTED_IP for record in records)
    assert all(record.solver_mode.value == "classical" for record in records)
    assert all(record.config_id == configuration.config_id for record in records)


def test_runner_persists_separate_distribution_dimensions() -> None:
    generator = SyntheticInstanceGenerator(
        seed=11,
        length_distribution="short_uniform_v1",
        demand_distribution="high_uniform_v1",
    )
    configuration = ClassicalBenchmarkConfig(
        generators=(generator,),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.length_distribution == "short_uniform_v1"
    assert record.demand_distribution == "high_uniform_v1"


def test_classical_runner_persists_component_runtimes() -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.total_runtime_seconds is not None
    assert record.master_problem_runtime is not None
    assert record.pricing_runtime is not None
    assert record.integer_master_runtime is not None
    assert record.column_management_runtime is not None
    assert record.unattributed_runtime is not None
    assert record.total_runtime_seconds == pytest.approx(
        record.master_problem_runtime
        + record.pricing_runtime
        + record.integer_master_runtime
        + record.column_management_runtime
        + record.verification_runtime
        + record.unattributed_runtime
    )


def test_classical_runner_keeps_solver_failures_as_records(monkeypatch) -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    def fail(_self):
        raise RuntimeError("pricing unavailable")

    monkeypatch.setattr("neural_cutting_stock.benchmarks.runner.ColumnGeneration.solve", fail)

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.run_status is RunStatus.SOLVER_ERROR
    assert record.error_message == "pricing unavailable"
    assert record.termination_reason == "solver_exception"


def test_runner_persists_successes_and_failures_without_filtering(tmp_path, monkeypatch) -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11), SyntheticInstanceGenerator(seed=12)),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )

    def fail_on_second_call(_self):
        if fail_on_second_call.calls:
            raise RuntimeError("pricing unavailable")
        fail_on_second_call.calls = True
        return ColumnGenerationResult(
            status="invalid_plan",
            patterns=((1, 0, 0),),
            rmp_result=None,
            pricing_result=None,
            integer_master_result=None,
            iterations=1,
            columns_added=0,
            duplicate_columns=0,
            termination_reason="invalid_plan",
        )

    fail_on_second_call.calls = False
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.runner.ColumnGeneration.solve", fail_on_second_call
    )
    output_path = tmp_path / "benchmark_runs.csv"

    records = ClassicalBenchmarkRunner(configuration).run(output_path)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(records) == 2
    assert [row["run_status"] for row in rows] == ["invalid_plan", "solver_error"]
    assert rows[0]["error_message"] == "invalid_plan"
    assert rows[1]["error_message"] == "pricing unavailable"


def test_timeout_status_is_retained_as_timeout(monkeypatch) -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.runner.ColumnGeneration.solve",
        lambda _self: ColumnGenerationResult(
            status="limit_reached",
            patterns=(),
            rmp_result=None,
            pricing_result=None,
            integer_master_result=None,
            iterations=0,
            columns_added=0,
            duplicate_columns=0,
            termination_reason="resource_limit",
        ),
    )

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.run_status is RunStatus.TIMEOUT


def test_write_raw_runs_writes_header_for_empty_table(tmp_path) -> None:
    output_path = tmp_path / "empty.csv"

    write_raw_runs(output_path, ())

    with output_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert rows == []
    assert "run_status" in reader.fieldnames
