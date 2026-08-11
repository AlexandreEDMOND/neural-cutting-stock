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


def test_runner_identity_and_run_ids_do_not_depend_on_generator_order() -> None:
    first = SyntheticInstanceGenerator(seed=11, number_of_types=2)
    second = SyntheticInstanceGenerator(seed=12, number_of_types=2)
    environment = EnvironmentMetadata("commit", "3.11", "deps", "machine")
    forward = ClassicalBenchmarkConfig(generators=(first, second), environment=environment)
    reverse = ClassicalBenchmarkConfig(generators=(second, first), environment=environment)

    forward_records = ClassicalBenchmarkRunner(forward).run()
    reverse_records = ClassicalBenchmarkRunner(reverse).run()

    assert forward.config_id == reverse.config_id
    assert {record.run_id for record in forward_records} == {
        record.run_id for record in reverse_records
    }


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
    assert record.peak_memory_bytes is not None
    assert record.exact_pricing_calls is not None
    assert record.exact_pricing_calls > 0
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
    assert [row["run_status"] for row in rows] == [
        record.run_status.value for record in sorted(records, key=lambda item: item.run_id)
    ]
    rows_by_error = {row["error_message"]: row for row in rows}
    assert rows_by_error["invalid_plan"]["run_status"] == "invalid_plan"
    assert rows_by_error["pricing unavailable"]["run_status"] == "solver_error"


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


def test_runner_passes_iteration_limit_and_persists_timeout(monkeypatch) -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11, number_of_types=3),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        max_cg_iterations=1,
    )

    def limited_solve(self):
        assert self.max_iterations == 1
        return ColumnGenerationResult(
            status="limit_reached",
            patterns=(),
            rmp_result=None,
            pricing_result=None,
            integer_master_result=None,
            iterations=1,
            columns_added=0,
            duplicate_columns=0,
            termination_reason="resource_limit",
        )

    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.runner.ColumnGeneration.solve", limited_solve
    )

    record = ClassicalBenchmarkRunner(configuration).run()[0]

    assert record.run_status is RunStatus.TIMEOUT
    assert record.termination_reason == "resource_limit"
    assert record.error_message == "resource_limit"
    assert record.number_of_cg_iterations == 1


def test_resource_limits_are_part_of_campaign_identity() -> None:
    generator = SyntheticInstanceGenerator(seed=11)
    environment = EnvironmentMetadata("commit", "3.11", "deps", "machine")

    unlimited = ClassicalBenchmarkConfig((generator,), environment=environment)
    limited = ClassicalBenchmarkConfig(
        (generator,), environment=environment, max_runtime_seconds=1.0
    )

    assert unlimited.config_id != limited.config_id


def test_paired_config_validates_shared_budget_and_resource_limits() -> None:
    from neural_cutting_stock.benchmarks import PairedBenchmarkConfig

    generator = SyntheticInstanceGenerator(seed=11)
    environment = EnvironmentMetadata("commit", "3.11", "deps", "machine")
    policy = object()

    with pytest.raises(ValueError, match="candidate_budget must be a positive integer"):
        PairedBenchmarkConfig((generator,), environment, policy, "model", candidate_budget=0)
    with pytest.raises(ValueError, match="max_runtime_seconds must be finite and positive"):
        PairedBenchmarkConfig((generator,), environment, policy, "model", max_runtime_seconds=0)
    with pytest.raises(ValueError, match="max_cg_iterations must be a positive integer"):
        PairedBenchmarkConfig((generator,), environment, policy, "model", max_cg_iterations=True)


def test_write_raw_runs_writes_header_for_empty_table(tmp_path) -> None:
    output_path = tmp_path / "empty.csv"

    write_raw_runs(output_path, ())

    with output_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert rows == []
    assert "run_status" in reader.fieldnames


def test_write_raw_runs_is_independent_of_execution_order(tmp_path) -> None:
    configuration = ClassicalBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=11), SyntheticInstanceGenerator(seed=12)),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
    )
    records = ClassicalBenchmarkRunner(configuration).run()
    forward_path = tmp_path / "forward.csv"
    reverse_path = tmp_path / "reverse.csv"

    write_raw_runs(forward_path, records)
    write_raw_runs(reverse_path, tuple(reversed(records)))

    assert forward_path.read_bytes() == reverse_path.read_bytes()
