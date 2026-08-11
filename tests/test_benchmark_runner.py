from neural_cutting_stock.benchmarks import (
    ClassicalBenchmarkConfig,
    ClassicalBenchmarkRunner,
    EnvironmentMetadata,
    RunStatus,
    SyntheticInstanceGenerator,
)


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
