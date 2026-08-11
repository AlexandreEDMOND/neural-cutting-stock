from dataclasses import replace

import pytest

from neural_cutting_stock.benchmarks import (
    BenchmarkRunRecord,
    EnvironmentMetadata,
    PairedBenchmarkConfig,
    PairedBenchmarkRunner,
    RunStatus,
    SolverMode,
    SyntheticInstanceGenerator,
    compare_paired_runs,
    quality_gated_speedup,
)
from neural_cutting_stock.learning import LearnedColumnSelectionPolicy, LinearColumnScoringModel


def _record(mode: SolverMode, run_id: str, **changes: object) -> BenchmarkRunRecord:
    values: dict[str, object] = {
        "run_id": run_id,
        "instance_id": "instance-1",
        "solver_mode": mode,
        "solver_version": f"{mode.value}-cg-v1",
        "seed": 12,
        "config_id": "config-1",
        "repetition": 0,
        "environment": EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        "stock_length": 100.0,
        "kerf": 0.0,
        "number_of_piece_types": 2,
        "total_demand": 10,
        "requested_length": 400.0,
        "length_distribution": "uniform_v1",
        "demand_distribution": "uniform_v1",
        "run_status": RunStatus.OPTIMAL_LP_RESTRICTED_IP,
        "master_status": "optimal",
        "pricing_status": "optimal",
        "integer_master_status": "optimal",
        "termination_reason": "no_improving_column",
        "objective_value": 5.0,
        "total_runtime_seconds": 2.0 if mode is SolverMode.CLASSICAL else 1.0,
        "plan_feasible": True,
    }
    if mode is SolverMode.NEURAL:
        values["model_id"] = "model-v1"
    values.update(changes)
    return BenchmarkRunRecord(**values)


def test_comparison_recomputes_quality_and_only_reports_admissible_speedup() -> None:
    comparisons = compare_paired_runs(
        (_record(SolverMode.CLASSICAL, "classical"), _record(SolverMode.NEURAL, "neural"))
    )

    assert comparisons[0].objective_difference_vs_classical == 0.0
    assert comparisons[0].speedup_vs_classical == 2.0
    assert comparisons[0].quality_preserved
    assert comparisons[0].comparable


def test_quality_degradation_keeps_pair_but_excludes_speedup() -> None:
    records = (
        _record(SolverMode.CLASSICAL, "classical"),
        _record(SolverMode.NEURAL, "neural", objective_value=6.0),
    )

    comparison = compare_paired_runs(records)[0]

    assert comparison.objective_difference_vs_classical == 1.0
    assert comparison.speedup_vs_classical == 2.0
    assert not comparison.quality_preserved


def test_quality_gated_speedup_rewards_only_admissible_end_to_end_gain() -> None:
    comparison = compare_paired_runs(
        (_record(SolverMode.CLASSICAL, "classical"), _record(SolverMode.NEURAL, "neural"))
    )[0]

    metric = quality_gated_speedup(comparison)

    assert metric.schema_version == "quality-gated-speedup-v1"
    assert metric.score == 2.0
    assert metric.quality_preserved
    assert metric.comparable


def test_quality_gated_speedup_zeroes_quality_violation_but_keeps_diagnostics() -> None:
    comparison = compare_paired_runs(
        (
            _record(SolverMode.CLASSICAL, "classical"),
            _record(SolverMode.NEURAL, "neural", objective_value=6.0),
        )
    )[0]

    metric = quality_gated_speedup(comparison)

    assert metric.score == 0.0
    assert metric.speedup_vs_classical == 2.0
    assert metric.objective_difference_vs_classical == 1.0
    assert not metric.quality_preserved


def test_comparison_rejects_missing_or_duplicate_pair() -> None:
    classical = _record(SolverMode.CLASSICAL, "classical")
    neural = _record(SolverMode.NEURAL, "neural")
    with pytest.raises(ValueError, match="missing paired"):
        compare_paired_runs((classical,))
    with pytest.raises(ValueError, match="duplicate neural"):
        compare_paired_runs((classical, neural, replace(neural, run_id="neural-2")))


def test_paired_runner_executes_both_modes_on_the_same_instance() -> None:
    configuration = PairedBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=12, number_of_types=2),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        policy=LearnedColumnSelectionPolicy(LinearColumnScoringModel((0.0,) * 32, 0.0), 1),
        model_id="model-v1",
    )

    records = PairedBenchmarkRunner(configuration).run()

    assert {record.solver_mode for record in records} == {SolverMode.CLASSICAL, SolverMode.NEURAL}
    assert len({record.instance_id for record in records}) == 1
    neural = next(record for record in records if record.solver_mode is SolverMode.NEURAL)
    assert neural.objective_difference_vs_classical == 0.0
    assert neural.speedup_vs_classical is not None
    assert neural.feature_preparation_runtime is not None
    assert neural.neural_inference_runtime is not None
    assert neural.number_of_candidates is not None
    assert neural.exact_fallback_calls is not None


def test_paired_runner_retains_neural_model_errors_as_raw_failures() -> None:
    class FailingPolicy:
        def select(self, state, candidates):
            del state, candidates
            raise RuntimeError("model inference failed")

    configuration = PairedBenchmarkConfig(
        generators=(SyntheticInstanceGenerator(seed=12, number_of_types=2),),
        environment=EnvironmentMetadata("commit", "3.11", "deps", "machine"),
        policy=FailingPolicy(),
        model_id="unavailable-model-v1",
    )

    records = PairedBenchmarkRunner(configuration).run()

    neural = next(record for record in records if record.solver_mode is SolverMode.NEURAL)
    assert neural.run_status is RunStatus.SOLVER_ERROR
    assert neural.termination_reason == "solver_exception"
    assert neural.error_message == "model inference failed"
    assert neural.model_id == "unavailable-model-v1"
    assert neural.objective_difference_vs_classical is None
    assert neural.speedup_vs_classical is None
