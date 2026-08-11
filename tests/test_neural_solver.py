from neural_cutting_stock.learning import (
    LearnedColumnSelectionPolicy,
    NeuralColumnGeneration,
    PatternScore,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration


class MisleadingScorer:
    def score(self, state, candidates):
        del state
        return tuple(
            PatternScore(candidate.pattern, 1.0 if candidate.pattern == (0, 1) else 0.0)
            for candidate in candidates
        )


def test_neural_solver_falls_back_to_exact_pricing_and_verifies_plan() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])
    result = NeuralColumnGeneration(
        instance,
        LearnedColumnSelectionPolicy(MisleadingScorer(), candidate_budget=1),
    ).solve()

    assert result.status == "converged"
    assert result.termination_reason == "no_improving_column"
    assert (1, 1) in result.patterns
    assert result.pricing_result is not None
    assert result.pricing_result.reduced_cost >= -1e-9
    assert result.integer_master_result is not None
    assert result.integer_master_result.objective_value == 2
    assert result.verification is not None
    assert result.verification.feasible
    assert result.verification.errors == ()


def test_neural_solver_preserves_classical_quality_with_kerf() -> None:
    instance = CuttingStockInstance(10, 1, [6, 4], [1, 2])
    classical = ColumnGeneration(instance).solve()
    neural = NeuralColumnGeneration(
        instance,
        LearnedColumnSelectionPolicy(MisleadingScorer(), candidate_budget=1),
    ).solve()

    assert classical.status == neural.status == "converged"
    assert classical.integer_master_result is not None
    assert neural.integer_master_result is not None
    assert (
        neural.integer_master_result.objective_value
        == classical.integer_master_result.objective_value
    )
    assert neural.verification is not None and neural.verification.feasible
    assert neural.verification.errors == ()


def test_neural_solver_is_reproducible_for_same_instance_and_policy() -> None:
    instance = CuttingStockInstance(11, 1, [2, 3, 5], [2, 2, 1])
    policy = LearnedColumnSelectionPolicy(MisleadingScorer(), candidate_budget=1)

    first_solver = NeuralColumnGeneration(instance, policy, instance_id="instance-1")
    second_solver = NeuralColumnGeneration(instance, policy, instance_id="instance-1")
    first = first_solver.solve()
    second = second_solver.solve()

    assert first.status == second.status == "converged"
    assert first.patterns == second.patterns
    assert first.rmp_result == second.rmp_result
    assert first.pricing_result == second.pricing_result
    assert first.integer_master_result == second.integer_master_result
    assert first.iterations == second.iterations
    assert first.columns_added == second.columns_added
    assert first.duplicate_columns == second.duplicate_columns
    assert first.termination_reason == second.termination_reason
    assert first.verification == second.verification
    assert first.rmp_states[0].patterns == second.rmp_states[0].patterns
    assert first.rmp_states[-1].patterns == second.rmp_states[-1].patterns
    assert (
        first_solver.runtime_profile.number_of_candidates
        == second_solver.runtime_profile.number_of_candidates
    )
    assert (
        first_solver.runtime_profile.number_of_selected_columns
        == second_solver.runtime_profile.number_of_selected_columns
    )
    assert (
        first_solver.runtime_profile.exact_fallback_calls
        == second_solver.runtime_profile.exact_fallback_calls
    )


def test_neural_solver_records_preparation_inference_and_fallback_work() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])
    solver = NeuralColumnGeneration(
        instance,
        LearnedColumnSelectionPolicy(MisleadingScorer(), candidate_budget=1),
    )

    solver.solve()

    profile = solver.runtime_profile
    assert profile.feature_preparation_runtime >= 0
    assert profile.neural_inference_runtime >= 0
    assert profile.number_of_candidates > 0
    assert profile.number_of_selected_columns > 0
    assert profile.exact_fallback_calls > 0


def test_neural_solver_rejects_invalid_resource_limits() -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])
    policy = LearnedColumnSelectionPolicy(MisleadingScorer())

    import pytest

    with pytest.raises(ValueError, match="max_runtime_seconds must be finite and positive"):
        NeuralColumnGeneration(instance, policy, max_runtime_seconds=0)
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        NeuralColumnGeneration(instance, policy, max_iterations=True)


def test_neural_solver_retains_iteration_timeout_without_claiming_convergence() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])
    result = NeuralColumnGeneration(
        instance,
        LearnedColumnSelectionPolicy(MisleadingScorer()),
        max_iterations=1,
    ).solve()

    assert result.status == "limit_reached"
    assert result.termination_reason == "resource_limit"
    assert result.integer_master_result is None
    assert result.verification is None
