from neural_cutting_stock.learning import (
    LearnedColumnSelectionPolicy,
    NeuralColumnGeneration,
    PatternScore,
)
from neural_cutting_stock.problem import CuttingStockInstance


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
    assert result.integer_master_result is not None
    assert result.integer_master_result.objective_value == 2
    assert result.verification is not None
    assert result.verification.feasible
    assert result.verification.errors == ()


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
