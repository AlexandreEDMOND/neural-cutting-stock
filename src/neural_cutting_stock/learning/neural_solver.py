"""Neural column-generation orchestration with an exact convergence guard."""

from dataclasses import dataclass
from time import perf_counter

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration, ColumnGenerationResult

from .candidates import deterministic_candidate_pool
from .interfaces import ColumnSelectionPolicy, PricingState


@dataclass(frozen=True, slots=True)
class NeuralRuntimeProfile:
    """Measured work performed by the neural selection layer."""

    feature_preparation_runtime: float = 0.0
    neural_inference_runtime: float = 0.0
    number_of_candidates: int = 0
    number_of_selected_columns: int = 0
    exact_fallback_calls: int = 0


class NeuralColumnGeneration:
    """Use learned selection only before exact pricing and plan verification."""

    def __init__(
        self,
        instance: CuttingStockInstance,
        policy: ColumnSelectionPolicy,
        candidate_budget: int | None = None,
        reduced_cost_tolerance: float = 1e-9,
        max_runtime_seconds: float | None = None,
        max_iterations: int | None = None,
        instance_id: str | None = None,
    ) -> None:
        if candidate_budget is not None and (
            isinstance(candidate_budget, bool)
            or not isinstance(candidate_budget, int)
            or candidate_budget < 1
        ):
            raise ValueError("candidate_budget must be a positive integer when present")
        self.instance = instance
        self.policy = policy
        self.candidate_budget = candidate_budget
        self.reduced_cost_tolerance = reduced_cost_tolerance
        self.max_runtime_seconds = max_runtime_seconds
        self.max_iterations = max_iterations
        self.instance_id = instance_id or "neural-instance"
        self.runtime_profile = NeuralRuntimeProfile()

    def solve(self) -> ColumnGenerationResult:
        """Solve with learned candidate selection and exact final pricing."""

        iteration_index = 0
        feature_preparation_runtime = 0.0
        neural_inference_runtime = 0.0
        number_of_candidates = 0
        number_of_selected_columns = 0
        exact_fallback_calls = 0

        def select_columns(instance, current_patterns, dual_values):
            nonlocal iteration_index, feature_preparation_runtime
            nonlocal neural_inference_runtime, number_of_candidates
            nonlocal number_of_selected_columns, exact_fallback_calls
            iteration_index += 1
            preparation_started = perf_counter()
            candidates = deterministic_candidate_pool(
                instance,
                dual_values,
                current_patterns,
                self.candidate_budget,
            )
            state = PricingState(
                self.instance_id,
                iteration_index,
                instance.stock_length,
                instance.kerf,
                tuple(instance.piece_lengths),
                tuple(instance.demands),
                tuple(dual_values),
                current_patterns,
                reduced_cost_tolerance=self.reduced_cost_tolerance,
            )
            feature_preparation_runtime += perf_counter() - preparation_started
            number_of_candidates += len(candidates)
            inference_started = perf_counter()
            decision = self.policy.select(state, candidates)
            neural_inference_runtime += perf_counter() - inference_started
            number_of_selected_columns += len(decision.selected_patterns)
            candidate_by_pattern = {candidate.pattern: candidate for candidate in candidates}
            selected_patterns = tuple(
                pattern
                for pattern in decision.selected_patterns
                if candidate_by_pattern[pattern].reduced_cost < -self.reduced_cost_tolerance
            )
            if not selected_patterns:
                exact_fallback_calls += 1
            return selected_patterns

        result = ColumnGeneration(
            self.instance,
            reduced_cost_tolerance=self.reduced_cost_tolerance,
            max_runtime_seconds=self.max_runtime_seconds,
            max_iterations=self.max_iterations,
            instance_id=self.instance_id,
            candidate_selector=select_columns,
        ).solve()
        self.runtime_profile = NeuralRuntimeProfile(
            feature_preparation_runtime,
            neural_inference_runtime,
            number_of_candidates,
            number_of_selected_columns,
            exact_fallback_calls,
        )
        return result


__all__ = ["NeuralColumnGeneration", "NeuralRuntimeProfile"]
