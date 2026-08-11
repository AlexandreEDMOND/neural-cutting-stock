from itertools import product

import pytest

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    IntegerMasterResult,
    IntegerRestrictedMasterProblem,
    PricingResult,
    RestrictedMasterProblem,
    RMPResult,
    verify_plan,
)


def test_column_generation_adds_shared_pattern_and_converges_exactly() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])

    result = ColumnGeneration(instance).solve()

    assert result.status == "converged"
    assert result.termination_reason == "no_improving_column"
    assert result.columns_added == 1
    assert (1, 1) in result.patterns
    assert result.rmp_result is not None
    assert result.rmp_result.objective_value == 1.5
    assert result.pricing_result is not None
    assert result.pricing_result.reduced_cost >= -1e-9
    assert result.integer_master_result is not None
    assert result.integer_master_result.objective_value == 2
    assert result.integrality_gap == 0.5
    assert result.integer_solution_guarantee == "optimal_over_generated_columns_only"
    assert result.duplicate_columns == 1


def test_column_generation_reports_component_runtimes() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])

    result = ColumnGeneration(instance).solve()

    components = (
        result.master_problem_runtime,
        result.pricing_runtime,
        result.integer_master_runtime,
        result.column_management_runtime,
        result.verification_runtime,
    )
    assert result.total_runtime_seconds > 0
    assert all(runtime > 0 for runtime in components)
    assert result.unattributed_runtime >= 0
    assert result.total_runtime_seconds == pytest.approx(
        sum(components) + result.unattributed_runtime
    )


def test_column_generation_records_rmp_state_for_each_iteration() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])

    result = ColumnGeneration(instance, instance_id="instance-1").solve()

    assert len(result.rmp_states) == result.iterations
    assert [state.iteration_index for state in result.rmp_states] == [1, 2]
    assert all(state.instance_id == "instance-1" for state in result.rmp_states)
    assert result.rmp_states[0].patterns == instance.initial_patterns()
    assert result.rmp_states[-1].result == result.rmp_result
    assert all(state.runtime_seconds > 0 for state in result.rmp_states)


def test_column_generation_converges_for_single_type_with_kerf() -> None:
    instance = CuttingStockInstance(10, 1, [6], [2])

    result = ColumnGeneration(instance).solve()

    assert result.status == "converged"
    assert result.patterns == ((1,),)
    assert result.rmp_result is not None
    assert result.rmp_result.objective_value == 2
    assert result.pricing_result is not None
    assert result.pricing_result.reduced_cost >= -1e-9
    assert result.integer_master_result is not None
    assert result.integer_master_result.objective_value == 2
    assert result.verification is not None
    assert result.verification.feasible
    assert result.verification.kerf_loss == 2
    assert result.verification.trim_loss == 6
    assert result.verification.total_waste == 8


def test_column_generation_counts_duplicate_non_improving_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def duplicate_pricing(self, dual_values: tuple[float, ...]) -> PricingResult:
        return PricingResult(0, (1,), 1.0, 0.0, "optimal")

    monkeypatch.setattr(
        "neural_cutting_stock.solver.column_generation.ExactPricing.solve", duplicate_pricing
    )

    result = ColumnGeneration(instance).solve()

    assert result.status == "converged"
    assert result.duplicate_columns == 1
    assert result.termination_reason == "no_improving_column"


def test_column_generation_rejects_improving_duplicate_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def improving_duplicate(self, dual_values: tuple[float, ...]) -> PricingResult:
        return PricingResult(0, (1,), 2.0, -1.0, "optimal")

    monkeypatch.setattr(
        "neural_cutting_stock.solver.column_generation.ExactPricing.solve",
        improving_duplicate,
    )

    result = ColumnGeneration(instance).solve()

    assert result.status == "solver_error"
    assert result.termination_reason == "improving_duplicate_column"
    assert result.duplicate_columns == 1


def test_column_generation_integer_plan_is_independently_verified() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])

    result = ColumnGeneration(instance).solve()

    assert result.integer_master_result is not None
    verification = verify_plan(
        instance, result.patterns, result.integer_master_result.column_values
    )

    assert verification.feasible
    assert verification.errors == ()
    assert verification.number_of_stock_bars == result.integer_master_result.objective_value
    assert result.verification == verification


def test_column_generation_rejects_an_invalid_integer_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def invalid_integer_master(self: IntegerRestrictedMasterProblem) -> IntegerMasterResult:
        return IntegerMasterResult(0, 1.0, (0,), "success")

    monkeypatch.setattr(IntegerRestrictedMasterProblem, "solve", invalid_integer_master)

    result = ColumnGeneration(instance).solve()

    assert result.status == "invalid_plan"
    assert result.termination_reason == "invalid_plan"
    assert result.verification is not None
    assert not result.verification.feasible


def test_column_generation_is_reproducible_for_same_instance() -> None:
    instance = CuttingStockInstance(11, 1, [2, 3, 5], [2, 2, 1])

    first = ColumnGeneration(instance).solve()
    second = ColumnGeneration(instance).solve()

    assert first.status == second.status
    assert first.patterns == second.patterns
    assert first.rmp_result == second.rmp_result
    assert first.pricing_result == second.pricing_result
    assert first.integer_master_result == second.integer_master_result
    assert first.iterations == second.iterations
    assert first.columns_added == second.columns_added
    assert first.duplicate_columns == second.duplicate_columns
    assert first.termination_reason == second.termination_reason
    assert first.verification == second.verification
    assert first.total_runtime_seconds > 0
    assert second.total_runtime_seconds > 0


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(10, 0, [6, 4], [1, 2]),
        CuttingStockInstance(11, 1, [2, 3, 5], [2, 2, 1]),
        CuttingStockInstance(12, 1, [2, 4, 5], [3, 2, 2]),
    ],
)
def test_converged_lp_matches_master_with_all_small_patterns(
    instance: CuttingStockInstance,
) -> None:
    all_patterns = tuple(
        pattern
        for pattern in product(*(range(demand + 1) for demand in instance.demands))
        if any(pattern) and instance.capacity_used(pattern) <= instance.stock_length
    )

    generated = ColumnGeneration(instance).solve()
    complete_master = RestrictedMasterProblem(instance, all_patterns).solve()

    assert generated.status == "converged"
    assert complete_master.status == 0
    assert generated.rmp_result is not None
    assert generated.pricing_result is not None
    assert generated.pricing_result.reduced_cost >= -1e-9
    assert generated.rmp_result.objective_value == pytest.approx(
        complete_master.objective_value
    )


def test_column_generation_rejects_invalid_reduced_cost_tolerance() -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    for tolerance in [-1.0, float("inf")]:
        try:
            ColumnGeneration(instance, tolerance)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reduced-cost tolerance was accepted")


def test_column_generation_stops_at_iteration_resource_limit() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])

    result = ColumnGeneration(instance, max_iterations=1).solve()

    assert result.status == "limit_reached"
    assert result.termination_reason == "resource_limit"
    assert result.iterations == 1
    assert result.integer_master_result is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_runtime_seconds": 0},
        {"max_runtime_seconds": float("inf")},
        {"max_iterations": 0},
        {"max_iterations": True},
    ],
)
def test_column_generation_rejects_invalid_resource_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ColumnGeneration(CuttingStockInstance(10, 0, [6], [1]), **kwargs)


@pytest.mark.parametrize(
    ("solver_status", "expected_status"),
    [(1, "limit_reached"), (2, "infeasible"), (4, "solver_error")],
)
def test_column_generation_reports_rmp_failure_status(
    monkeypatch: pytest.MonkeyPatch, solver_status: int, expected_status: str
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def failed_rmp(self: RestrictedMasterProblem) -> RMPResult:
        return RMPResult(solver_status, None, (), (), "failure")

    monkeypatch.setattr(RestrictedMasterProblem, "solve", failed_rmp)

    result = ColumnGeneration(instance).solve()

    assert result.status == expected_status
    assert result.termination_reason == "rmp_failed"


def test_column_generation_reports_pricing_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def failed_pricing(self, dual_values: tuple[float, ...]) -> PricingResult:
        return PricingResult(2, (), None, None, "infeasible")

    monkeypatch.setattr(
        "neural_cutting_stock.solver.column_generation.ExactPricing.solve", failed_pricing
    )

    result = ColumnGeneration(instance).solve()

    assert result.status == "infeasible"
    assert result.termination_reason == "pricing_failed"


def test_column_generation_reports_integer_master_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    def limited_integer_master(self: IntegerRestrictedMasterProblem) -> IntegerMasterResult:
        return IntegerMasterResult(1, None, (), "time limit")

    monkeypatch.setattr(IntegerRestrictedMasterProblem, "solve", limited_integer_master)

    result = ColumnGeneration(instance).solve()

    assert result.status == "limit_reached"
    assert result.termination_reason == "integer_master_failed"
