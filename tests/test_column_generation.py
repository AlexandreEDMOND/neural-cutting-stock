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
    assert result.duplicate_columns == 0


def test_column_generation_is_reproducible_for_same_instance() -> None:
    instance = CuttingStockInstance(11, 1, [2, 3, 5], [2, 2, 1])

    first = ColumnGeneration(instance).solve()
    second = ColumnGeneration(instance).solve()

    assert first == second


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(10, 0, [6, 4], [1, 2]),
        CuttingStockInstance(11, 1, [2, 3, 5], [2, 2, 1]),
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
