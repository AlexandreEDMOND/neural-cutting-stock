from itertools import product

import pytest

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    CompleteIntegerMaster,
    IntegerRestrictedMasterProblem,
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
    iter_maximal_patterns,
)


def _all_feasible_patterns(instance: CuttingStockInstance) -> tuple[tuple[int, ...], ...]:
    return tuple(
        pattern
        for pattern in product(*(range(demand + 1) for demand in instance.demands))
        if any(pattern) and instance.capacity_used(pattern) <= instance.stock_length
    )


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(10, 0, [2, 3], [5, 5]),
        CuttingStockInstance(6, 1, [2, 3], [3, 2]),
        CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2]),
        CuttingStockInstance(12, 2, [2, 4], [3, 2]),
        CuttingStockInstance(9, 3, [3], [4]),
    ],
)
def test_complete_master_matches_exhaustive_integer_master(
    instance: CuttingStockInstance,
) -> None:
    complete = CompleteIntegerMaster(instance).solve()
    exhaustive = IntegerRestrictedMasterProblem(instance, _all_feasible_patterns(instance)).solve()

    assert complete.status == 0
    assert exhaustive.status == 0
    assert complete.objective_value == exhaustive.objective_value


def test_kerf_exercised_optimum() -> None:
    instance = CuttingStockInstance(10, 1, [4], [5])

    result = CompleteIntegerMaster(instance).solve()

    assert result.status == 0
    assert result.objective_value == 3


def test_certified_lower_bound_proven_at_optimality() -> None:
    instance = CuttingStockInstance(10, 0, [2, 3], [5, 5])

    result = CompleteIntegerMaster(instance).solve()

    assert result.certified_lower_bound is not None
    assert result.objective_value is not None
    assert result.certified_lower_bound <= result.objective_value
    assert result.certified_lower_bound == pytest.approx(result.objective_value)


def test_solve_is_deterministic() -> None:
    instance = CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2])

    assert CompleteIntegerMaster(instance).solve() == CompleteIntegerMaster(instance).solve()


def test_column_values_cover_every_demand_within_enumerated_patterns() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])

    result = CompleteIntegerMaster(instance).solve()

    assert result.status == 0
    patterns = list(iter_maximal_patterns(instance))
    produced = [0] * instance.number_of_types
    for pattern, count in zip(patterns, result.column_values, strict=True):
        assert count >= 0
        for index, pieces in enumerate(pattern):
            produced[index] += pieces * count
    assert all(got >= want for got, want in zip(produced, instance.demands, strict=True))


def test_pattern_count_reports_enumerated_columns() -> None:
    instance = CuttingStockInstance(12, 2, [2, 4], [3, 2])

    result = CompleteIntegerMaster(instance).solve()

    assert result.number_of_patterns == len(list(iter_maximal_patterns(instance)))


def test_enumeration_guard_propagates_instead_of_truncating() -> None:
    instance = CuttingStockInstance(20, 0, [3, 5], [6, 4])
    limits = MaximalPatternLimits(max_search_space_size=10)

    with pytest.raises(PatternEnumerationLimitExceeded):
        CompleteIntegerMaster(instance, limits).solve()
