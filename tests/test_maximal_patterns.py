import itertools
from itertools import product

import pytest

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    IntegerRestrictedMasterProblem,
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
    iter_maximal_patterns,
)


def _is_box_maximal(instance: CuttingStockInstance, pattern: tuple[int, ...]) -> bool:
    for index, count in enumerate(pattern):
        if count >= instance.demands[index]:
            continue
        extended = (*pattern[:index], count + 1, *pattern[index + 1 :])
        if instance.capacity_used(extended) <= instance.stock_length:
            return False
    return True


def _bruteforce_maximal_patterns(instance: CuttingStockInstance) -> set[tuple[int, ...]]:
    return {
        pattern
        for pattern in product(*(range(demand + 1) for demand in instance.demands))
        if any(pattern)
        and instance.capacity_used(pattern) <= instance.stock_length
        and _is_box_maximal(instance, pattern)
    }


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(10, 0, [2, 3], [5, 5]),
        CuttingStockInstance(6, 1, [2, 3], [3, 2]),
        CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2]),
        CuttingStockInstance(12, 2, [2, 4], [3, 2]),
        CuttingStockInstance(9, 3, [3], [4]),
        CuttingStockInstance(100, 0, [30], [1]),
    ],
)
def test_enumeration_matches_independent_bruteforce_filter(
    instance: CuttingStockInstance,
) -> None:
    enumerated = list(iter_maximal_patterns(instance))

    assert enumerated
    assert set(enumerated) == _bruteforce_maximal_patterns(instance)


def test_enumeration_is_deterministic_and_lexicographic() -> None:
    instance = CuttingStockInstance(10, 0, [2, 3], [5, 5])

    first = list(iter_maximal_patterns(instance))
    second = list(iter_maximal_patterns(instance))

    assert first == second == sorted(first)


def test_kerf_convention_bounds_the_enumerated_counts() -> None:
    instance = CuttingStockInstance(10, 1, [4], [5])

    assert list(iter_maximal_patterns(instance)) == [(2,)]


def test_decimal_boundary_patterns_stay_exact() -> None:
    instance = CuttingStockInstance(0.5, 0.1, [0.1, 0.15], [9, 2])

    assert list(iter_maximal_patterns(instance)) == [(0, 2), (1, 1), (2, 0)]


def test_demand_saturated_pattern_is_maximal_despite_remaining_capacity() -> None:
    instance = CuttingStockInstance(100, 0, [30], [1])

    assert list(iter_maximal_patterns(instance)) == [(1,)]


def test_iteration_streams_without_eager_materialization() -> None:
    instance = CuttingStockInstance(40, 0, [6, 7, 8], [20, 20, 20])

    iterator = iter_maximal_patterns(instance)

    assert iter(iterator) is iterator
    assert next(iterator) == (0, 0, 5)


def test_pattern_budget_guard_stops_instead_of_truncating_silently() -> None:
    instance = CuttingStockInstance(20, 0, [3, 5], [6, 4])
    limits = MaximalPatternLimits(max_search_space_size=10_000, max_patterns=3)

    iterator = iter_maximal_patterns(instance, limits)

    assert list(itertools.islice(iterator, 3)) == [(0, 4), (1, 3), (3, 2)]
    with pytest.raises(PatternEnumerationLimitExceeded, match="max_patterns=3"):
        next(iterator)


def test_search_space_guard_refuses_instance_before_any_iteration() -> None:
    instance = CuttingStockInstance(60, 0, [7, 11, 13], [30, 30, 30])
    limits = MaximalPatternLimits(max_search_space_size=10)

    with pytest.raises(
        PatternEnumerationLimitExceeded, match="max_search_space_size=10"
    ):
        iter_maximal_patterns(instance, limits)


@pytest.mark.parametrize(
    "limits",
    [
        MaximalPatternLimits(max_search_space_size=0),
        MaximalPatternLimits(max_search_space_size=-1),
        MaximalPatternLimits(max_search_space_size=1.5),
        MaximalPatternLimits(max_patterns=0),
        MaximalPatternLimits(max_patterns=True),
    ],
)
def test_limits_must_be_positive_integers(limits: MaximalPatternLimits) -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    with pytest.raises(ValueError, match="positive integer"):
        iter_maximal_patterns(instance, limits)


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(7, 0, [2, 3], [1, 1]),
        CuttingStockInstance(6, 1, [2, 3], [3, 2]),
        CuttingStockInstance(10, 0, [2, 3], [5, 5]),
    ],
)
def test_complete_master_optimum_survives_restriction_to_maximal_patterns(
    instance: CuttingStockInstance,
) -> None:
    def optimum(patterns: tuple[tuple[int, ...], ...]) -> float | None:
        result = IntegerRestrictedMasterProblem(instance, patterns).solve()
        assert result.status == 0
        return result.objective_value

    every_feasible_pattern = tuple(
        pattern
        for pattern in product(*(range(demand + 1) for demand in instance.demands))
        if any(pattern) and instance.capacity_used(pattern) <= instance.stock_length
    )

    assert optimum(every_feasible_pattern) == optimum(
        tuple(iter_maximal_patterns(instance))
    )
