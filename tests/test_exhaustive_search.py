import pytest

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    CompleteIntegerMaster,
    ExhaustiveIntegerOptimum,
    ExhaustiveIntegerSearch,
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
    iter_maximal_patterns,
)


@pytest.mark.parametrize(
    "instance",
    [
        CuttingStockInstance(10, 0, [2, 3], [5, 5]),
        CuttingStockInstance(6, 1, [2, 3], [3, 2]),
        CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2]),
        CuttingStockInstance(12, 2, [2, 4], [3, 2]),
        CuttingStockInstance(9, 3, [3], [4]),
        CuttingStockInstance(10, 0, [6, 4], [3, 2]),
    ],
)
def test_exhaustive_search_agrees_with_the_milp_complete_master(
    instance: CuttingStockInstance,
) -> None:
    milp = CompleteIntegerMaster(instance).solve()
    search = ExhaustiveIntegerSearch(instance).solve()

    assert milp.status == 0
    assert milp.objective_value is not None
    assert search.optimum_bars == int(milp.objective_value)
    assert search.number_of_patterns == len(list(iter_maximal_patterns(instance)))


def test_kerf_exercised_optimum_without_any_milp_solver() -> None:
    instance = CuttingStockInstance(10, 1, [4], [5])

    search = ExhaustiveIntegerSearch(instance).solve()

    assert search.optimum_bars == 3


def test_search_is_deterministic() -> None:
    instance = CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2])

    assert ExhaustiveIntegerSearch(instance).solve() == ExhaustiveIntegerSearch(instance).solve()


def test_result_type_reports_optimum_and_pattern_count() -> None:
    instance = CuttingStockInstance(12, 2, [2, 4], [3, 2])

    result = ExhaustiveIntegerSearch(instance).solve()

    assert isinstance(result, ExhaustiveIntegerOptimum)
    assert result.number_of_patterns >= 1


def test_enumeration_guard_propagates_instead_of_truncating() -> None:
    instance = CuttingStockInstance(20, 0, [3, 5], [6, 4])

    with pytest.raises(PatternEnumerationLimitExceeded):
        ExhaustiveIntegerSearch(
            instance, MaximalPatternLimits(max_search_space_size=10)
        ).solve()
