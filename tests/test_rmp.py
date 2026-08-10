from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import IntegerRestrictedMasterProblem, RestrictedMasterProblem


def test_rmp_solves_covering_master_and_extracts_nonnegative_duals() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])
    rmp = RestrictedMasterProblem(instance, instance.initial_patterns())

    result = rmp.solve()

    assert result.status == 0
    assert result.objective_value == 2
    assert result.column_values == (1.0, 1.0)
    assert all(value >= 0 for value in result.dual_values)
    assert (
        sum(
            demand * dual
            for demand, dual in zip(instance.demands, result.dual_values, strict=True)
        )
        == 2
    )
    assert all(
        sum(dual * count for dual, count in zip(result.dual_values, pattern, strict=True)) <= 1
        for pattern in rmp.patterns
    )


def test_rmp_rejects_duplicate_or_empty_patterns() -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    for patterns, message in [(((1,), (1,)), "distinct"), (((0,),), "non-empty")]:
        try:
            RestrictedMasterProblem(instance, patterns)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid patterns were accepted")


def test_integer_master_returns_feasible_restricted_plan() -> None:
    instance = CuttingStockInstance(10, 0, [6, 4], [1, 2])
    patterns = (*instance.initial_patterns(), (1, 1))

    result = IntegerRestrictedMasterProblem(instance, patterns).solve()

    assert result.status == 0
    assert result.objective_value == 2
    assert all(value >= 0 for value in result.column_values)
    produced = tuple(
        sum(
            value * pattern[index]
            for value, pattern in zip(result.column_values, patterns, strict=True)
        )
        for index in range(instance.number_of_types)
    )
    assert all(
        actual >= demand for actual, demand in zip(produced, instance.demands, strict=True)
    )
