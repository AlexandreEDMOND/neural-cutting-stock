from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    IntegerRestrictedMasterProblem,
    RestrictedMasterProblem,
    verify_plan,
)


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


def test_masters_reject_patterns_exceeding_capacity_with_kerf() -> None:
    instance = CuttingStockInstance(10, 1, [6], [1])

    for master in (RestrictedMasterProblem, IntegerRestrictedMasterProblem):
        try:
            master(instance, ((2,),))
        except ValueError as error:
            assert "stock capacity" in str(error)
        else:
            raise AssertionError("an over-capacity pattern was accepted")


def test_masters_reject_patterns_exceeding_demand() -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    for master in (RestrictedMasterProblem, IntegerRestrictedMasterProblem):
        try:
            master(instance, ((2,),))
        except ValueError as error:
            assert "demands" in str(error)
        else:
            raise AssertionError("a demand-exceeding pattern was accepted")


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


def test_plan_verification_checks_coverage_and_material_balance() -> None:
    instance = CuttingStockInstance(10, 1, [6, 3], [1, 2])
    patterns = ((0, 1), (2, 0))

    verification = verify_plan(instance, patterns, (1, 1))

    assert verification.feasible
    assert verification.errors == ()
    assert verification.number_of_stock_bars == 2
    assert verification.produced_counts == (2, 1)
    assert verification.kerf_loss == 3
    assert verification.trim_loss == 5
    assert verification.total_waste == 8
    assert verification.total_waste == (
        verification.overproduction_length
        + verification.kerf_loss
        + verification.trim_loss
    )


def test_plan_verification_reports_infeasible_pattern_and_coverage() -> None:
    instance = CuttingStockInstance(10, 0, [6, 3], [1, 2])

    verification = verify_plan(instance, ((1, 0),), (1,))

    assert not verification.feasible
    assert any("does not cover every demand" in error for error in verification.errors)
