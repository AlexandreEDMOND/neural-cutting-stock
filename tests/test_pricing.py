from itertools import product

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ExactPricing


def test_exact_pricing_matches_exhaustive_enumeration() -> None:
    instance = CuttingStockInstance(10, 1, [2, 3], [3, 2])
    dual_values = (0.6, 0.75)

    result = ExactPricing(instance).solve(dual_values)
    feasible_values = [
        sum(value * count for value, count in zip(dual_values, pattern, strict=True))
        for pattern in product(*(range(demand + 1) for demand in instance.demands))
        if instance.capacity_used(pattern) <= instance.stock_length
    ]

    assert result.status == 0
    assert result.dual_value == max(feasible_values)
    assert result.reduced_cost == 1 - result.dual_value
    assert instance.capacity_used(result.pattern) <= instance.stock_length


def test_pricing_rejects_invalid_dual_vector() -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    for dual_values in [(), (-0.1,), (float("inf"),)]:
        try:
            ExactPricing(instance).solve(dual_values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dual values were accepted")


def test_exact_pricing_excludes_the_empty_pattern() -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    result = ExactPricing(instance).solve((0.0,))

    assert result.status == 0
    assert result.pattern == (1,)
    assert result.dual_value == 0.0
    assert result.reduced_cost == 1.0
