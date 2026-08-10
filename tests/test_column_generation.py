from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration


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
    assert result.duplicate_columns == 0


def test_column_generation_rejects_invalid_reduced_cost_tolerance() -> None:
    instance = CuttingStockInstance(10, 0, [6], [1])

    for tolerance in [-1.0, float("inf")]:
        try:
            ColumnGeneration(instance, tolerance)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reduced-cost tolerance was accepted")
