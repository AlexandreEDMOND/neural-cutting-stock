"""Exact integer pricing for the one-dimensional Cutting Stock problem."""

from dataclasses import dataclass
from decimal import Decimal
from numbers import Real

import numpy as np
from scipy.optimize import LinearConstraint, milp

from neural_cutting_stock.problem import AnyCuttingStockInstance


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Result of maximizing the dual value of a feasible pattern."""

    status: int
    pattern: tuple[int, ...]
    dual_value: float | None
    reduced_cost: float | None
    message: str


class ExactPricing:
    """Solve the bounded integer-knapsack pricing problem exactly.

    The knapsack capacity is ``instance.stock_length``, which for a declared
    multi-format instance is the largest declared format: with non-negative
    covering duals, the best column over all declared formats always lies in
    that pattern set.
    """

    def __init__(self, instance: AnyCuttingStockInstance) -> None:
        self.instance = instance

    def solve(self, dual_values: tuple[float, ...]) -> PricingResult:
        """Return a maximum-dual-value pattern and its reduced cost."""

        if len(dual_values) != self.instance.number_of_types:
            raise ValueError("dual_values must contain one value per piece type")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
            for value in dual_values
        ):
            raise ValueError("dual_values must be finite and non-negative")

        weights, stock_length = _integer_capacity_coefficients(self.instance)
        result = milp(
            c=-np.asarray(dual_values, dtype=float),
            integrality=np.ones(self.instance.number_of_types),
            bounds=(np.zeros(self.instance.number_of_types), self.instance.demands),
            constraints=(
                LinearConstraint(weights, -np.inf, stock_length),
                LinearConstraint(
                    np.ones(self.instance.number_of_types), 1, np.inf
                ),
            ),
        )
        if not result.success:
            return PricingResult(result.status, (), None, None, result.message)

        pattern = tuple(int(round(value)) for value in result.x)
        dual_value = float(np.dot(dual_values, pattern))
        return PricingResult(
            status=result.status,
            pattern=pattern,
            dual_value=dual_value,
            reduced_cost=1.0 - dual_value,
            message=result.message,
        )


def _integer_capacity_coefficients(
    instance: AnyCuttingStockInstance,
) -> tuple[np.ndarray, float]:
    """Scale decimal capacities so exact boundary patterns remain feasible."""

    values = [
        Decimal(str(length)) + Decimal(str(instance.kerf))
        for length in instance.piece_lengths
    ]
    stock_length = Decimal(str(instance.stock_length))
    scale = max(
        0,
        *(-value.as_tuple().exponent for value in (*values, stock_length)),
    )
    multiplier = Decimal(10) ** scale
    weights = np.asarray([float(value * multiplier) for value in values], dtype=float)
    return weights, float(stock_length * multiplier)
