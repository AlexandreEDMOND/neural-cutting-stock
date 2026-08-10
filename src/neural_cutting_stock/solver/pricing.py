"""Exact integer pricing for the one-dimensional Cutting Stock problem."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, milp

from neural_cutting_stock.problem import CuttingStockInstance


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Result of maximizing the dual value of a feasible pattern."""

    status: int
    pattern: tuple[int, ...]
    dual_value: float | None
    reduced_cost: float | None
    message: str


class ExactPricing:
    """Solve the bounded integer-knapsack pricing problem exactly."""

    def __init__(self, instance: CuttingStockInstance) -> None:
        self.instance = instance

    def solve(self, dual_values: tuple[float, ...]) -> PricingResult:
        """Return a maximum-dual-value pattern and its reduced cost."""

        if len(dual_values) != self.instance.number_of_types:
            raise ValueError("dual_values must contain one value per piece type")
        if any(not np.isfinite(value) or value < 0 for value in dual_values):
            raise ValueError("dual_values must be finite and non-negative")

        weights = np.asarray(
            [length + self.instance.kerf for length in self.instance.piece_lengths],
            dtype=float,
        )
        result = milp(
            c=-np.asarray(dual_values, dtype=float),
            integrality=np.ones(self.instance.number_of_types),
            bounds=(np.zeros(self.instance.number_of_types), self.instance.demands),
            constraints=(
                LinearConstraint(weights, -np.inf, self.instance.stock_length),
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
