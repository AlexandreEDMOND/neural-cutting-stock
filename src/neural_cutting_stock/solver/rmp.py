"""Linear Restricted Master Problem for 1D Cutting Stock."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from neural_cutting_stock.problem import CuttingStockInstance


@dataclass(frozen=True, slots=True)
class RMPResult:
    """Result of solving the linear restricted master problem."""

    status: int
    objective_value: float | None
    column_values: tuple[float, ...]
    dual_values: tuple[float, ...]
    message: str


class RestrictedMasterProblem:
    """Solve a linear covering master over a fixed set of cutting patterns."""

    def __init__(
        self,
        instance: CuttingStockInstance,
        patterns: tuple[tuple[int, ...], ...],
    ) -> None:
        if not patterns:
            raise ValueError("patterns must not be empty")
        if len(set(patterns)) != len(patterns):
            raise ValueError("patterns must be distinct")
        for pattern in patterns:
            if instance.capacity_used(pattern) > instance.stock_length:
                raise ValueError("patterns must respect stock capacity")
            if not any(pattern):
                raise ValueError("patterns must be non-empty")
        self.instance = instance
        self.patterns = patterns

    def solve(self) -> RMPResult:
        """Minimize the number of bars and return coverage dual values.

        HiGHS receives ``-A x <= -d`` because the master constraints are
        covering constraints. Its inequality marginals therefore have the
        opposite sign of the non-negative covering duals.
        """

        matrix = np.asarray(self.patterns, dtype=float).T
        result = linprog(
            c=np.ones(len(self.patterns)),
            A_ub=-matrix,
            b_ub=-np.asarray(self.instance.demands, dtype=float),
            bounds=(0, None),
            method="highs",
        )
        if not result.success:
            return RMPResult(
                status=result.status,
                objective_value=None,
                column_values=(),
                dual_values=(),
                message=result.message,
            )

        duals = np.maximum(0.0, -np.asarray(result.ineqlin.marginals))
        return RMPResult(
            status=result.status,
            objective_value=float(result.fun),
            column_values=tuple(float(value) for value in result.x),
            dual_values=tuple(float(value) for value in duals),
            message=result.message,
        )
