"""Integer restricted master problem for 1D Cutting Stock."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, milp

from neural_cutting_stock.problem import CuttingStockInstance


@dataclass(frozen=True, slots=True)
class IntegerMasterResult:
    """Result of solving the integer restricted master problem."""

    status: int
    objective_value: float | None
    column_values: tuple[int, ...]
    message: str


class IntegerRestrictedMasterProblem:
    """Solve the integer covering master over generated cutting patterns."""

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

    def solve(self) -> IntegerMasterResult:
        """Minimize the number of bars while covering every demand."""

        matrix = np.asarray(self.patterns, dtype=float).T
        result = milp(
            c=np.ones(len(self.patterns)),
            integrality=np.ones(len(self.patterns)),
            bounds=(np.zeros(len(self.patterns)), np.full(len(self.patterns), np.inf)),
            constraints=LinearConstraint(
                matrix,
                np.asarray(self.instance.demands, dtype=float),
                np.full(self.instance.number_of_types, np.inf),
            ),
        )
        if not result.success:
            return IntegerMasterResult(result.status, None, (), result.message)

        return IntegerMasterResult(
            status=result.status,
            objective_value=float(result.fun),
            column_values=tuple(int(round(value)) for value in result.x),
            message=result.message,
        )
