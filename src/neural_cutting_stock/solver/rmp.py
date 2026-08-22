"""Linear Restricted Master Problem for 1D Cutting Stock."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from neural_cutting_stock.problem import AnyCuttingStockInstance

from ._patterns import validate_patterns


@dataclass(frozen=True, slots=True)
class RMPResult:
    """Result of solving the linear restricted master problem."""

    status: int
    objective_value: float | None
    column_values: tuple[float, ...]
    dual_values: tuple[float, ...]
    message: str


@dataclass(frozen=True, slots=True)
class RMPState:
    """Immutable observation of one RMP solve in a CG trajectory."""

    iteration_index: int
    instance_id: str | None
    patterns: tuple[tuple[int, ...], ...]
    result: RMPResult
    runtime_seconds: float

    def __post_init__(self) -> None:
        if self.iteration_index < 1:
            raise ValueError("iteration_index must start at 1")
        if self.instance_id is not None and not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty when present")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")


class RestrictedMasterProblem:
    """Solve a linear covering master over a fixed set of cutting patterns."""

    def __init__(
        self,
        instance: AnyCuttingStockInstance,
        patterns: tuple[tuple[int, ...], ...],
    ) -> None:
        validate_patterns(instance, patterns)
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
