"""Complete integer master solved exactly by MILP over enumerated patterns."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, milp

from neural_cutting_stock.problem import AnyCuttingStockInstance

from .maximal_patterns import MaximalPatternLimits, iter_maximal_patterns


@dataclass(frozen=True, slots=True)
class CompleteMasterResult:
    """Outcome of the complete integer master solved over enumerated patterns."""

    status: int
    objective_value: float | None
    column_values: tuple[int, ...]
    certified_lower_bound: float | None
    number_of_patterns: int
    message: str


class CompleteIntegerMaster:
    """Solve the complete integer master by MILP over every maximal pattern.

    Every plan of the complete demand-bounded master extends bar by bar to a
    plan over maximal patterns with the same number of bars, so the MILP
    optimum over the enumerated maximal patterns is the proven integer optimum
    of the complete master. For a declared multi-format instance the
    enumeration runs against the largest declared format, whose master has
    the same optimum (see ``MultiFormatCuttingStockInstance.stock_length``).
    HiGHS runs with a zero relative gap and its branch-and-bound dual bound
    is reported as the certified lower bound attached to that proof.
    """

    def __init__(
        self,
        instance: AnyCuttingStockInstance,
        limits: MaximalPatternLimits | None = None,
    ) -> None:
        self.instance = instance
        self.limits = limits

    def solve(self) -> CompleteMasterResult:
        """Enumerate every maximal pattern, then minimize bars by MILP.

        Enumeration guards propagate unchanged instead of truncating an exact
        reference silently.
        """

        patterns = tuple(iter_maximal_patterns(self.instance, self.limits))
        if not patterns:
            raise ValueError("enumeration produced no maximal pattern")
        matrix = np.asarray(patterns, dtype=float).T
        result = milp(
            c=np.ones(len(patterns)),
            integrality=np.ones(len(patterns)),
            bounds=(np.zeros(len(patterns)), np.full(len(patterns), np.inf)),
            constraints=LinearConstraint(
                matrix,
                np.asarray(self.instance.demands, dtype=float),
                np.full(self.instance.number_of_types, np.inf),
            ),
            options={"mip_rel_gap": 0.0},
        )
        if not result.success:
            return CompleteMasterResult(
                status=result.status,
                objective_value=None,
                column_values=(),
                certified_lower_bound=_finite_bound(result.mip_dual_bound),
                number_of_patterns=len(patterns),
                message=result.message,
            )

        return CompleteMasterResult(
            status=result.status,
            objective_value=float(result.fun),
            column_values=tuple(int(round(value)) for value in result.x),
            certified_lower_bound=_finite_bound(result.mip_dual_bound),
            number_of_patterns=len(patterns),
            message=result.message,
        )


def _finite_bound(bound: float | None) -> float | None:
    """Keep only finite dual bounds usable as certified lower bounds."""

    if bound is None or not np.isfinite(bound):
        return None
    return float(bound)
