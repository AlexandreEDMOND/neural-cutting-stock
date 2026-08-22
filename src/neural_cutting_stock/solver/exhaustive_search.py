"""Independent exhaustive search for the complete integer master optimum."""

from dataclasses import dataclass
from functools import cache

from neural_cutting_stock.problem import CuttingStockInstance

from .maximal_patterns import MaximalPatternLimits, iter_maximal_patterns


@dataclass(frozen=True, slots=True)
class ExhaustiveIntegerOptimum:
    """Integer optimum proven by pure enumeration instead of a MILP solver."""

    optimum_bars: int
    number_of_patterns: int


class ExhaustiveIntegerSearch:
    """Recompute the complete integer master optimum without any MILP solver.

    The search walks the same demand-bounded maximal patterns as the MILP
    route, so it shares the documented domain reduction, but it replaces the
    branch-and-bound solver with a memoized recursion over residual demand
    vectors. Agreement between the two routes therefore validates the model
    and its optimality argument rather than one shared black box.

    Completeness: every optimal plan contains a bar producing a piece of the
    lowest-index type whose demand is still unmet, so branching only over
    such bars loses no solution. Bars sharing a pattern are decided together;
    beyond the multiplicity that fully covers every type the pattern can
    serve, extra copies only add bars, so the search stops there.
    """

    def __init__(
        self,
        instance: CuttingStockInstance,
        limits: MaximalPatternLimits | None = None,
    ) -> None:
        self.instance = instance
        self.limits = limits

    def solve(self) -> ExhaustiveIntegerOptimum:
        """Enumerate maximal patterns once, then minimize bars by search.

        Enumeration guards propagate unchanged instead of truncating an
        exact reference silently.
        """

        patterns = tuple(iter_maximal_patterns(self.instance, self.limits))
        if not patterns:
            raise ValueError("enumeration produced no maximal pattern")
        demands = self.instance.demands
        covering: list[list[int]] = [[] for _ in demands]
        for index, pattern in enumerate(patterns):
            for piece_type, count in enumerate(pattern):
                if count > 0:
                    covering[piece_type].append(index)

        @cache
        def min_bars(residual: tuple[int, ...]) -> int:
            first_unmet = next(
                (piece_type for piece_type, left in enumerate(residual) if left > 0),
                None,
            )
            if first_unmet is None:
                return 0
            best: int | None = None
            for index in covering[first_unmet]:
                pattern = patterns[index]
                multiplicity_cap = max(
                    -(-left // count)
                    for left, count in zip(residual, pattern, strict=True)
                    if left > 0 and count > 0
                )
                for multiplicity in range(1, multiplicity_cap + 1):
                    remaining = tuple(
                        max(left - multiplicity * count, 0)
                        for left, count in zip(residual, pattern, strict=True)
                    )
                    candidate = multiplicity + min_bars(remaining)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                raise ValueError("no maximal pattern serves an unmet demand")
            return best

        return ExhaustiveIntegerOptimum(min_bars(demands), len(patterns))
