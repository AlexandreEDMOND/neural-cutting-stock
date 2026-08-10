"""Classical LP column generation for one-dimensional Cutting Stock."""

import math
from dataclasses import dataclass

from neural_cutting_stock.problem import CuttingStockInstance

from .integer_master import IntegerMasterResult, IntegerRestrictedMasterProblem
from .pricing import ExactPricing, PricingResult
from .rmp import RestrictedMasterProblem, RMPResult


@dataclass(frozen=True, slots=True)
class ColumnGenerationResult:
    """Result of the LP relaxation at exact pricing convergence."""

    status: str
    patterns: tuple[tuple[int, ...], ...]
    rmp_result: RMPResult | None
    pricing_result: PricingResult | None
    integer_master_result: IntegerMasterResult | None
    iterations: int
    columns_added: int
    duplicate_columns: int
    termination_reason: str


class ColumnGeneration:
    """Iteratively solve the RMP and add exact pricing columns."""

    def __init__(
        self, instance: CuttingStockInstance, reduced_cost_tolerance: float = 1e-9
    ) -> None:
        if not math.isfinite(reduced_cost_tolerance) or reduced_cost_tolerance < 0:
            raise ValueError("reduced_cost_tolerance must be finite and non-negative")
        self.instance = instance
        self.reduced_cost_tolerance = reduced_cost_tolerance

    def solve(self) -> ColumnGenerationResult:
        """Return the generated patterns once exact pricing finds no improvement."""

        patterns = list(self.instance.initial_patterns())
        columns_added = 0
        duplicate_columns = 0
        iterations = 0
        rmp_result: RMPResult | None = None
        pricing_result: PricingResult | None = None
        integer_master_result: IntegerMasterResult | None = None

        while True:
            iterations += 1
            rmp_result = RestrictedMasterProblem(self.instance, tuple(patterns)).solve()
            if rmp_result.status != 0:
                return ColumnGenerationResult(
                    "solver_error",
                    tuple(patterns),
                    rmp_result,
                    pricing_result,
                    integer_master_result,
                    iterations,
                    columns_added,
                    duplicate_columns,
                    "rmp_failed",
                )

            pricing_result = ExactPricing(self.instance).solve(rmp_result.dual_values)
            if pricing_result.status != 0:
                return ColumnGenerationResult(
                    "solver_error",
                    tuple(patterns),
                    rmp_result,
                    pricing_result,
                    integer_master_result,
                    iterations,
                    columns_added,
                    duplicate_columns,
                    "pricing_failed",
                )
            if pricing_result.reduced_cost is None or pricing_result.pattern == ():
                return ColumnGenerationResult(
                    "solver_error",
                    tuple(patterns),
                    rmp_result,
                    pricing_result,
                    integer_master_result,
                    iterations,
                    columns_added,
                    duplicate_columns,
                    "pricing_returned_no_pattern",
                )
            if pricing_result.reduced_cost >= -self.reduced_cost_tolerance:
                integer_master_result = IntegerRestrictedMasterProblem(
                    self.instance, tuple(patterns)
                ).solve()
                if integer_master_result.status != 0:
                    return ColumnGenerationResult(
                        "solver_error",
                        tuple(patterns),
                        rmp_result,
                        pricing_result,
                        integer_master_result,
                        iterations,
                        columns_added,
                        duplicate_columns,
                        "integer_master_failed",
                    )
                return ColumnGenerationResult(
                    "converged",
                    tuple(patterns),
                    rmp_result,
                    pricing_result,
                    integer_master_result,
                    iterations,
                    columns_added,
                    duplicate_columns,
                    "no_improving_column",
                )
            if pricing_result.pattern in patterns:
                duplicate_columns += 1
                return ColumnGenerationResult(
                    "solver_error",
                    tuple(patterns),
                    rmp_result,
                    pricing_result,
                    integer_master_result,
                    iterations,
                    columns_added,
                    duplicate_columns,
                    "improving_duplicate_column",
                )

            patterns.append(pricing_result.pattern)
            columns_added += 1
