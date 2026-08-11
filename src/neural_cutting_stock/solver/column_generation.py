"""Classical LP column generation for one-dimensional Cutting Stock."""

import math
from dataclasses import dataclass
from time import perf_counter

from neural_cutting_stock.problem import CuttingStockInstance

from .integer_master import IntegerMasterResult, IntegerRestrictedMasterProblem
from .pricing import ExactPricing, PricingResult
from .rmp import RestrictedMasterProblem, RMPResult
from .verification import PlanVerification, verify_plan


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
    verification: PlanVerification | None = None
    total_runtime_seconds: float = 0.0
    master_problem_runtime: float = 0.0
    pricing_runtime: float = 0.0
    integer_master_runtime: float = 0.0
    column_management_runtime: float = 0.0
    verification_runtime: float = 0.0
    unattributed_runtime: float = 0.0

    @property
    def integrality_gap(self) -> float | None:
        """Return the restricted integer objective minus the converged LP bound."""

        if self.rmp_result is None or self.rmp_result.objective_value is None:
            return None
        if self.integer_master_result is None:
            return None
        if self.integer_master_result.objective_value is None:
            return None
        return self.integer_master_result.objective_value - self.rmp_result.objective_value

    @property
    def integer_solution_guarantee(self) -> str | None:
        """Describe the scope of the restricted integer master's guarantee."""

        if self.integer_master_result is None:
            return None
        return "optimal_over_generated_columns_only"


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

        started = perf_counter()
        master_problem_runtime = 0.0
        pricing_runtime = 0.0
        integer_master_runtime = 0.0
        column_management_runtime = 0.0
        verification_runtime = 0.0

        def make_result(
            status: str,
            termination_reason: str,
            verification: PlanVerification | None = None,
        ) -> ColumnGenerationResult:
            total_runtime = perf_counter() - started
            instrumented_runtime = (
                master_problem_runtime
                + pricing_runtime
                + integer_master_runtime
                + column_management_runtime
                + verification_runtime
            )
            return ColumnGenerationResult(
                status,
                tuple(patterns),
                rmp_result,
                pricing_result,
                integer_master_result,
                iterations,
                columns_added,
                duplicate_columns,
                termination_reason,
                verification,
                total_runtime,
                master_problem_runtime,
                pricing_runtime,
                integer_master_runtime,
                column_management_runtime,
                verification_runtime,
                total_runtime - instrumented_runtime,
            )

        management_started = perf_counter()
        patterns = list(self.instance.initial_patterns())
        column_management_runtime += perf_counter() - management_started
        columns_added = 0
        duplicate_columns = 0
        iterations = 0
        rmp_result: RMPResult | None = None
        pricing_result: PricingResult | None = None
        integer_master_result: IntegerMasterResult | None = None

        while True:
            iterations += 1
            component_started = perf_counter()
            rmp_result = RestrictedMasterProblem(self.instance, tuple(patterns)).solve()
            master_problem_runtime += perf_counter() - component_started
            if rmp_result.status != 0:
                return make_result(_failure_status(rmp_result.status), "rmp_failed")

            component_started = perf_counter()
            pricing_result = ExactPricing(self.instance).solve(rmp_result.dual_values)
            pricing_runtime += perf_counter() - component_started
            if pricing_result.status != 0:
                return make_result(_failure_status(pricing_result.status), "pricing_failed")
            if pricing_result.reduced_cost is None or pricing_result.pattern == ():
                return make_result("solver_error", "pricing_returned_no_pattern")
            management_started = perf_counter()
            is_duplicate = pricing_result.pattern in patterns
            if is_duplicate:
                duplicate_columns += 1
            column_management_runtime += perf_counter() - management_started
            if is_duplicate and pricing_result.reduced_cost < -self.reduced_cost_tolerance:
                return make_result("solver_error", "improving_duplicate_column")
            if pricing_result.reduced_cost >= -self.reduced_cost_tolerance:
                component_started = perf_counter()
                integer_master_result = IntegerRestrictedMasterProblem(
                    self.instance, tuple(patterns)
                ).solve()
                integer_master_runtime += perf_counter() - component_started
                if integer_master_result.status != 0:
                    return make_result(
                        _failure_status(integer_master_result.status), "integer_master_failed"
                    )
                component_started = perf_counter()
                verification = verify_plan(
                    self.instance, tuple(patterns), integer_master_result.column_values
                )
                verification_runtime += perf_counter() - component_started
                if (
                    not verification.feasible
                    or verification.number_of_stock_bars
                    != integer_master_result.objective_value
                ):
                    return make_result("invalid_plan", "invalid_plan", verification)
                return make_result("converged", "no_improving_column", verification)
            management_started = perf_counter()
            patterns.append(pricing_result.pattern)
            columns_added += 1
            column_management_runtime += perf_counter() - management_started


def _failure_status(solver_status: int) -> str:
    """Translate HiGHS' common failure statuses into public result statuses."""

    if solver_status == 1:
        return "limit_reached"
    if solver_status == 2:
        return "infeasible"
    return "solver_error"
