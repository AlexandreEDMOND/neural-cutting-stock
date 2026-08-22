"""Independent verification of MILP exact references under `exact-reference-v1`.

Every optimal reference is re-derived from its instance instead of trusting
the solver that produced it: the plan is replayed through the independent
plan checker, the linear relaxation of the complete master must stay below
the recorded integer optimum, and an optional pure-enumeration search may
recompute the optimum without any MILP solver. Disagreements are reported
as errors and never silently repaired.
"""

from dataclasses import dataclass

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import (
    CompleteMasterResult,
    ExhaustiveIntegerSearch,
    MaximalPatternLimits,
    PlanVerification,
    RestrictedMasterProblem,
    iter_maximal_patterns,
    verify_plan,
)

from .exact_reference import (
    ExactReferenceMethod,
    ExactReferenceRecord,
    ExactReferenceStatus,
)


@dataclass(frozen=True, slots=True)
class ExactReferenceVerification:
    """Outcome of the independent checks applied to one exact reference."""

    instance_id: str
    plan_verification: PlanVerification | None
    lp_bound_bars: float | None
    exhaustive_optimum_bars: int | None
    errors: tuple[str, ...]
    passed: bool


def verify_milp_exact_reference(
    instance_id: str,
    instance: AnyCuttingStockInstance,
    outcome: CompleteMasterResult,
    record: ExactReferenceRecord,
    *,
    limits: MaximalPatternLimits | None = None,
    cross_check_with_enumeration: bool = False,
) -> ExactReferenceVerification:
    """Re-derive every claim of one optimal MILP reference from scratch.

    The plan carried by ``outcome`` over the enumerated maximal patterns is
    checked independently, the complete-master LP relaxation must not exceed
    the recorded integer optimum within the declared feasibility tolerance,
    and when ``cross_check_with_enumeration`` is set the optimum is recomputed
    by pure enumeration and compared. Only optimal records carry verifiable
    numerical claims; anything else is refused explicitly.
    """

    if not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")
    if record.instance_id != instance_id:
        raise ValueError("record does not belong to this instance_id")
    if record.reference_method is not ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS:
        raise ValueError("verification supports only MILP-on-enumerated-patterns references")
    if record.status is not ExactReferenceStatus.OPTIMAL:
        raise ValueError("only an optimal reference carries a verifiable plan")

    errors: list[str] = []
    plan_verification: PlanVerification | None = None
    lp_bound_bars: float | None = None
    exhaustive_optimum_bars: int | None = None

    patterns = tuple(iter_maximal_patterns(instance, limits))
    if len(patterns) != outcome.number_of_patterns:
        errors.append(
            "enumeration produced "
            f"{len(patterns)} patterns while the outcome reports {outcome.number_of_patterns}"
        )
    elif outcome.status != 0 or outcome.objective_value is None:
        errors.append("outcome is not a proven optimum")
    else:
        plan_verification = verify_plan(
            instance,
            patterns,
            outcome.column_values,
            tolerance=record.feasibility_tolerance,
        )
        if not plan_verification.feasible:
            errors.extend(f"plan check failed: {error}" for error in plan_verification.errors)
        if plan_verification.number_of_stock_bars != record.integer_optimum_bars:
            errors.append(
                "plan uses "
                f"{plan_verification.number_of_stock_bars} bars while the reference claims "
                f"{record.integer_optimum_bars}"
            )
        if (
            abs(outcome.objective_value - record.integer_optimum_bars)
            > record.integrality_tolerance
        ):
            errors.append(
                f"outcome objective {outcome.objective_value} disagrees with the claimed "
                f"optimum {record.integer_optimum_bars}"
            )

        lp_result = RestrictedMasterProblem(instance, patterns).solve()
        if lp_result.status != 0 or lp_result.objective_value is None:
            errors.append(f"LP relaxation of the complete master failed: {lp_result.message}")
        else:
            lp_bound_bars = lp_result.objective_value
            if lp_bound_bars > record.integer_optimum_bars + record.feasibility_tolerance:
                errors.append(
                    f"LP relaxation bound {lp_bound_bars} exceeds the integer optimum "
                    f"{record.integer_optimum_bars}"
                )
        if record.certified_lower_bound_bars is None:
            errors.append("optimal reference carries no certified lower bound")
        elif (
            record.certified_lower_bound_bars
            > record.integer_optimum_bars + record.feasibility_tolerance
        ):
            errors.append(
                f"certified lower bound {record.certified_lower_bound_bars} exceeds the "
                f"integer optimum {record.integer_optimum_bars}"
            )

    if cross_check_with_enumeration:
        cross_check = ExhaustiveIntegerSearch(instance, limits).solve()
        exhaustive_optimum_bars = cross_check.optimum_bars
        if exhaustive_optimum_bars != record.integer_optimum_bars:
            errors.append(
                f"enumeration finds {exhaustive_optimum_bars} bars while the reference "
                f"claims {record.integer_optimum_bars}"
            )

    return ExactReferenceVerification(
        instance_id=instance_id,
        plan_verification=plan_verification,
        lp_bound_bars=lp_bound_bars,
        exhaustive_optimum_bars=exhaustive_optimum_bars,
        errors=tuple(errors),
        passed=not errors,
    )


__all__ = ["ExactReferenceVerification", "verify_milp_exact_reference"]
