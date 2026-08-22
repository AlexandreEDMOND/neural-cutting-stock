"""Publication guardrails for Neural-QC refinement attempts.

Nothing a quality agent produces may reach a report without passing three
guardrails, enforced here in one place:

1. **Independent verification** — the published baseline and final plans are
   re-verified from their persisted patterns and values by
   :func:`neural_cutting_stock.solver.verify_plan`; the verifications embedded
   in the refinement record are never trusted as proof.
2. **Honest statuses** — every published solution carries the three-way
   comparison status ``improved``/``equal``/``degraded`` recomputed from the
   independently verified bar counts; a label contradicting the measurement
   can never be published. The refinement episode itself only ever improves or
   stays equal, so ``degraded`` exists for any future publication path whose
   final solution ends worse than its classical start: it must then be named
   degraded, never relabeled.
3. **Preserved failures** — an attempt that raises (non-converged classical
   start, agent crash, plan failing its independent verification) becomes a
   failure record with its reason and message instead of disappearing; no
   exception is silently swallowed and no failed instance is silently dropped
   from a future aggregation.

Like the refinement result it gates, a published solution keeps the
``optimal_over_generated_columns_only`` scope of its classical start and never
claims global optimality.
"""

import math
from dataclasses import dataclass

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import PlanVerification, verify_plan

from .neural_qc import NeuralQCBudget, NeuralQCRefinementResult, run_neural_quality_refinement
from .quality_agent import QualityAgent

NEURAL_QC_PUBLICATION_SCHEMA_VERSION = "neural-qc-publication-v1"

PUBLICATION_STATUS_IMPROVED = "improved"
PUBLICATION_STATUS_EQUAL = "equal"
PUBLICATION_STATUS_DEGRADED = "degraded"

OUTCOME_SOLUTION = "published_solution"
OUTCOME_FAILURE = "preserved_failure"

FAILURE_BASELINE_VERIFICATION = "baseline_plan_failed_independent_verification"
FAILURE_FINAL_VERIFICATION = "final_plan_failed_independent_verification"
FAILURE_REFINEMENT_ERROR = "refinement_error"

_STATUSES = (
    PUBLICATION_STATUS_IMPROVED,
    PUBLICATION_STATUS_EQUAL,
    PUBLICATION_STATUS_DEGRADED,
)
_OUTCOMES = (OUTCOME_SOLUTION, OUTCOME_FAILURE)


def classify_publication_status(initial_bars: int, final_bars: int) -> str:
    """Return the honest three-way comparison of two measured bar counts."""

    if not isinstance(initial_bars, int) or isinstance(initial_bars, bool) or initial_bars < 1:
        raise ValueError("initial_bars must be a positive integer")
    if not isinstance(final_bars, int) or isinstance(final_bars, bool) or final_bars < 1:
        raise ValueError("final_bars must be a positive integer")
    if final_bars < initial_bars:
        return PUBLICATION_STATUS_IMPROVED
    if final_bars > initial_bars:
        return PUBLICATION_STATUS_DEGRADED
    return PUBLICATION_STATUS_EQUAL


@dataclass(frozen=True, slots=True)
class NeuralQCPublishedSolution:
    """A final solution that passed independent verification before publication.

    Both embedded verifications were recomputed here from the persisted
    patterns and values, not copied from the refinement record. ``status`` is
    always the honest recomputed comparison against the classical start.
    """

    instance_id: str
    status: str
    initial_bars: int
    final_bars: int
    final_patterns: tuple[tuple[int, ...], ...]
    final_column_values: tuple[int, ...]
    baseline_verification: PlanVerification
    final_verification: PlanVerification
    schema_version: str = NEURAL_QC_PUBLICATION_SCHEMA_VERSION

    @property
    def bars_saved(self) -> int:
        """Return the verified bar reduction against the classical start."""

        return self.initial_bars - self.final_bars

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if self.schema_version != NEURAL_QC_PUBLICATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown publication status: {self.status!r}")
        for name in ("initial_bars", "final_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, verification in (
            ("baseline_verification", self.baseline_verification),
            ("final_verification", self.final_verification),
        ):
            if not verification.feasible:
                raise ValueError(f"{name} does not verify; nothing unverified is publishable")
        if self.baseline_verification.number_of_stock_bars != self.initial_bars:
            raise ValueError("baseline_verification disagrees with initial_bars")
        if self.final_verification.number_of_stock_bars != self.final_bars:
            raise ValueError("final_verification disagrees with final_bars")
        expected_status = classify_publication_status(self.initial_bars, self.final_bars)
        if self.status != expected_status:
            raise ValueError(
                f"status {self.status!r} contradicts the independently verified bar counts"
            )


@dataclass(frozen=True, slots=True)
class NeuralQCAttemptRecord:
    """One refinement attempt as it may enter a publication.

    Exactly one side is populated: a :class:`NeuralQCPublishedSolution` under
    ``outcome == "published_solution"``, or a preserved failure carrying its
    machine-readable ``failure_reason`` and human-readable ``failure_message``
    otherwise. Failures are first-class records, never silent omissions.
    """

    instance_id: str
    outcome: str
    solution: NeuralQCPublishedSolution | None = None
    failure_reason: str | None = None
    failure_message: str | None = None
    schema_version: str = NEURAL_QC_PUBLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unknown attempt outcome: {self.outcome!r}")
        has_solution = self.solution is not None
        has_failure = self.failure_reason is not None or self.failure_message is not None
        if self.outcome == OUTCOME_SOLUTION and (not has_solution or has_failure):
            raise ValueError("a published_solution record carries its solution and no failure")
        if self.outcome == OUTCOME_FAILURE and (
            has_solution
            or not isinstance(self.failure_reason, str)
            or not self.failure_reason.strip()
        ):
            raise ValueError(
                "a preserved_failure record carries a non-empty failure_reason and no solution"
            )


def publish_refinement_result(
    instance: AnyCuttingStockInstance,
    result: NeuralQCRefinementResult,
    *,
    verification_tolerance: float = 1e-9,
) -> NeuralQCAttemptRecord:
    """Gate one refinement result through the publication guardrails.

    The classical start and the final plan are both re-verified from the
    persisted patterns and values; a plan that fails this independent check is
    returned as a preserved failure together with the verifier diagnostics
    instead of being published. A verified plan is published with its honest
    recomputed comparison status.
    """

    if not math.isfinite(verification_tolerance) or verification_tolerance < 0:
        raise ValueError("verification_tolerance must be finite and non-negative")

    baseline = verify_plan(
        instance, result.initial_patterns, result.initial_column_values, verification_tolerance
    )
    final = verify_plan(
        instance, result.final_patterns, result.final_column_values, verification_tolerance
    )

    if not baseline.feasible or baseline.number_of_stock_bars != result.initial_bars:
        return _failed_verification_record(
            result.instance_id, FAILURE_BASELINE_VERIFICATION, baseline
        )
    if not final.feasible or final.number_of_stock_bars != result.final_bars:
        return _failed_verification_record(result.instance_id, FAILURE_FINAL_VERIFICATION, final)

    return NeuralQCAttemptRecord(
        instance_id=result.instance_id,
        outcome=OUTCOME_SOLUTION,
        solution=NeuralQCPublishedSolution(
            instance_id=result.instance_id,
            status=classify_publication_status(result.initial_bars, result.final_bars),
            initial_bars=result.initial_bars,
            final_bars=result.final_bars,
            final_patterns=result.final_patterns,
            final_column_values=result.final_column_values,
            baseline_verification=baseline,
            final_verification=final,
        ),
    )


def attempt_quality_refinement(
    instance: AnyCuttingStockInstance,
    instance_id: str,
    agent: QualityAgent,
    *,
    budget: NeuralQCBudget,
    verification_tolerance: float = 1e-9,
) -> NeuralQCAttemptRecord:
    """Run one refinement attempt and always return a publishable record.

    Caller misuse (invalid identifiers, agent, budget or tolerance) still
    raises immediately: those are programming errors, not experiment outcomes.
    Every failure raised by the refinement itself — non-converged classical
    start, crashed proposal, plan failing its independent verification — is
    preserved as a failure record with its type and message, so a future
    evaluation campaign keeps every failed instance visible instead of
    dropping it.
    """

    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")
    if not callable(getattr(agent, "propose", None)):
        raise ValueError("agent must expose a propose(observation) method")
    if not isinstance(budget, NeuralQCBudget):
        raise ValueError("budget must be a NeuralQCBudget")
    if not math.isfinite(verification_tolerance) or verification_tolerance < 0:
        raise ValueError("verification_tolerance must be finite and non-negative")
    try:
        result = run_neural_quality_refinement(
            instance,
            instance_id,
            agent,
            budget=budget,
            verification_tolerance=verification_tolerance,
        )
        return publish_refinement_result(
            instance, result, verification_tolerance=verification_tolerance
        )
    except Exception as error:
        return NeuralQCAttemptRecord(
            instance_id=instance_id,
            outcome=OUTCOME_FAILURE,
            failure_reason=FAILURE_REFINEMENT_ERROR,
            failure_message=f"{type(error).__name__}: {error}",
        )


def _failed_verification_record(
    instance_id: str, reason: str, verification: PlanVerification
) -> NeuralQCAttemptRecord:
    return NeuralQCAttemptRecord(
        instance_id=instance_id,
        outcome=OUTCOME_FAILURE,
        failure_reason=reason,
        failure_message=(
            "; ".join(verification.errors)
            or (
                "independent verification counted "
                f"{verification.number_of_stock_bars} stock bars"
            )
        ),
    )


__all__ = [
    "FAILURE_BASELINE_VERIFICATION",
    "FAILURE_FINAL_VERIFICATION",
    "FAILURE_REFINEMENT_ERROR",
    "NEURAL_QC_PUBLICATION_SCHEMA_VERSION",
    "OUTCOME_FAILURE",
    "OUTCOME_SOLUTION",
    "PUBLICATION_STATUS_DEGRADED",
    "PUBLICATION_STATUS_EQUAL",
    "PUBLICATION_STATUS_IMPROVED",
    "NeuralQCAttemptRecord",
    "NeuralQCPublishedSolution",
    "attempt_quality_refinement",
    "classify_publication_status",
    "publish_refinement_result",
]
