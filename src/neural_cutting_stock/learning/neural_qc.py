"""Neural-QC pipeline: verified agent refinement of the classical solution.

The pipeline is the Phase 9 integration entry point. It starts from the
restricted integer solution of the classical column-generation loop — the
only starting point the project trusts — then hands the versioned
``quality-agent-interface-v1`` observations to any agent implementing the
:class:`QualityAgent` protocol and reviews every proposal through the
systematic independent verification of :func:`verify_proposal`.

The episode applies the agent until one of the two declared terminations:

- ``improvement_budget_exhausted``: the declared step budget
  (:class:`NeuralQCBudget`) is exhausted;
- ``converged_no_further_improvement``: ``stall_patience`` consecutive
  reviews failed to improve the incumbent, so the refinement has nothing
  left to add under this agent.

Neither termination ever means global optimality: the final plan keeps the
``optimal_over_generated_columns_only`` scope of its classical start, it is
verified independently one last time before the result is built, and every
review — rejections and invalid plans included — stays recorded in the
result. The incumbent only ever moves on a verified strict bar reduction,
so ``improved`` versus ``unchanged`` are the only honest statuses.
"""

import math
from dataclasses import dataclass
from time import perf_counter

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import ColumnGeneration, PlanVerification, verify_plan

from .quality_agent import ProposalReview, QualityAgent
from .quality_env import QualityRefinementEnv

NEURAL_QC_PIPELINE_SCHEMA_VERSION = "neural-qc-pipeline-v1"

STATUS_IMPROVED = "improved"
STATUS_UNCHANGED = "unchanged"
TERMINATION_BUDGET_EXHAUSTED = "improvement_budget_exhausted"
TERMINATION_CONVERGED = "converged_no_further_improvement"

_STATUSES = (STATUS_IMPROVED, STATUS_UNCHANGED)
_TERMINATION_REASONS = (TERMINATION_BUDGET_EXHAUSTED, TERMINATION_CONVERGED)


@dataclass(frozen=True, slots=True)
class NeuralQCBudget:
    """The declared improvement budget of one refinement episode.

    ``max_steps`` caps the number of proposals reviewed for one instance;
    ``stall_patience`` declares after how many consecutive non-improving
    reviews the episode terminates as converged instead of running to the
    budget.
    """

    max_steps: int
    stall_patience: int = 1

    def __post_init__(self) -> None:
        for name in ("max_steps", "stall_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stall_patience > self.max_steps:
            raise ValueError("stall_patience cannot exceed max_steps")


@dataclass(frozen=True, slots=True)
class NeuralQCRefinementResult:
    """Complete, honest record of one Neural-QC refinement episode.

    Every counter covers the whole episode; ``reviews`` preserves rejected
    and invalid proposals alongside accepted ones. ``final_verification``
    is the independent check of the published plan, whose scope remains the
    generated columns of the classical start.
    """

    instance_id: str
    status: str
    termination_reason: str
    initial_bars: int
    final_bars: int
    steps_taken: int
    accepted_steps: int
    invalid_steps: int
    initial_patterns: tuple[tuple[int, ...], ...]
    initial_column_values: tuple[int, ...]
    final_patterns: tuple[tuple[int, ...], ...]
    final_column_values: tuple[int, ...]
    final_verification: PlanVerification
    reviews: tuple[ProposalReview, ...]
    total_runtime_seconds: float
    schema_version: str = NEURAL_QC_PIPELINE_SCHEMA_VERSION

    @property
    def bars_saved(self) -> int:
        """Return the verified bar reduction against the classical start."""

        return self.initial_bars - self.final_bars

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown refinement status: {self.status!r}")
        if self.termination_reason not in _TERMINATION_REASONS:
            raise ValueError(f"unknown termination reason: {self.termination_reason!r}")
        for name in ("initial_bars", "final_bars", "steps_taken"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("accepted_steps", "invalid_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.final_bars > self.initial_bars:
            raise ValueError("the incumbent can never end with more bars than it started with")
        if self.accepted_steps > self.steps_taken or self.invalid_steps > self.steps_taken:
            raise ValueError("step counters cannot exceed steps_taken")
        if len(self.reviews) != self.steps_taken:
            raise ValueError("reviews must cover exactly the steps taken")
        if len(self.initial_patterns) != len(self.initial_column_values):
            raise ValueError("initial patterns and values must have the same length")
        if len(self.final_patterns) != len(self.final_column_values):
            raise ValueError("final patterns and values must have the same length")
        if not math.isfinite(self.total_runtime_seconds) or self.total_runtime_seconds < 0:
            raise ValueError("total_runtime_seconds must be finite and non-negative")
        expected_status = STATUS_IMPROVED if self.bars_saved > 0 else STATUS_UNCHANGED
        if self.status != expected_status:
            raise ValueError(f"status {self.status!r} contradicts the measured bars")
        if not self.final_verification.feasible:
            raise ValueError("the final plan does not verify")
        if self.final_verification.number_of_stock_bars != self.final_bars:
            raise ValueError("final_verification disagrees with final_bars")


def run_neural_quality_refinement(
    instance: AnyCuttingStockInstance,
    instance_id: str,
    agent: QualityAgent,
    *,
    budget: NeuralQCBudget,
    verification_tolerance: float = 1e-9,
) -> NeuralQCRefinementResult:
    """Refine the classical restricted integer solution with one quality agent.

    The classical column-generation loop must converge to a verified integer
    solution; anything else raises instead of being silently skipped. Each
    agent proposal is reviewed by the independent verifier inside
    :class:`QualityRefinementEnv`; acceptance requires a strict verified bar
    reduction, and the loop stops at the declared budget or once
    ``stall_patience`` consecutive reviews bring no improvement.
    """

    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")
    if not callable(getattr(agent, "propose", None)):
        raise ValueError("agent must expose a propose(observation) method")
    if not isinstance(budget, NeuralQCBudget):
        raise ValueError("budget must be a NeuralQCBudget")
    if not math.isfinite(verification_tolerance) or verification_tolerance < 0:
        raise ValueError("verification_tolerance must be finite and non-negative")

    started = perf_counter()
    cg_result = ColumnGeneration(instance, instance_id=instance_id).solve()
    if cg_result.status != "converged":
        raise ValueError(
            f"classical column generation did not converge on {instance_id}: "
            f"status {cg_result.status} ({cg_result.termination_reason})"
        )
    integer_master = cg_result.integer_master_result
    if (
        integer_master is None
        or integer_master.objective_value is None
        or cg_result.verification is None
        or not cg_result.verification.feasible
    ):
        raise ValueError(f"classical run on {instance_id} has no verified integer solution")

    environment = QualityRefinementEnv(
        instance,
        instance_id,
        cg_result.patterns,
        cg_result.patterns,
        integer_master.column_values,
        max_steps=budget.max_steps,
        tolerance=verification_tolerance,
    )

    termination_reason = TERMINATION_BUDGET_EXHAUSTED
    consecutive_rejections = 0
    while environment.steps_taken < budget.max_steps:
        proposal = agent.propose(environment.observation)
        step = environment.step(proposal)
        if step.accepted:
            consecutive_rejections = 0
            continue
        consecutive_rejections += 1
        if consecutive_rejections >= budget.stall_patience:
            termination_reason = TERMINATION_CONVERGED
            break

    final_observation = environment.observation
    final_verification = verify_plan(
        instance,
        final_observation.solution_patterns,
        final_observation.solution_column_values,
        verification_tolerance,
    )
    if (
        not final_verification.feasible
        or final_verification.number_of_stock_bars != environment.current_bars
    ):
        raise ValueError(
            f"the refined plan on {instance_id} failed its independent final verification"
        )

    reviews = environment.reviews
    return NeuralQCRefinementResult(
        instance_id=instance_id,
        status=STATUS_IMPROVED if environment.total_bars_saved > 0 else STATUS_UNCHANGED,
        termination_reason=termination_reason,
        initial_bars=environment.initial_bars,
        final_bars=environment.current_bars,
        steps_taken=len(reviews),
        accepted_steps=sum(1 for review in reviews if review.accepted),
        invalid_steps=sum(
            1
            for review in reviews
            if not (
                review.baseline_verification.feasible and review.proposal_verification.feasible
            )
        ),
        initial_patterns=cg_result.patterns,
        initial_column_values=integer_master.column_values,
        final_patterns=final_observation.solution_patterns,
        final_column_values=final_observation.solution_column_values,
        final_verification=final_verification,
        reviews=reviews,
        total_runtime_seconds=perf_counter() - started,
    )


__all__ = [
    "NEURAL_QC_PIPELINE_SCHEMA_VERSION",
    "STATUS_IMPROVED",
    "STATUS_UNCHANGED",
    "TERMINATION_BUDGET_EXHAUSTED",
    "TERMINATION_CONVERGED",
    "NeuralQCBudget",
    "NeuralQCRefinementResult",
    "run_neural_quality_refinement",
]
