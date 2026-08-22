"""Iterative-refinement RL environment on top of the quality-agent interface.

An episode is the iterative refinement of one restricted integer solution: it
starts from the verified integer solution of the classical column-generation
loop and every step submits one :class:`QualityAgentProposal` to the systematic
independent review of :func:`verify_proposal`. The observation is exactly the
versioned ``quality-agent-interface-v1`` input: the column pool generated so
far, which stays constant during the episode, plus the incumbent solution,
which only moves on verified strict improvements.

The reward follows the quality ordering of the project (bar count first):

- any proposal failing verification — invalid plan or broken baseline —
  receives the fixed strict ``invalid_plan_penalty`` (strictly negative);
- otherwise the reward is the verified signed bar reduction against the
  reviewed incumbent: at least one for an accepted improvement, zero for a
  valid equal-bar plan, negative for a valid worse plan. Under the documented
  material identity ``total_waste = bars * stock_length - requested_length``
  the requested length is constant, so every positive bar reduction is also a
  strictly smaller total loss; the incumbent waste stays exposed for reporting.

The environment never certifies optimality: the only termination is the
declared step budget. Rejected proposals leave the incumbent unchanged and
remain recorded in the episode history; failures are never silently filtered.
"""

import math
from dataclasses import dataclass

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import PlanVerification, verify_plan

from .quality_agent import (
    ProposalReview,
    QualityAgentInput,
    QualityAgentProposal,
    verify_proposal,
)

QUALITY_REFINEMENT_ENV_SCHEMA_VERSION = "quality-refinement-env-v1"
DEFAULT_INVALID_PLAN_PENALTY = -1.0


@dataclass(frozen=True, slots=True)
class RefinementStep:
    """One verified environment transition.

    ``observation`` is the next state offered to the agent; ``truncated``
    becomes true once the declared step budget is exhausted, which is the only
    termination an honest refinement episode has.
    """

    observation: QualityAgentInput
    reward: float
    accepted: bool
    review: ProposalReview
    steps_taken: int
    truncated: bool
    total_bars_saved: int


class QualityRefinementEnv:
    """Refinement episode whose every transition is scored from verification."""

    def __init__(
        self,
        instance: AnyCuttingStockInstance,
        instance_id: str,
        column_pool: tuple[tuple[int, ...], ...],
        solution_patterns: tuple[tuple[int, ...], ...],
        solution_column_values: tuple[int, ...],
        *,
        max_steps: int = 10,
        invalid_plan_penalty: float = DEFAULT_INVALID_PLAN_PENALTY,
        tolerance: float = 1e-9,
    ) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not math.isfinite(invalid_plan_penalty) or invalid_plan_penalty >= 0:
            raise ValueError("invalid_plan_penalty must be finite and strictly negative")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        self._instance = instance
        self._instance_id = instance_id
        self._column_pool = column_pool
        self._initial_patterns = solution_patterns
        self._initial_values = solution_column_values
        self._max_steps = max_steps
        self._invalid_plan_penalty = float(invalid_plan_penalty)
        self._tolerance = tolerance
        self._current_patterns: tuple[tuple[int, ...], ...]
        self._current_values: tuple[int, ...]
        self._current_verification: PlanVerification
        self._initial_verification: PlanVerification
        self._steps_taken: int
        self._total_bars_saved: int
        self._reviews: list[ProposalReview]
        self.reset()

    def reset(self) -> QualityAgentInput:
        """Restart the episode from the declared classical integer solution."""

        baseline = verify_plan(
            self._instance, self._initial_patterns, self._initial_values, self._tolerance
        )
        if not baseline.feasible:
            raise ValueError(f"the initial solution does not verify: {list(baseline.errors)}")
        self._current_patterns = self._initial_patterns
        self._current_values = self._initial_values
        self._current_verification = baseline
        self._initial_verification = baseline
        self._steps_taken = 0
        self._total_bars_saved = 0
        self._reviews = []
        return self.observation

    def step(self, proposal: QualityAgentProposal) -> RefinementStep:
        """Review one proposal, move the incumbent on acceptance and score it."""

        if self._steps_taken >= self._max_steps:
            raise RuntimeError("episode budget is exhausted; reset() starts a new episode")
        review = verify_proposal(self._instance, self.observation, proposal, self._tolerance)
        self._reviews.append(review)
        plans_feasible = (
            review.baseline_verification.feasible and review.proposal_verification.feasible
        )
        reward = float(review.bars_saved) if plans_feasible else self._invalid_plan_penalty
        if review.accepted:
            self._current_patterns = proposal.patterns
            self._current_values = proposal.column_values
            self._current_verification = review.proposal_verification
            self._total_bars_saved += review.bars_saved
        self._steps_taken += 1
        return RefinementStep(
            observation=self.observation,
            reward=reward,
            accepted=review.accepted,
            review=review,
            steps_taken=self._steps_taken,
            truncated=self._steps_taken >= self._max_steps,
            total_bars_saved=self._total_bars_saved,
        )

    @property
    def schema_version(self) -> str:
        """Return the versioned identity of this environment contract."""

        return QUALITY_REFINEMENT_ENV_SCHEMA_VERSION

    @property
    def observation(self) -> QualityAgentInput:
        """Return the current pool-and-solution state in the versioned shape."""

        return QualityAgentInput(
            instance_id=self._instance_id,
            stock_length=self._instance.stock_length,
            kerf=self._instance.kerf,
            piece_lengths=self._instance.piece_lengths,
            demands=self._instance.demands,
            column_pool=self._column_pool,
            solution_patterns=self._current_patterns,
            solution_column_values=self._current_values,
        )

    @property
    def max_steps(self) -> int:
        """Return the declared step budget of one episode."""

        return self._max_steps

    @property
    def steps_taken(self) -> int:
        """Return the number of reviewed proposals in the current episode."""

        return self._steps_taken

    @property
    def reviews(self) -> tuple[ProposalReview, ...]:
        """Return every review of the current episode, rejections included."""

        return tuple(self._reviews)

    @property
    def initial_bars(self) -> int:
        """Return the verified bar count of the classical starting solution."""

        return self._initial_verification.number_of_stock_bars

    @property
    def current_bars(self) -> int:
        """Return the verified bar count of the current incumbent solution."""

        return self._current_verification.number_of_stock_bars

    @property
    def current_waste(self) -> float:
        """Return the verified total material loss of the incumbent solution."""

        return self._current_verification.total_waste

    @property
    def total_bars_saved(self) -> int:
        """Return the accumulated verified bar reduction of the episode."""

        return self._total_bars_saved


__all__ = [
    "DEFAULT_INVALID_PLAN_PENALTY",
    "QUALITY_REFINEMENT_ENV_SCHEMA_VERSION",
    "QualityRefinementEnv",
    "RefinementStep",
]
