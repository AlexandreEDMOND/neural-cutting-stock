"""Versioned boundary objects for the Phase 9 quality agent.

The quality agent receives an instance view, the generated column pool and
the current restricted integer solution, and proposes refined cutting plans
whose columns absent from the pool are the supplementary proposed columns.
It never replaces the solver and never self-certifies: every proposal is
systematically reviewed by the independent classical verifier before it can
improve anything, and acceptance only ever means a verified strict reduction
of the bar count against the reviewed baseline, never global optimality.
"""

import math
from dataclasses import dataclass
from typing import Protocol

from neural_cutting_stock.problem import AnyCuttingStockInstance
from neural_cutting_stock.solver import PlanVerification, verify_plan

from .interfaces import _finite_nonnegative, _validate_pattern

QUALITY_AGENT_INTERFACE_SCHEMA_VERSION = "quality-agent-interface-v1"


@dataclass(frozen=True, slots=True)
class QualityAgentInput:
    """Inputs presented to a quality agent for one refinement step.

    The fields mirror the classical instance view (largest declared format
    for a multi-format instance), the column pool generated so far and the
    current integer solution of the restricted master.
    """

    instance_id: str
    stock_length: float
    kerf: float
    piece_lengths: tuple[float, ...]
    demands: tuple[int, ...]
    column_pool: tuple[tuple[int, ...], ...]
    solution_patterns: tuple[tuple[int, ...], ...]
    solution_column_values: tuple[int, ...]
    schema_version: str = QUALITY_AGENT_INTERFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if self.schema_version != QUALITY_AGENT_INTERFACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if len(self.piece_lengths) == 0 or len(self.piece_lengths) != len(self.demands):
            raise ValueError("piece_lengths and demands must have the same non-zero length")
        _finite_nonnegative("stock_length", self.stock_length)
        if self.stock_length <= 0:
            raise ValueError("stock_length must be positive")
        _finite_nonnegative("kerf", self.kerf)
        if any(length <= 0 or not math.isfinite(length) for length in self.piece_lengths):
            raise ValueError("piece_lengths must contain finite positive values")
        if any(
            not isinstance(demand, int) or isinstance(demand, bool) or demand <= 0
            for demand in self.demands
        ):
            raise ValueError("demands must contain positive integers")
        if len(self.solution_patterns) != len(self.solution_column_values):
            raise ValueError(
                "solution_patterns and solution_column_values must have the same length"
            )
        for pattern in self.column_pool:
            _validate_pattern(pattern, len(self.piece_lengths), "column_pool")
        for pattern in self.solution_patterns:
            _validate_pattern(pattern, len(self.piece_lengths), "solution_patterns")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.solution_column_values
        ):
            raise ValueError("solution_column_values must contain non-negative integers")


@dataclass(frozen=True, slots=True)
class QualityAgentProposal:
    """A refined cutting plan proposed by a quality agent.

    ``patterns`` and ``column_values`` describe one complete candidate plan;
    every proposed pattern absent from the current pool is a supplementary
    proposed column. The proposal carries no guarantee of its own.
    """

    patterns: tuple[tuple[int, ...], ...]
    column_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.patterns) != len(self.column_values):
            raise ValueError("patterns and column_values must have the same length")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.column_values
        ):
            raise ValueError("column_values must contain non-negative integers")
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("patterns must be distinct")


class QualityAgent(Protocol):
    """Minimal proposal contract consumed by future Neural-QC orchestration."""

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal: ...


@dataclass(frozen=True, slots=True)
class ProposalReview:
    """Outcome of the systematic independent verification of one proposal.

    Both verifications are kept whatever the outcome: failures are recorded,
    never silently filtered. ``accepted`` is true only when both plans are
    feasible and the proposal uses strictly fewer bars than the baseline.
    """

    accepted: bool
    errors: tuple[str, ...]
    baseline_verification: PlanVerification
    proposal_verification: PlanVerification
    baseline_bars: int
    proposed_bars: int

    @property
    def bars_saved(self) -> int:
        """Return the verified bar reduction, negative when the plan is worse."""

        return self.baseline_bars - self.proposed_bars


def verify_proposal(
    instance: AnyCuttingStockInstance,
    observation: QualityAgentInput,
    proposal: QualityAgentProposal,
    tolerance: float = 1e-9,
) -> ProposalReview:
    """Review an agent proposal through the independent classical verifier.

    The contract is systematic and independent of the agent:

    1. the incoming solution is itself verified first — a broken baseline can
       never serve as a comparison point;
    2. the proposal is verified for capacity under the documented kerf rule,
       demand coverage and material balance by :func:`verify_plan`, exactly
       like any solver-produced plan;
    3. acceptance requires both plans feasible and a strict bar reduction;
       equal or worse proposals are rejected and preserved in the review.
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    _check_observation_matches_instance(instance, observation)

    baseline = verify_plan(
        instance,
        observation.solution_patterns,
        observation.solution_column_values,
        tolerance,
    )
    candidate = verify_plan(instance, proposal.patterns, proposal.column_values, tolerance)

    errors = [f"baseline: {message}" for message in baseline.errors]
    errors.extend(f"proposal: {message}" for message in candidate.errors)
    if (
        baseline.feasible
        and candidate.feasible
        and candidate.number_of_stock_bars >= baseline.number_of_stock_bars
    ):
        errors.append("proposal does not reduce the number of stock bars")

    return ProposalReview(
        accepted=not errors,
        errors=tuple(errors),
        baseline_verification=baseline,
        proposal_verification=candidate,
        baseline_bars=baseline.number_of_stock_bars,
        proposed_bars=candidate.number_of_stock_bars,
    )


def _check_observation_matches_instance(
    instance: AnyCuttingStockInstance,
    observation: QualityAgentInput,
) -> None:
    if instance.stock_length != observation.stock_length:
        raise ValueError("observation stock_length does not match the instance")
    if instance.kerf != observation.kerf:
        raise ValueError("observation kerf does not match the instance")
    if tuple(instance.piece_lengths) != tuple(observation.piece_lengths):
        raise ValueError("observation piece_lengths do not match the instance")
    if tuple(instance.demands) != tuple(observation.demands):
        raise ValueError("observation demands do not match the instance")


__all__ = [
    "QUALITY_AGENT_INTERFACE_SCHEMA_VERSION",
    "ProposalReview",
    "QualityAgent",
    "QualityAgentInput",
    "QualityAgentProposal",
    "verify_proposal",
]
