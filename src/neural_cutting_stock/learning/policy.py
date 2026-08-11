"""Deterministic policies for selecting scored pricing candidates."""

from collections.abc import Sequence

from .interfaces import (
    ColumnScoringModel,
    ColumnSelectionDecision,
    PatternCandidate,
    PricingState,
)


class LearnedColumnSelectionPolicy:
    """Select the highest-scoring candidates subject to an explicit budget.

    The policy only chooses among candidates supplied by the classical pricing
    layer. It neither generates candidates nor declares pricing convergence.
    """

    def __init__(self, model: ColumnScoringModel, candidate_budget: int | None = None) -> None:
        if candidate_budget is not None and (
            isinstance(candidate_budget, bool)
            or not isinstance(candidate_budget, int)
            or candidate_budget < 1
        ):
            raise ValueError("candidate_budget must be a positive integer when present")
        self.model = model
        self.candidate_budget = candidate_budget

    def select(
        self, state: PricingState, candidates: Sequence[PatternCandidate]
    ) -> ColumnSelectionDecision:
        """Score candidates and return at most ``candidate_budget`` patterns."""

        candidates = tuple(candidates)
        if len({candidate.pattern for candidate in candidates}) != len(candidates):
            raise ValueError("candidates must contain distinct patterns")
        scored_candidates = tuple(self.model.score(state, candidates))
        if len(scored_candidates) != len(candidates):
            raise ValueError("model must return one score per candidate")
        candidate_patterns = {candidate.pattern for candidate in candidates}
        if {score.pattern for score in scored_candidates} != candidate_patterns:
            raise ValueError("model scores must correspond to the supplied candidates")

        ranked = sorted(scored_candidates, key=lambda score: (-score.score, score.pattern))
        if self.candidate_budget is not None:
            ranked = ranked[: self.candidate_budget]
        selected_patterns = tuple(score.pattern for score in ranked)
        return ColumnSelectionDecision(scored_candidates, selected_patterns)


__all__ = ["LearnedColumnSelectionPolicy"]
