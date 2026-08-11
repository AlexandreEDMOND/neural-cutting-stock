"""Permutation-invariant, fixed-width features for pricing observations."""

import math
from collections.abc import Sequence

from .interfaces import PatternCandidate, PricingState

FEATURE_SCHEMA_VERSION = "pricing-features-v1"


def pricing_features(state: PricingState, candidate: PatternCandidate) -> tuple[float, ...]:
    """Build a fixed-width feature vector for one state and candidate.

    Type-specific values are reduced to symmetric statistics, so reordering the
    piece types and applying the same reordering to a candidate leaves the
    vector unchanged. The vector width therefore does not depend on the number
    of types. The exact reduced cost is included as an observation from the
    classical pricing layer, not as a learned decision.
    """

    return pricing_features_batch(state, (candidate,))[0]


def pricing_features_batch(
    state: PricingState, candidates: Sequence[PatternCandidate]
) -> tuple[tuple[float, ...], ...]:
    """Build features for candidates while reusing the state computations.

    Reuse is limited to one batch because column generation rebuilds the state
    after every RMP iteration.
    """

    number_of_types = len(state.piece_lengths)
    total_demand = sum(state.demands)
    current_usage = [0] * number_of_types
    for pattern in state.current_patterns:
        for index, count in enumerate(pattern):
            current_usage[index] += count

    normalized_lengths = [length / state.stock_length for length in state.piece_lengths]
    normalized_demands = [demand / total_demand for demand in state.demands]
    normalized_duals = list(state.dual_values)
    normalized_current = [
        usage / demand for usage, demand in zip(current_usage, state.demands, strict=True)
    ]
    state_summary = []
    common_features = [
        state.stock_length,
        state.kerf / state.stock_length,
        float(number_of_types),
        total_demand,
    ]
    for values in (normalized_lengths, normalized_demands, normalized_duals, normalized_current):
        state_summary.extend(_summary(values))

    rows = []
    for candidate in candidates:
        if len(candidate.pattern) != number_of_types:
            raise ValueError("candidate pattern must follow state piece_lengths order")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in candidate.pattern
        ):
            raise ValueError("candidate pattern must contain non-negative integers")
        normalized_candidate = [
            count / demand for count, demand in zip(candidate.pattern, state.demands, strict=True)
        ]
        features = [*common_features, candidate.reduced_cost, *state_summary]
        features.extend(_summary(normalized_candidate))
        features.extend(
            (
                sum(
                    (length + state.kerf) * count
                    for length, count in zip(state.piece_lengths, candidate.pattern, strict=True)
                )
                / state.stock_length,
                math.fsum(
                    dual * count
                    for dual, count in zip(state.dual_values, candidate.pattern, strict=True)
                ),
            )
        )
        rows.append(tuple(features))
    return tuple(rows)


def _summary(values: Sequence[float]) -> tuple[float, ...]:
    """Return fixed-width symmetric statistics for a non-empty type sequence."""

    if not values:
        raise ValueError("cannot summarize an empty sequence")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return (mean, math.sqrt(variance), min(values), max(values), math.fsum(values))
