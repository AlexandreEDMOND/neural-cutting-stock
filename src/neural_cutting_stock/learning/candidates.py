"""Deterministic candidate pools for learned column selection."""

import math
from collections.abc import Sequence
from heapq import nsmallest
from itertools import product
from numbers import Real

from neural_cutting_stock.problem import CuttingStockInstance

from .interfaces import PatternCandidate, _validate_positive_integer


def deterministic_candidate_pool(
    instance: CuttingStockInstance,
    dual_values: Sequence[float],
    current_patterns: Sequence[tuple[int, ...]] = (),
    max_candidates: int | None = None,
) -> tuple[PatternCandidate, ...]:
    """Return a reproducible, demand-bounded pool of feasible new patterns.

    Patterns are ordered by increasing reduced cost and then lexicographically.
    This helper is intentionally separate from :class:`ExactPricing`: the exact
    pricing call remains the convergence guard and is not changed by this pool.
    """

    _validate_dual_values(instance, dual_values)
    if max_candidates is not None:
        _validate_positive_integer(max_candidates, "max_candidates")

    number_of_types = instance.number_of_types
    excluded = set(current_patterns)
    for pattern in excluded:
        _validate_pattern(pattern, number_of_types, "current_patterns")

    def candidate_patterns():
        max_counts = tuple(
            min(
                demand,
                math.floor(
                    instance.stock_length
                    / (piece_length + instance.kerf)
                ),
            )
            for piece_length, demand in zip(
                instance.piece_lengths, instance.demands, strict=True
            )
        )
        for pattern in product(*(range(max_count + 1) for max_count in max_counts)):
            if not any(pattern) or pattern in excluded:
                continue
            if instance.capacity_used(pattern) > instance.stock_length:
                continue
            reduced_cost = 1.0 - math.fsum(
                dual * count for dual, count in zip(dual_values, pattern, strict=True)
            )
            yield PatternCandidate(pattern, reduced_cost)

    candidates = candidate_patterns()
    if max_candidates is not None:
        return tuple(
            nsmallest(
                max_candidates,
                candidates,
                key=lambda candidate: (candidate.reduced_cost, candidate.pattern),
            )
        )

    result = list(candidates)
    result.sort(key=lambda candidate: (candidate.reduced_cost, candidate.pattern))
    return tuple(result)


def _validate_dual_values(instance: CuttingStockInstance, dual_values: Sequence[float]) -> None:
    if len(dual_values) != instance.number_of_types or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        for value in dual_values
    ):
        raise ValueError("dual_values must be finite and non-negative")


def _validate_pattern(pattern: tuple[int, ...], number_of_types: int, name: str) -> None:
    if len(pattern) != number_of_types or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in pattern
    ):
        raise ValueError(f"{name} patterns must contain non-negative integers")
