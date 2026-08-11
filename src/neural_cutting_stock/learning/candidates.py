"""Deterministic candidate pools for learned column selection."""

import math
from collections.abc import Sequence
from itertools import product
from numbers import Real

from neural_cutting_stock.problem import CuttingStockInstance

from .interfaces import PatternCandidate


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
    if max_candidates is not None and (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer when present")

    number_of_types = instance.number_of_types
    excluded = set(current_patterns)
    for pattern in excluded:
        _validate_pattern(pattern, number_of_types, "current_patterns")

    candidates = []
    for pattern in product(*(range(demand + 1) for demand in instance.demands)):
        if not any(pattern) or pattern in excluded:
            continue
        if instance.capacity_used(pattern) > instance.stock_length:
            continue
        reduced_cost = 1.0 - math.fsum(
            dual * count for dual, count in zip(dual_values, pattern, strict=True)
        )
        candidates.append(PatternCandidate(pattern, reduced_cost))

    candidates.sort(key=lambda candidate: (candidate.reduced_cost, candidate.pattern))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return tuple(candidates)


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
