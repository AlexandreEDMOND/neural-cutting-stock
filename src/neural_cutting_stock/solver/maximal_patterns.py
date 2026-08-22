"""Exhaustive enumeration of maximal cutting patterns for bounded instances.

The complete pattern set of an instance (docs/formulation.md, sections 3-4)
holds the non-empty piece-count vectors bounded by the demands whose kerf-aware
capacity fits one bar. A pattern is maximal when no further piece can be added
without violating capacity or a demand bound. Every plan of the complete
integer master extends bar by bar to a plan over maximal patterns only with the
same number of bars, so maximal patterns suffice to express optimal plans of
the complete master while keeping the enumeration tractable.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal

from neural_cutting_stock.problem import AnyCuttingStockInstance


@dataclass(frozen=True, slots=True)
class MaximalPatternLimits:
    """Deterministic bounds keeping exhaustive pattern enumeration tractable.

    ``max_search_space_size`` refuses instances whose demand-bounded,
    capacity-clipped count grid exceeds the declared size before iteration
    starts. ``max_patterns`` caps how many maximal patterns may be produced;
    exceeding it raises instead of silently truncating an exact reference.
    """

    max_search_space_size: int = 10_000_000
    max_patterns: int = 100_000


class PatternEnumerationLimitExceeded(RuntimeError):
    """Raised when a maximal-pattern enumeration guard is exceeded."""


def iter_maximal_patterns(
    instance: AnyCuttingStockInstance,
    limits: MaximalPatternLimits | None = None,
) -> Iterator[tuple[int, ...]]:
    """Yield every maximal pattern of ``instance`` in lexicographic order.

    Generation is lazy: patterns stream one at a time in a deterministic order
    derived from the normalized instance, and no collection is materialized.
    The memory guards are checked eagerly so unsuitable instances fail before
    any pattern is produced.
    """

    if limits is None:
        limits = MaximalPatternLimits()
    _validate_limits(limits)
    stock = Decimal(str(instance.stock_length))
    weights = tuple(
        Decimal(str(length)) + Decimal(str(instance.kerf))
        for length in instance.piece_lengths
    )
    max_counts: list[int] = []
    search_space_size = 1
    for weight, demand in zip(weights, instance.demands, strict=True):
        highest = min(int(demand), int(stock // weight))
        max_counts.append(highest)
        search_space_size *= highest + 1
    if search_space_size > limits.max_search_space_size:
        raise PatternEnumerationLimitExceeded(
            "exhaustive enumeration refused: search space "
            f"{search_space_size} exceeds max_search_space_size="
            f"{limits.max_search_space_size}"
        )
    return _generate(
        instance.demands, weights, stock, tuple(max_counts), limits.max_patterns
    )


def _generate(
    demands: tuple[int, ...],
    weights: tuple[Decimal, ...],
    stock: Decimal,
    max_counts: tuple[int, ...],
    max_patterns: int,
) -> Iterator[tuple[int, ...]]:
    """Walk the count grid depth-first and stream maximal patterns."""

    counts = [0] * len(weights)

    def walk(index: int, remaining: Decimal) -> Iterator[tuple[int, ...]]:
        if index == len(weights):
            if _is_maximal(counts, demands, weights, remaining):
                yield tuple(counts)
            return
        weight = weights[index]
        highest = min(max_counts[index], int(remaining // weight))
        for count in range(highest + 1):
            counts[index] = count
            yield from walk(index + 1, remaining - count * weight)

    for produced, pattern in enumerate(walk(0, stock), start=1):
        if produced > max_patterns:
            raise PatternEnumerationLimitExceeded(
                "exhaustive enumeration stopped: more than "
                f"max_patterns={max_patterns} maximal patterns exist"
            )
        yield pattern


def _is_maximal(
    counts: list[int],
    demands: tuple[int, ...],
    weights: tuple[Decimal, ...],
    remaining: Decimal,
) -> bool:
    return all(
        count == demand or weight > remaining
        for count, demand, weight in zip(counts, demands, weights, strict=True)
    )


def _validate_limits(limits: MaximalPatternLimits) -> None:
    for name in ("max_search_space_size", "max_patterns"):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
