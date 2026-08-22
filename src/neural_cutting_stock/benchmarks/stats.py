"""Shared numeric aggregation helpers for benchmark and publication modules."""

from collections.abc import Iterable


def median(values: Iterable[float | None]) -> float | None:
    """Return the median of the non-None values as a float, or None when empty."""

    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return float(numbers[middle])
    return (numbers[middle - 1] + numbers[middle]) / 2
