"""Shared validation for generated Cutting Stock patterns."""

from neural_cutting_stock.problem import AnyCuttingStockInstance


def validate_patterns(
    instance: AnyCuttingStockInstance,
    patterns: tuple[tuple[int, ...], ...],
) -> None:
    """Validate the patterns accepted by a restricted master problem.

    A pattern is accepted when its kerf-aware capacity fits at least one
    declared stock length, which for a declared multi-format instance is the
    largest one (see ``MultiFormatCuttingStockInstance.stock_length``).
    """

    if not patterns:
        raise ValueError("patterns must not be empty")
    if len(set(patterns)) != len(patterns):
        raise ValueError("patterns must be distinct")
    for pattern in patterns:
        if instance.capacity_used(pattern) > instance.stock_length:
            raise ValueError("patterns must respect stock capacity")
        if any(
            count > demand
            for count, demand in zip(pattern, instance.demands, strict=True)
        ):
            raise ValueError("patterns must not exceed demands")
        if not any(pattern):
            raise ValueError("patterns must be non-empty")
