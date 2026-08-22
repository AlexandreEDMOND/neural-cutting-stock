"""Independent verification of integer Cutting Stock plans."""

import math
from dataclasses import dataclass
from numbers import Integral

from neural_cutting_stock.problem import AnyCuttingStockInstance


@dataclass(frozen=True, slots=True)
class PlanVerification:
    """Feasibility and material accounting for a restricted integer plan."""

    feasible: bool
    errors: tuple[str, ...]
    number_of_stock_bars: int
    produced_counts: tuple[int, ...]
    requested_length: float
    produced_length: float
    overproduction_length: float
    kerf_loss: float
    trim_loss: float
    total_waste: float


def verify_plan(
    instance: AnyCuttingStockInstance,
    patterns: tuple[tuple[int, ...], ...],
    column_values: tuple[int, ...],
    tolerance: float = 1e-9,
) -> PlanVerification:
    """Verify an integer plan without relying on a solver result status.

    For a declared multi-format instance, capacity and trim accounting use
    the largest declared format, the only format a generated plan needs.
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if len(patterns) != len(column_values):
        raise ValueError("patterns and column_values must have the same length")

    errors: list[str] = []
    produced_counts = [0] * instance.number_of_types
    number_of_bars = 0
    produced_length = 0.0
    kerf_loss = 0.0
    trim_loss = 0.0

    for index, (pattern, value) in enumerate(zip(patterns, column_values, strict=True)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            errors.append(f"column_values[{index}] must be a non-negative integer")
            continue
        try:
            capacity = instance.capacity_used(pattern)
        except ValueError as error:
            errors.append(f"patterns[{index}] is invalid: {error}")
            continue
        if not any(pattern):
            errors.append(f"patterns[{index}] must be non-empty")
            continue
        if capacity > instance.stock_length + tolerance:
            errors.append(f"patterns[{index}] exceeds stock capacity")
            continue
        if any(
            count > demand
            for count, demand in zip(pattern, instance.demands, strict=True)
        ):
            errors.append(f"patterns[{index}] exceeds demand bounds")
            continue

        multiplicity = int(value)
        number_of_bars += multiplicity
        for piece_index, count in enumerate(pattern):
            produced_counts[piece_index] += multiplicity * count
        piece_length = sum(
            length * count
            for length, count in zip(instance.piece_lengths, pattern, strict=True)
        )
        piece_count = sum(pattern)
        produced_length += multiplicity * piece_length
        kerf_loss += multiplicity * instance.kerf * piece_count
        trim_loss += multiplicity * (instance.stock_length - capacity)

    if any(
        produced < demand
        for produced, demand in zip(produced_counts, instance.demands, strict=True)
    ):
        errors.append("plan does not cover every demand")

    requested_length = sum(
        length * demand
        for length, demand in zip(instance.piece_lengths, instance.demands, strict=True)
    )
    overproduction_length = produced_length - requested_length
    total_waste = number_of_bars * instance.stock_length - requested_length
    expected_waste = overproduction_length + kerf_loss + trim_loss
    if abs(total_waste - expected_waste) > tolerance:
        errors.append("material balance identity is not satisfied")

    return PlanVerification(
        feasible=not errors,
        errors=tuple(errors),
        number_of_stock_bars=number_of_bars,
        produced_counts=tuple(produced_counts),
        requested_length=requested_length,
        produced_length=produced_length,
        overproduction_length=overproduction_length,
        kerf_loss=kerf_loss,
        trim_loss=trim_loss,
        total_waste=total_waste,
    )
