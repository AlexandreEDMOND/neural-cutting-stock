"""Validated instances for the one-dimensional Cutting Stock problem."""

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class CuttingStockInstance:
    """A normalized, validated 1D Cutting Stock instance.

    The kerf convention reserves ``kerf`` capacity for every produced piece,
    including the last one on a bar.
    """

    stock_length: float
    kerf: float
    piece_lengths: tuple[float, ...]
    demands: tuple[int, ...]

    def __post_init__(self) -> None:
        stock_length = _finite_real(self.stock_length, "stock_length")
        kerf = _finite_real(self.kerf, "kerf")
        if stock_length <= 0:
            raise ValueError("stock_length must be strictly positive")
        if kerf < 0:
            raise ValueError("kerf must be non-negative")

        lengths = tuple(_finite_real(value, "piece_lengths") for value in self.piece_lengths)
        demands = tuple(_positive_integer(value, "demands") for value in self.demands)
        if not lengths:
            raise ValueError("piece_lengths must not be empty")
        if len(lengths) != len(demands):
            raise ValueError("piece_lengths and demands must have the same length")

        normalized: dict[float, int] = {}
        for length, demand in zip(lengths, demands, strict=True):
            if length <= 0:
                raise ValueError("piece lengths must be strictly positive")
            if length + kerf > stock_length:
                raise ValueError("every piece must fit on a bar with kerf")
            normalized[length] = normalized.get(length, 0) + demand

        object.__setattr__(self, "stock_length", stock_length)
        object.__setattr__(self, "kerf", kerf)
        object.__setattr__(self, "piece_lengths", tuple(sorted(normalized)))
        object.__setattr__(
            self, "demands", tuple(normalized[length] for length in sorted(normalized))
        )

    @property
    def number_of_types(self) -> int:
        """Return the number of distinct piece lengths."""

        return len(self.piece_lengths)

    def capacity_used(self, pattern: tuple[int, ...]) -> float:
        """Return capacity consumed by a pattern under the documented kerf rule."""

        if len(pattern) != self.number_of_types:
            raise ValueError("pattern must contain one count per piece type")
        if any(
            not isinstance(count, Integral) or isinstance(count, bool) or count < 0
            for count in pattern
        ):
            raise ValueError("pattern counts must be non-negative integers")
        return sum(
            (length + self.kerf) * count
            for length, count in zip(self.piece_lengths, pattern, strict=True)
        )


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return int(value)
