"""Declared multi-stock-length variant of the one-dimensional Cutting Stock problem."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .instance import CuttingStockInstance

MULTI_STOCK_FORMAT_SCHEMA_VERSION = "multi-stock-format-v1"

_STOCK_FORMAT_COUNT_RANGE = range(2, 4)


@dataclass(frozen=True, slots=True)
class MultiFormatCuttingStockInstance:
    """A validated 1D Cutting Stock instance cutting pieces from several stock lengths.

    This stays a monodimensional variant: every bar still carries pieces along a
    single axis, each cut pattern is tied to one declared stock length, and the
    documented conservative kerf rule reserves ``kerf`` capacity for every produced
    piece. Every demanded piece must fit alone on the largest declared stock length;
    shorter formats remain available to the patterns they can host.

    The instance also satisfies the single-format surface consumed by the
    classical components (``stock_length``, ``initial_patterns``, capacity and
    demand accessors): see :attr:`stock_length` for why solving that view is
    exact for this variant.
    """

    stock_lengths: tuple[float, ...]
    kerf: float
    piece_lengths: tuple[float, ...]
    demands: tuple[int, ...]

    _reference: CuttingStockInstance = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        stock_lengths = _validated_stock_lengths(self.stock_lengths)
        # Reuse the single-format invariants against the largest declared bar: it
        # normalizes and merges piece types, validates demands and enforces that
        # every piece fits with kerf on at least one declared stock length.
        reference = CuttingStockInstance(
            max(stock_lengths), self.kerf, self.piece_lengths, self.demands
        )
        object.__setattr__(self, "stock_lengths", stock_lengths)
        object.__setattr__(self, "kerf", reference.kerf)
        object.__setattr__(self, "piece_lengths", reference.piece_lengths)
        object.__setattr__(self, "demands", reference.demands)
        object.__setattr__(self, "_reference", reference)

    @property
    def number_of_types(self) -> int:
        """Return the number of distinct piece lengths."""

        return len(self.piece_lengths)

    @property
    def stock_length(self) -> float:
        """Return the capacity view solved by the classical components.

        This is the largest declared stock length, identical to
        ``largest_stock_length``; it repeats the single-format name so the
        column-generation loop, the exact pricing and the complete-master
        reference accept the declared variant unchanged. The reduction is
        exact rather than an approximation:

        - a pattern fits at least one declared format if and only if it fits
          the largest one, because capacity grows monotonically with the
          piece counts while the formats are sorted ascending;
        - every bar costs one unit whatever its declared format, so each bar
          of any feasible plan can be replaced by the same pattern cut from
          the largest declared format without changing coverage or the bar
          count.

        Both the LP relaxation and the integer master therefore admit optima
        over largest-format patterns equal to those of the declared variant,
        and the non-negative covering duals make the knapsack pricing over
        this capacity bound find the best column among all format choices.
        """

        return self.stock_lengths[-1]

    @property
    def largest_stock_length(self) -> float:
        """Return the largest declared stock length."""

        return self.stock_lengths[-1]

    def initial_patterns(self) -> tuple[tuple[int, ...], ...]:
        """Return demand-bounded homogeneous patterns on the largest format.

        One pattern per piece type, each carrying as many copies of its type
        as the largest declared format hosts within the demand. Because every
        piece fits alone on that format, each count is at least one and the
        initial restricted master stays feasible.
        """

        return self._reference.initial_patterns()

    def capacity_used(self, pattern: tuple[int, ...]) -> float:
        """Return capacity consumed by a pattern under the documented kerf rule."""

        return self._reference.capacity_used(pattern)

    def fits_on(self, pattern: tuple[int, ...], stock_length: float) -> bool:
        """Return whether a pattern fits on one declared stock length with kerf."""

        if not any(stock_length == length for length in self.stock_lengths):
            raise ValueError("stock_length must be one of the declared stock lengths")
        return self.capacity_used(pattern) <= stock_length

    def to_dict(self) -> dict[str, Any]:
        """Return the flat, JSON-ready representation of the declared variant."""

        return {
            "schema_version": MULTI_STOCK_FORMAT_SCHEMA_VERSION,
            "stock_lengths": list(self.stock_lengths),
            "kerf": self.kerf,
            "piece_lengths": list(self.piece_lengths),
            "demands": list(self.demands),
        }

    @classmethod
    def from_dict(cls, value: object) -> "MultiFormatCuttingStockInstance":
        """Build an instance from the persisted JSON representation."""

        if not isinstance(value, dict):
            raise ValueError("multi-stock-format instance must be a JSON object")
        if value.get("schema_version") != MULTI_STOCK_FORMAT_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        expected_keys = {"schema_version", "stock_lengths", "kerf", "piece_lengths", "demands"}
        missing_keys = expected_keys - value.keys()
        if missing_keys:
            raise ValueError(f"missing fields: {sorted(missing_keys)}")
        unknown_keys = value.keys() - expected_keys
        if unknown_keys:
            raise ValueError(f"unknown fields: {sorted(unknown_keys)}")
        return cls(value["stock_lengths"], value["kerf"], value["piece_lengths"], value["demands"])


def _validated_stock_lengths(value: Iterable[object]) -> tuple[float, ...]:
    try:
        candidates = tuple(value)
    except TypeError as error:
        raise ValueError("stock_lengths must be iterable") from error
    if len(candidates) not in _STOCK_FORMAT_COUNT_RANGE:
        raise ValueError("stock_lengths must declare between two and three stock lengths")
    lengths = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, Real):
            raise ValueError("stock_lengths must contain real numbers")
        try:
            number = float(candidate)
        except OverflowError as error:
            raise ValueError("stock_lengths must contain finite numbers") from error
        if not math.isfinite(number):
            raise ValueError("stock_lengths must contain finite numbers")
        if number <= 0:
            raise ValueError("stock_lengths must contain strictly positive numbers")
        lengths.append(number)
    if len(set(lengths)) != len(lengths):
        raise ValueError("stock_lengths must be distinct")
    return tuple(sorted(lengths))
