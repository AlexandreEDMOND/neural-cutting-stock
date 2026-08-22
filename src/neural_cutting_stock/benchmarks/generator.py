"""Deterministic synthetic instances for benchmark preparation."""

import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from numbers import Real

from neural_cutting_stock.problem import CuttingStockInstance

TIGHT_RATIO_LENGTH_DISTRIBUTION = "tight_ratio_v1"
AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION = "awkward_divisibility_v1"


@dataclass(frozen=True, slots=True)
class SyntheticInstanceGenerator:
    """Generate reproducible instances from a small, explicit configuration.

    The ``uniform_integer_v1`` labels keep the historical uniform sampler, and
    any other label stays an inert metadata name sampled uniformly as well.
    Two recognized structured profiles activate targeted samplers instead:
    ``tight_ratio_v1`` draws piece lengths from the narrow band whose kerf-aware
    natural multiplicity is exactly two pieces per bar, and
    ``awkward_divisibility_v1`` builds each demand as ``quotient *
    natural_multiplicity + remainder`` with a strictly positive remainder, so
    homogeneous patterns always overshoot the demand. Both profiles remain
    deterministic under the seed and stay inside their configured ranges.
    """

    seed: int
    stock_length: float = 100.0
    kerf: float = 0.0
    number_of_types: int = 3
    piece_length_range: tuple[int, int] = (10, 90)
    demand_range: tuple[int, int] = (1, 10)
    length_distribution: str = "uniform_integer_v1"
    demand_distribution: str = "uniform_integer_v1"

    name = "uniform_integer_v1"
    version = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.stock_length, bool)
            or not isinstance(self.stock_length, Real)
            or not math.isfinite(self.stock_length)
            or self.stock_length <= 0
        ):
            raise ValueError("stock_length must be finite and strictly positive")
        if (
            isinstance(self.kerf, bool)
            or not isinstance(self.kerf, Real)
            or not math.isfinite(self.kerf)
            or self.kerf < 0
        ):
            raise ValueError("kerf must be finite and non-negative")
        if (
            not isinstance(self.number_of_types, int)
            or isinstance(self.number_of_types, bool)
            or self.number_of_types <= 0
        ):
            raise ValueError("number_of_types must be a positive integer")
        _validate_range(self.piece_length_range, "piece_length_range")
        _validate_range(self.demand_range, "demand_range")
        _validate_text(self.length_distribution, "length_distribution")
        _validate_text(self.demand_distribution, "demand_distribution")
        if self.piece_length_range[1] + self.kerf > self.stock_length:
            raise ValueError("every piece must fit on a bar with kerf")

    def generate(self) -> CuttingStockInstance:
        """Return one instance; the same configuration always yields the same data."""

        rng = random.Random(self.seed)
        lengths = self._sample_lengths(rng)
        demands = self._sample_demands(rng, lengths)
        return CuttingStockInstance(self.stock_length, self.kerf, lengths, demands)

    def _sample_lengths(self, rng: random.Random) -> list[float]:
        lower, upper = self.piece_length_range
        available_lengths = range(lower, upper + 1)
        if self.length_distribution == TIGHT_RATIO_LENGTH_DISTRIBUTION:
            window_lower, window_upper = _tight_ratio_window(
                self.stock_length, self.kerf
            )
            available_lengths = [
                value
                for value in available_lengths
                if window_lower <= value <= window_upper
            ]
            if not available_lengths:
                raise ValueError(
                    "the tight-ratio multiplicity-two window does not intersect "
                    "piece_length_range"
                )
        if self.number_of_types > len(available_lengths):
            raise ValueError("number_of_types exceeds the available length values")
        return rng.sample(list(available_lengths), self.number_of_types)

    def _sample_demands(
        self, rng: random.Random, lengths: list[float]
    ) -> list[int]:
        demand_lower, demand_upper = self.demand_range
        if self.demand_distribution != AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION:
            return [rng.randint(demand_lower, demand_upper) for _ in lengths]
        demands: list[int] = []
        for length in lengths:
            multiplicity = _natural_multiplicity(self.stock_length, self.kerf, length)
            remainder_upper = min(multiplicity - 1, demand_upper - multiplicity)
            valid_remainders = [
                remainder
                for remainder in range(1, remainder_upper + 1)
                if max(1, -(-(demand_lower - remainder) // multiplicity))
                <= (demand_upper - remainder) // multiplicity
            ]
            if not valid_remainders:
                if multiplicity >= 2:
                    raise ValueError(
                        "demand_range admits no demand non-divisible by the "
                        f"natural multiplicity {multiplicity} of length {length}"
                    )
                demands.append(rng.randint(demand_lower, demand_upper))
                continue
            remainder = rng.choice(valid_remainders)
            quotient_upper = (demand_upper - remainder) // multiplicity
            quotient_lower = max(1, -(-(demand_lower - remainder) // multiplicity))
            quotient = rng.randint(quotient_lower, quotient_upper)
            demands.append(quotient * multiplicity + remainder)
        return demands

    @property
    def instance_id(self) -> str:
        """Return a stable identifier for the generated, normalized instance."""

        instance = self.generate()
        payload = json.dumps(
            {
                "stock_length": instance.stock_length,
                "kerf": instance.kerf,
                "piece_lengths": instance.piece_lengths,
                "demands": instance.demands,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return sha256(payload).hexdigest()

    @property
    def family_id(self) -> str:
        """Return a stable identifier for the generator family, excluding its seed."""

        payload = json.dumps(
            {
                "generator_name": self.name,
                "generator_version": self.version,
                "stock_length": self.stock_length,
                "kerf": self.kerf,
                "number_of_types": self.number_of_types,
                "piece_length_range": self.piece_length_range,
                "demand_range": self.demand_range,
                "length_distribution": self.length_distribution,
                "demand_distribution": self.demand_distribution,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return sha256(payload).hexdigest()


def _tight_ratio_window(stock_length: float, kerf: float) -> tuple[int, int]:
    """Return the integer lengths admitting exactly two kerf-aware pieces per bar.

    A length admits two pieces when ``stock_length/3 < length + kerf <=``
    ``stock_length/2``; the returned bounds follow from that band using the
    decimal spelling of the configuration.
    """

    stock = Decimal(str(stock_length))
    kerf_width = Decimal(str(kerf))
    upper_bound = stock / 2 - kerf_width
    lower_bound = stock / 3 - kerf_width
    if upper_bound <= 0:
        raise ValueError("no length admits two pieces per bar with this kerf")
    return int(lower_bound) + 1, int(upper_bound)


def _natural_multiplicity(stock_length: float, kerf: float, length: float) -> int:
    """Return how many copies of ``length`` fit on one bar under the kerf rule."""

    return int(
        Decimal(str(stock_length)) // (Decimal(str(length)) + Decimal(str(kerf)))
    )


def _validate_range(value: tuple[int, int], name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        or value[0] <= 0
        or value[0] > value[1]
    ):
        raise ValueError(f"{name} must be an increasing pair of positive integers")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
