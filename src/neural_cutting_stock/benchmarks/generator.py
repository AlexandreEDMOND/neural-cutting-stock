"""Deterministic synthetic instances for benchmark preparation."""

import json
import math
import random
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real

from neural_cutting_stock.problem import CuttingStockInstance


@dataclass(frozen=True, slots=True)
class SyntheticInstanceGenerator:
    """Generate reproducible instances from a small, explicit configuration."""

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
        if self.piece_length_range[1] > self.stock_length - self.kerf:
            raise ValueError("piece_length_range contains pieces that do not fit")

    def generate(self) -> CuttingStockInstance:
        """Return one instance; the same configuration always yields the same data."""

        rng = random.Random(self.seed)
        lower, upper = self.piece_length_range
        available_lengths = range(lower, upper + 1)
        if self.number_of_types > len(available_lengths):
            raise ValueError("number_of_types exceeds the available length values")
        lengths = rng.sample(list(available_lengths), self.number_of_types)
        demand_lower, demand_upper = self.demand_range
        demands = [rng.randint(demand_lower, demand_upper) for _ in lengths]
        return CuttingStockInstance(self.stock_length, self.kerf, lengths, demands)

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
