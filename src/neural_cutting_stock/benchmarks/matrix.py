"""Explicit Cartesian matrices for classical benchmark instances."""

from dataclasses import dataclass
from numbers import Real

from .generator import SyntheticInstanceGenerator


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """A named integer distribution configuration used by the generator."""

    name: str
    value_range: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("distribution name must be a non-empty string")
        _validate_range(self.value_range, "distribution value_range")


@dataclass(frozen=True, slots=True)
class BenchmarkMatrix:
    """Cross product of independent instance dimensions.

    Each returned generator represents exactly one cell. Distribution names are
    metadata labels while their ranges define the deterministic uniform sampler.
    """

    seeds: tuple[int, ...]
    number_of_types: tuple[int, ...]
    stock_lengths: tuple[float, ...]
    kerfs: tuple[float, ...]
    length_distributions: tuple[DistributionSpec, ...]
    demand_distributions: tuple[DistributionSpec, ...]

    def __post_init__(self) -> None:
        for name in (
            "seeds",
            "number_of_types",
            "stock_lengths",
            "kerfs",
            "length_distributions",
            "demand_distributions",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds):
            raise ValueError("seeds must contain integers")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in self.number_of_types
        ):
            raise ValueError("number_of_types must contain positive integers")
        if any(
            isinstance(length, bool) or not isinstance(length, Real) or length <= 0
            for length in self.stock_lengths
        ):
            raise ValueError("stock_lengths must contain positive real numbers")
        if any(
            isinstance(kerf, bool) or not isinstance(kerf, Real) or kerf < 0
            for kerf in self.kerfs
        ):
            raise ValueError("kerfs must contain non-negative real numbers")

    @property
    def size(self) -> int:
        """Return the number of generator cells in this matrix."""

        return (
            len(self.seeds)
            * len(self.number_of_types)
            * len(self.stock_lengths)
            * len(self.kerfs)
            * len(self.length_distributions)
            * len(self.demand_distributions)
        )

    def generators(self) -> tuple[SyntheticInstanceGenerator, ...]:
        """Build cells in deterministic dimension order."""

        return tuple(
            SyntheticInstanceGenerator(
                seed=seed,
                stock_length=stock_length,
                kerf=kerf,
                number_of_types=number_of_types,
                piece_length_range=length_distribution.value_range,
                demand_range=demand_distribution.value_range,
                length_distribution=length_distribution.name,
                demand_distribution=demand_distribution.name,
            )
            for seed in self.seeds
            for number_of_types in self.number_of_types
            for stock_length in self.stock_lengths
            for kerf in self.kerfs
            for length_distribution in self.length_distributions
            for demand_distribution in self.demand_distributions
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
