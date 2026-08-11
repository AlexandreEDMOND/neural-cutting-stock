"""Leakage-safe, deterministic partitions for trajectory collection."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .generator import SyntheticInstanceGenerator

PARTITION_SCHEMA_VERSION = "trajectory-partitions-v1"


class DatasetPartition(StrEnum):
    """Allowed partitions for future learning data."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class PartitionAssignment:
    """The partition assigned to one generator without collecting a trajectory."""

    seed: int
    family_id: str
    partition: DatasetPartition


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    """Explicit seed and family split fixed before trajectory collection."""

    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    train_families: tuple[str, ...]
    validation_families: tuple[str, ...]
    test_families: tuple[str, ...]
    schema_version: str = PARTITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARTITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        seed_sets = self._seed_sets()
        family_sets = self._family_sets()
        _validate_disjoint(seed_sets, "seeds", _validate_seed)
        _validate_disjoint(family_sets, "families", _validate_family)
        if not all(seed_sets.values()) or not all(family_sets.values()):
            raise ValueError("each partition must contain at least one seed and family")

    def assign(self, generator: SyntheticInstanceGenerator) -> PartitionAssignment:
        """Assign a generator, rejecting unknown or cross-partition combinations."""

        seed_partition = self._lookup(generator.seed, self._seed_sets(), "seed")
        family_partition = self._lookup(generator.family_id, self._family_sets(), "family")
        if seed_partition is not family_partition:
            raise ValueError(
                "generator seed and family belong to different partitions: "
                f"seed={seed_partition.value}, family={family_partition.value}"
            )
        return PartitionAssignment(generator.seed, generator.family_id, seed_partition)

    def assignments(
        self, generators: tuple[SyntheticInstanceGenerator, ...]
    ) -> tuple[PartitionAssignment, ...]:
        """Return stable assignments in the supplied generator order."""

        return tuple(self.assign(generator) for generator in generators)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic manifest-ready representation."""

        return {
            "schema_version": self.schema_version,
            "train_seeds": list(self.train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "test_seeds": list(self.test_seeds),
            "train_families": list(self.train_families),
            "validation_families": list(self.validation_families),
            "test_families": list(self.test_families),
        }

    def _seed_sets(self) -> dict[DatasetPartition, tuple[int, ...]]:
        return {
            DatasetPartition.TRAIN: self.train_seeds,
            DatasetPartition.VALIDATION: self.validation_seeds,
            DatasetPartition.TEST: self.test_seeds,
        }

    def _family_sets(self) -> dict[DatasetPartition, tuple[str, ...]]:
        return {
            DatasetPartition.TRAIN: self.train_families,
            DatasetPartition.VALIDATION: self.validation_families,
            DatasetPartition.TEST: self.test_families,
        }

    @staticmethod
    def _lookup(value, groups, label: str) -> DatasetPartition:
        matches = [partition for partition, values in groups.items() if value in values]
        if len(matches) != 1:
            raise ValueError(f"unknown {label} for partition plan: {value!r}")
        return matches[0]


def _validate_disjoint(groups, label: str, validator) -> None:
    seen = set()
    for values in groups.values():
        if not isinstance(values, tuple) or not values:
            raise ValueError(f"{label} must be non-empty tuples")
        for value in values:
            validator(value, label)
            if value in seen:
                raise ValueError(f"{label} must be disjoint across partitions")
            seen.add(value)


def _validate_seed(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must contain integers")


def _validate_family(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must contain non-empty strings")
