"""Deterministic learning examples built only from replay-validated trajectories."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .partitions import DatasetPartition
from .trajectory import ColumnGenerationTrajectory, replay_trajectory

DATASET_SCHEMA_VERSION = "trajectory-dataset-v1"


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """One candidate decision observed at one RMP iteration."""

    trajectory_id: str
    instance_id: str
    partition: DatasetPartition
    iteration_index: int
    dual_values: tuple[float, ...]
    candidate_pattern: tuple[int, ...]
    candidate_reduced_cost: float
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "instance_id": self.instance_id,
            "partition": self.partition.value,
            "iteration_index": self.iteration_index,
            "dual_values": list(self.dual_values),
            "candidate_pattern": list(self.candidate_pattern),
            "candidate_reduced_cost": self.candidate_reduced_cost,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryDataset:
    """Versioned collection of examples and its validated source identities."""

    examples: tuple[DatasetExample, ...]
    trajectory_ids: tuple[str, ...]
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if tuple(sorted(self.trajectory_ids)) != self.trajectory_ids:
            raise ValueError("trajectory_ids must be sorted")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("trajectory_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready representation."""

        return {
            "schema_version": self.schema_version,
            "trajectory_ids": list(self.trajectory_ids),
            "examples": [example.to_dict() for example in self.examples],
        }


def build_dataset(
    trajectories: tuple[ColumnGenerationTrajectory, ...],
    partitions: Mapping[str, DatasetPartition | str],
) -> TrajectoryDataset:
    """Replay every source and build examples only when every source is valid.

    Candidate fields are optional in the trajectory schema. An iteration without a recorded
    candidate pool therefore contributes no examples, rather than inventing candidates or labels.
    """

    ordered = tuple(sorted(trajectories, key=lambda item: item.metadata.trajectory_id))
    trajectory_ids = tuple(item.metadata.trajectory_id for item in ordered)
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory_ids must be unique")
    if set(partitions) != set(trajectory_ids):
        raise ValueError("partitions must contain exactly one entry per trajectory")

    examples: list[DatasetExample] = []
    instance_partitions: dict[str, DatasetPartition] = {}
    for trajectory in ordered:
        validation = replay_trajectory(trajectory)
        if not validation.valid:
            raise ValueError(
                f"trajectory {trajectory.metadata.trajectory_id!r} is invalid: "
                + "; ".join(validation.errors)
            )
        partition = DatasetPartition(partitions[trajectory.metadata.trajectory_id])
        previous_partition = instance_partitions.setdefault(
            trajectory.metadata.instance_id, partition
        )
        if previous_partition is not partition:
            raise ValueError(
                f"instance {trajectory.metadata.instance_id!r} appears in multiple partitions"
            )
        for iteration in trajectory.iterations:
            if iteration.candidate_patterns is None:
                continue
            assert iteration.candidate_reduced_costs is not None
            selected = set(iteration.selected_patterns or ())
            for pattern, reduced_cost in zip(
                iteration.candidate_patterns, iteration.candidate_reduced_costs, strict=True
            ):
                if iteration.dual_values is None:
                    raise ValueError(
                        f"trajectory {trajectory.metadata.trajectory_id!r} has candidates "
                        f"without dual values at iteration {iteration.iteration_index}"
                    )
                examples.append(
                    DatasetExample(
                        trajectory.metadata.trajectory_id,
                        trajectory.metadata.instance_id,
                        partition,
                        iteration.iteration_index,
                        iteration.dual_values,
                        pattern,
                        reduced_cost,
                        pattern in selected,
                    )
                )
    return TrajectoryDataset(tuple(examples), trajectory_ids)
