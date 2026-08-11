"""Deterministic learning examples built only from replay-validated trajectories."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import corpus_statistics, read_corpus_manifest, trajectory_sha256
from .partitions import DatasetPartition, PartitionPlan
from .trajectory import ColumnGenerationTrajectory, read_trajectory, replay_trajectory

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


def load_phase3_dataset(manifest_path: str | Path) -> TrajectoryDataset:
    """Load and validate the reproducible Phase 3 examples from a corpus manifest.

    The manifest is the source of partition assignments and file identities. Every trajectory is
    hash-checked before :func:`build_dataset` replays it through the exact classical solver.
    """

    manifest_path = Path(manifest_path)
    manifest = read_corpus_manifest(manifest_path)
    partition_plan_value = manifest.get("partition_plan")
    if not isinstance(partition_plan_value, dict):
        raise ValueError("corpus manifest must contain a partition_plan")
    try:
        plan = PartitionPlan(
            **{
                name: tuple(partition_plan_value[name])
                for name in (
                    "train_seeds",
                    "validation_seeds",
                    "test_seeds",
                    "train_families",
                    "validation_families",
                    "test_families",
                )
            },
            schema_version=partition_plan_value.get("schema_version"),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid partition_plan") from error

    root = manifest_path.parent.resolve()
    trajectories: list[ColumnGenerationTrajectory] = []
    partitions: dict[str, DatasetPartition] = {}
    for entry in manifest["trajectories"]:
        if not isinstance(entry, dict):
            raise ValueError("trajectory manifest entries must be objects")
        try:
            trajectory_id = entry["trajectory_id"]
            relative_path = Path(entry["path"])
            partition = DatasetPartition(entry["partition"])
            seed = entry["seed"]
            family_id = entry["family_id"]
            expected_hash = entry["sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid trajectory manifest entry") from error
        if not isinstance(trajectory_id, str) or not isinstance(expected_hash, str):
            raise ValueError("trajectory manifest identity fields must be strings")
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise ValueError(f"trajectory path escapes corpus directory: {relative_path}")
        trajectory = read_trajectory(path)
        if trajectory.metadata.trajectory_id != trajectory_id:
            raise ValueError(f"trajectory identity differs for {relative_path}")
        if trajectory.metadata.seed != seed:
            raise ValueError(f"trajectory seed differs for {trajectory_id!r}")
        if trajectory_sha256(trajectory) != expected_hash:
            raise ValueError(f"trajectory hash differs for {trajectory_id!r}")
        if not _partition_plan_contains(plan, partition, seed, family_id):
            raise ValueError(f"trajectory {trajectory_id!r} does not match partition plan")
        if trajectory_id in partitions:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id!r}")
        trajectories.append(trajectory)
        partitions[trajectory_id] = partition

    dataset = build_dataset(tuple(trajectories), partitions)
    actual_statistics = corpus_statistics(tuple(trajectories), partitions)
    if actual_statistics != manifest["statistics"]:
        raise ValueError("corpus statistics differ from manifest")
    return dataset


def _partition_plan_contains(
    plan: PartitionPlan, partition: DatasetPartition, seed: int, family_id: str
) -> bool:
    seed_values = {
        DatasetPartition.TRAIN: plan.train_seeds,
        DatasetPartition.VALIDATION: plan.validation_seeds,
        DatasetPartition.TEST: plan.test_seeds,
    }[partition]
    family_values = {
        DatasetPartition.TRAIN: plan.train_families,
        DatasetPartition.VALIDATION: plan.validation_families,
        DatasetPartition.TEST: plan.test_families,
    }[partition]
    return seed in seed_values and family_id in family_values
