"""Manifest and statistics helpers for small, replayable trajectory corpora."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .partitions import DatasetPartition
from .trajectory import ColumnGenerationTrajectory

CORPUS_SCHEMA_VERSION = "phase-3-corpus-v1"


def trajectory_sha256(trajectory: ColumnGenerationTrajectory) -> str:
    """Hash the canonical persisted representation of a trajectory."""

    payload = json.dumps(trajectory.to_dict(), ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def corpus_statistics(
    trajectories: tuple[ColumnGenerationTrajectory, ...],
    partitions: dict[str, DatasetPartition | str],
) -> dict[str, Any]:
    """Return counts computed only from replayable trajectories and assignments."""

    if set(partitions) != {item.metadata.trajectory_id for item in trajectories}:
        raise ValueError("partitions must contain exactly one entry per trajectory")
    ordered = tuple(sorted(trajectories, key=lambda item: item.metadata.trajectory_id))
    status_counts = Counter(item.status.value for item in ordered)
    partition_counts = Counter(
        DatasetPartition(partitions[item.metadata.trajectory_id]).value for item in ordered
    )
    return {
        "trajectory_count": len(ordered),
        "instance_count": len({item.metadata.instance_id for item in ordered}),
        "status_counts": dict(sorted(status_counts.items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "iteration_count": sum(len(item.iterations) for item in ordered),
        "columns_added": sum(
            iteration.columns_added or 0
            for item in ordered
            for iteration in item.iterations
        ),
        "selected_pattern_count": sum(
            len(iteration.selected_patterns or ())
            for item in ordered
            for iteration in item.iterations
        ),
    }


def read_corpus_manifest(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate a versioned corpus manifest."""

    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported corpus manifest")
    if not isinstance(manifest.get("trajectories"), list) or not manifest["trajectories"]:
        raise ValueError("corpus manifest must contain trajectories")
    if not isinstance(manifest.get("statistics"), dict):
        raise ValueError("corpus manifest must contain statistics")
    return manifest
