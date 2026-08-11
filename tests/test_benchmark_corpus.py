import json
from pathlib import Path

from test_benchmark_dataset import _trajectory

from neural_cutting_stock.benchmarks import (
    CORPUS_SCHEMA_VERSION,
    DatasetPartition,
    TrajectoryStatus,
    corpus_statistics,
    read_corpus_manifest,
    read_trajectory,
    trajectory_sha256,
)


def test_corpus_statistics_are_derived_from_trajectories() -> None:
    trajectory = _trajectory()

    statistics = corpus_statistics(
        (trajectory,), {trajectory.metadata.trajectory_id: DatasetPartition.TEST}
    )

    assert statistics == {
        "trajectory_count": 1,
        "instance_count": 1,
        "status_counts": {TrajectoryStatus.CONVERGED.value: 1},
        "partition_counts": {DatasetPartition.TEST.value: 1},
        "iteration_count": 1,
        "columns_added": 0,
        "selected_pattern_count": 1,
    }


def test_corpus_manifest_reader_validates_version_and_shape(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "trajectories": [{"path": "trajectory.json"}],
                "statistics": {},
            }
        ),
        encoding="utf-8",
    )

    assert read_corpus_manifest(path)["schema_version"] == CORPUS_SCHEMA_VERSION


def test_trajectory_hash_is_stable_for_same_serialized_content() -> None:
    trajectory = _trajectory()

    assert trajectory_sha256(trajectory) == trajectory_sha256(trajectory)


def test_versioned_phase3_corpus_matches_manifest_and_replays() -> None:
    root = Path("data/phase-3-corpus")
    manifest = read_corpus_manifest(root / "manifest.json")
    trajectories = tuple(
        read_trajectory(root / entry["path"]) for entry in manifest["trajectories"]
    )
    partitions = {
        trajectory.metadata.trajectory_id: entry["partition"]
        for trajectory, entry in zip(trajectories, manifest["trajectories"], strict=True)
    }

    assert [trajectory_sha256(item) for item in trajectories] == [
        entry["sha256"] for entry in manifest["trajectories"]
    ]
    assert corpus_statistics(trajectories, partitions) == manifest["statistics"]
