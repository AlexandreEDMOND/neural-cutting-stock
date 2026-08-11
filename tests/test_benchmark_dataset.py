import pytest
from test_benchmark_schema import _trajectory_metadata

from neural_cutting_stock.benchmarks import (
    DATASET_SCHEMA_VERSION,
    ColumnGenerationTrajectory,
    DatasetPartition,
    TrajectoryIteration,
    TrajectoryStatus,
    build_dataset,
    load_phase3_dataset,
)


def _trajectory(trajectory_id: str = "trajectory-1", instance_id: str = "instance-1"):
    metadata = _trajectory_metadata(trajectory_id=trajectory_id, instance_id=instance_id)
    return ColumnGenerationTrajectory(
        metadata,
        (
            TrajectoryIteration(
                1,
                "optimal",
                dual_values=(0.5, 0.25),
                candidate_patterns=((2, 0), (1, 1)),
                candidate_reduced_costs=(0.2, -0.1),
                selected_patterns=((1, 1),),
            ),
        ),
        TrajectoryStatus.CONVERGED,
        "no_improving_column",
    )


def test_dataset_replays_sources_and_preserves_partitioned_examples(monkeypatch) -> None:
    trajectory = _trajectory()
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.dataset.replay_trajectory",
        lambda value: type("Validation", (), {"valid": True, "errors": ()})(),
    )

    dataset = build_dataset((trajectory,), {"trajectory-1": DatasetPartition.TRAIN})

    assert dataset.schema_version == DATASET_SCHEMA_VERSION
    assert dataset.trajectory_ids == ("trajectory-1",)
    assert [example.selected for example in dataset.examples] == [False, True]
    assert dataset.examples[1].partition is DatasetPartition.TRAIN
    assert dataset.to_dict()["examples"][1]["candidate_pattern"] == [1, 1]


def test_dataset_rejects_instance_leakage_across_partitions(monkeypatch) -> None:
    first = _trajectory("trajectory-1")
    second = _trajectory("trajectory-2")
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.dataset.replay_trajectory",
        lambda value: type("Validation", (), {"valid": True, "errors": ()})(),
    )

    with pytest.raises(ValueError, match="appears in multiple partitions"):
        build_dataset(
            (first, second),
            {"trajectory-1": DatasetPartition.TRAIN, "trajectory-2": DatasetPartition.TEST},
        )


def test_dataset_is_invariant_to_trajectory_input_order(monkeypatch) -> None:
    first = _trajectory("trajectory-1", "instance-1")
    second = _trajectory("trajectory-2", "instance-2")
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.dataset.replay_trajectory",
        lambda value: type("Validation", (), {"valid": True, "errors": ()})(),
    )
    partitions = {"trajectory-1": DatasetPartition.TRAIN, "trajectory-2": DatasetPartition.TEST}

    ordered = build_dataset((first, second), partitions)
    reversed_order = build_dataset((second, first), partitions)

    assert reversed_order == ordered


def test_dataset_rejects_invalid_trajectory_without_partial_output(monkeypatch) -> None:
    trajectory = _trajectory()
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.dataset.replay_trajectory",
        lambda value: type("Validation", (), {"valid": False, "errors": ("dual differs",)})(),
    )

    with pytest.raises(ValueError, match="is invalid: dual differs"):
        build_dataset((trajectory,), {"trajectory-1": "validation"})


def test_dataset_rejects_missing_partition_or_missing_duals(monkeypatch) -> None:
    trajectory = _trajectory()
    monkeypatch.setattr(
        "neural_cutting_stock.benchmarks.dataset.replay_trajectory",
        lambda value: type("Validation", (), {"valid": True, "errors": ()})(),
    )
    with pytest.raises(ValueError, match="exactly one entry"):
        build_dataset((trajectory,), {})

    no_duals = _trajectory()
    no_duals = type(no_duals)(
        no_duals.metadata,
        (
            TrajectoryIteration(
                1,
                "optimal",
                candidate_patterns=((1, 1),),
                candidate_reduced_costs=(-1.0,),
            ),
        ),
        no_duals.status,
        no_duals.termination_reason,
    )
    with pytest.raises(ValueError, match="without dual values"):
        build_dataset((no_duals,), {"trajectory-1": "test"})


def test_phase3_loader_reconstructs_examples_and_partitions() -> None:
    dataset = load_phase3_dataset("data/phase-3-corpus/manifest.json")

    assert dataset.trajectory_ids == tuple(sorted(dataset.trajectory_ids))
    assert dataset.examples == ()


def test_phase3_loader_is_invariant_to_manifest_trajectory_order(tmp_path) -> None:
    import json
    from pathlib import Path

    source = Path("data/phase-3-corpus/manifest.json")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["trajectories"] = list(reversed(manifest["trajectories"]))
    copied = tmp_path / "manifest.json"
    for entry in manifest["trajectories"]:
        source_trajectory = source.parent / entry["path"]
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_trajectory.read_bytes())
    copied.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_phase3_dataset(copied) == load_phase3_dataset(source)


def test_phase3_loader_rejects_hash_mismatch(tmp_path) -> None:
    import json
    from pathlib import Path

    source = Path("data/phase-3-corpus/manifest.json")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    entry = manifest["trajectories"][0]
    entry["sha256"] = "0" * 64
    target_manifest = tmp_path / "manifest.json"
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / entry["path"]
    target.parent.mkdir()
    target.write_bytes((source.parent / entry["path"]).read_bytes())

    with pytest.raises(ValueError, match="hash differs"):
        load_phase3_dataset(target_manifest)
