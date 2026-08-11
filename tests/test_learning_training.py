import json

from neural_cutting_stock.benchmarks import DatasetExample, DatasetPartition, TrajectoryDataset
from neural_cutting_stock.learning import (
    TRAINING_ARTIFACT_SCHEMA_VERSION,
    train_artifact,
    write_training_artifact,
)


def _dataset() -> TrajectoryDataset:
    common = {
        "trajectory_id": "trajectory-1",
        "instance_id": "instance-1",
        "partition": DatasetPartition.TRAIN,
        "iteration_index": 1,
        "dual_values": (0.5, 0.25),
        "stock_length": 100.0,
        "kerf": 0.0,
        "piece_lengths": (20.0, 40.0),
        "demands": (3, 2),
    }
    return TrajectoryDataset(
        (
            DatasetExample(
                candidate_pattern=(1, 0), candidate_reduced_cost=0.2, selected=False, **common
            ),
            DatasetExample(
                candidate_pattern=(0, 1), candidate_reduced_cost=-0.1, selected=True, **common
            ),
        ),
        ("trajectory-1",),
    )


def test_training_persists_seed_config_and_model_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "neural_cutting_stock.learning.training.load_phase3_dataset", lambda _: _dataset()
    )

    output = tmp_path / "model.json"
    artifact = write_training_artifact("manifest.json", output, 42, {"ridge": 0.0})
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert persisted["schema_version"] == TRAINING_ARTIFACT_SCHEMA_VERSION
    assert persisted["seed"] == 42
    assert persisted["config"] == {"ridge": 0.0}
    assert persisted["metadata"]["example_count"] == 2
    assert persisted["model"]["feature_width"] > 0


def test_training_rejects_empty_training_partition(monkeypatch) -> None:
    dataset = TrajectoryDataset((), ())
    monkeypatch.setattr(
        "neural_cutting_stock.learning.training.load_phase3_dataset", lambda _: dataset
    )

    import pytest

    with pytest.raises(ValueError, match="contains no candidate examples"):
        train_artifact("manifest.json", 1, {})
