import json
from dataclasses import replace

from neural_cutting_stock.benchmarks import DatasetExample, DatasetPartition, TrajectoryDataset
from neural_cutting_stock.learning import (
    RANKING_CUTOFFS,
    TRAINING_ARTIFACT_SCHEMA_VERSION,
    LinearColumnScoringModel,
    PatternCandidate,
    PricingState,
    evaluate_model,
    load_training_artifact,
    pricing_features,
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


def test_loading_artifact_reconstructs_the_compatible_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "neural_cutting_stock.learning.training.load_phase3_dataset", lambda _: _dataset()
    )

    output = tmp_path / "model.json"
    artifact = write_training_artifact("manifest.json", output, 42, {})
    loaded = load_training_artifact(output)

    assert loaded.weights == tuple(artifact["model"]["weights"])
    assert loaded.bias == artifact["model"]["bias"]
    assert loaded.feature_width == artifact["model"]["feature_width"]


def test_loading_artifact_rejects_incompatible_schema(tmp_path) -> None:
    artifact = {
        "schema_version": "linear-training-artifact-v0",
        "model_schema_version": "linear-scorer-v1",
        "feature_schema_version": "pricing-features-v1",
        "dataset_schema_version": "trajectory-dataset-v1",
        "model": {"feature_width": 1, "weights": [1.0], "bias": 0.0},
    }
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_training_artifact(path)


def test_loading_artifact_rejects_unavailable_path(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="invalid training artifact"):
        load_training_artifact(tmp_path / "missing-model.json")


def test_loading_artifact_rejects_inconsistent_model_shape(tmp_path) -> None:
    artifact = {
        "schema_version": TRAINING_ARTIFACT_SCHEMA_VERSION,
        "model_schema_version": "linear-scorer-v1",
        "feature_schema_version": "pricing-features-v1",
        "dataset_schema_version": "trajectory-dataset-v1",
        "model": {"feature_width": 2, "weights": [1.0], "bias": 0.0},
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="feature_width must match"):
        load_training_artifact(path)


def test_training_rejects_empty_training_partition(monkeypatch) -> None:
    dataset = TrajectoryDataset((), ())
    monkeypatch.setattr(
        "neural_cutting_stock.learning.training.load_phase3_dataset", lambda _: dataset
    )

    import pytest

    with pytest.raises(ValueError, match="contains no candidate examples"):
        train_artifact("manifest.json", 1, {})


def test_evaluation_reports_fixed_out_of_sample_ranking_metrics() -> None:
    dataset = _dataset()
    example = dataset.examples[0]
    state = PricingState(
        instance_id=example.instance_id,
        iteration_index=example.iteration_index,
        stock_length=example.stock_length,
        kerf=example.kerf,
        piece_lengths=example.piece_lengths,
        demands=example.demands,
        dual_values=example.dual_values,
        current_patterns=example.current_patterns,
        rmp_objective_value=example.rmp_objective_value,
    )
    candidates = tuple(
        PatternCandidate(item.candidate_pattern, item.candidate_reduced_cost)
        for item in dataset.examples
    )
    model = LinearColumnScoringModel.fit(
        tuple(pricing_features(state, candidate) for candidate in candidates),
        (0.0, 1.0),
    )

    # The fitted model is intentionally evaluated on a non-training partition.
    validation = TrajectoryDataset(
        tuple(
            replace(example, partition=DatasetPartition.VALIDATION)
            for example in dataset.examples
        ),
        dataset.trajectory_ids,
    )
    result = evaluate_model(validation, model, DatasetPartition.VALIDATION)

    assert result["schema_version"] == "ranking-evaluation-v1"
    assert result["evaluated_group_count"] == 1
    assert result["positive_example_count"] == 1
    assert tuple(RANKING_CUTOFFS) == (1, 3, 5)
    assert result["metrics"]["learned"] == {
        "hit_rate_at_1": 1.0,
        "hit_rate_at_3": 1.0,
        "hit_rate_at_5": 1.0,
        "mrr": 1.0,
        "ndcg_at_5": 1.0,
    }
    assert result["metrics"]["exact_reduced_cost"]["mrr"] == 1.0
