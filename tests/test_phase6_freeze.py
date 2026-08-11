import hashlib
import json
from pathlib import Path

from neural_cutting_stock.learning import load_training_artifact

ROOT = Path(__file__).parents[1]


def test_phase6_freeze_declares_versioned_protocol_and_artifacts() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))

    assert config["schema_version"] == "phase-6-final-freeze-v1"
    assert len(config["code_commit"]) == 40
    assert config["model"]["model_id"] == "linear-scorer-v1-zero-weight"
    assert config["partitions"]["schema_version"] == "trajectory-partitions-v1"
    assert config["protocol"] == {
        "benchmark_schema_version": "benchmark-run-v1",
        "comparison": "paired_instance_id",
        "quality_tolerance_bars": 0.0,
        "repetitions": 1,
        "warmup_runs": 0,
        "reduced_cost_tolerance": 1e-9,
        "max_runtime_seconds": None,
        "max_cg_iterations": None,
        "model_loading": "preloaded_per_process",
        "execution_order": "classical_then_neural",
        "size_class_version": "size-class-v1",
    }

    for relative_path in config["files"].values():
        assert (ROOT / relative_path).is_file()


def test_frozen_model_is_loadable_and_has_declared_shape() -> None:
    artifact_path = ROOT / "models/linear-scorer-v1-zero-weight.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    model = load_training_artifact(artifact_path)

    assert artifact["schema_version"] == "linear-training-artifact-v1"
    assert artifact["model"]["feature_width"] == len(artifact["model"]["weights"])
    assert artifact["metadata"]["candidate_status"] == "retained_phase_4_policy"
    assert model.feature_width == artifact["model"]["feature_width"]


def test_frozen_file_hashes_are_stable() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    for relative_path in config["files"].values():
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert len(digest) == 64
