import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from neural_cutting_stock.learning.reproducibility import (
    TRAINING_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_CURVES_SCHEMA_VERSION,
    TrainingCurvePoint,
    TrainingCurves,
    load_checkpoint,
    read_curves_json,
    restore_module_state,
    save_checkpoint,
    set_reproducible_seed,
    write_curves_json,
)


@pytest.mark.parametrize("bad_seed", [-1, True, "42", 1.5, None, 2**32])
def test_set_reproducible_seed_rejects_invalid_seeds(bad_seed) -> None:
    with pytest.raises(ValueError):
        set_reproducible_seed(bad_seed)


def test_set_reproducible_seed_makes_torch_initialization_deterministic() -> None:
    set_reproducible_seed(123)
    first = nn.Linear(4, 3)

    set_reproducible_seed(123)
    second = nn.Linear(4, 3)

    set_reproducible_seed(124)
    third = nn.Linear(4, 3)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)
    assert not torch.equal(first.weight, third.weight)


def test_set_reproducible_seed_seeds_python_and_numpy_generators() -> None:
    set_reproducible_seed(7)
    python_first = [random.random() for _ in range(3)]
    numpy_first = list(np.random.rand(3))

    set_reproducible_seed(7)
    python_second = [random.random() for _ in range(3)]
    numpy_second = list(np.random.rand(3))

    assert python_first == python_second
    assert numpy_first == numpy_second


def test_set_reproducible_seed_returns_traced_environment_metadata() -> None:
    metadata = set_reproducible_seed(42)

    assert metadata["seed"] == 42
    assert metadata["torch_version"] == str(torch.__version__)
    assert isinstance(metadata["cuda_available"], bool)
    assert metadata["python_version"]
    assert metadata["platform"]


def test_training_curve_point_validation() -> None:
    with pytest.raises(ValueError):
        TrainingCurvePoint(-1, {"loss": 0.5})
    with pytest.raises(ValueError):
        TrainingCurvePoint(True, {"loss": 0.5})
    with pytest.raises(ValueError):
        TrainingCurvePoint(0, {})
    with pytest.raises(ValueError):
        TrainingCurvePoint(0, {"loss": float("nan")})
    with pytest.raises(ValueError):
        TrainingCurvePoint(0, {"loss": "low"})
    with pytest.raises(ValueError):
        TrainingCurvePoint(0, {"": 0.5})


def test_training_curves_require_strictly_increasing_steps() -> None:
    curves = TrainingCurves.from_points(
        (
            TrainingCurvePoint(0, {"loss": 1.0}),
            TrainingCurvePoint(2, {"loss": 0.5}),
        )
    )
    extended = curves.extended(5, {"loss": 0.25})

    assert len(extended.points) == 3
    assert extended.points[-1].metrics == {"loss": 0.25}
    with pytest.raises(ValueError):
        curves.extended(1, {"loss": 0.9})
    with pytest.raises(ValueError):
        TrainingCurves(
            (
                TrainingCurvePoint(0, {"loss": 1.0}),
                TrainingCurvePoint(0, {"loss": 1.0}),
            )
        )


def test_training_curves_payload_round_trip() -> None:
    curves = TrainingCurves(
        (
            TrainingCurvePoint(0, {"loss": 1.0}),
            TrainingCurvePoint(1, {"loss": 0.5, "bars_saved": 2.0}),
        )
    )

    restored = TrainingCurves.from_payload(curves.to_payload())

    assert restored == curves
    assert curves.to_payload()["schema_version"] == TRAINING_CURVES_SCHEMA_VERSION


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-mapping",
        {"schema_version": "training-curves-v0", "points": []},
        {"schema_version": TRAINING_CURVES_SCHEMA_VERSION},
        {
            "schema_version": TRAINING_CURVES_SCHEMA_VERSION,
            "points": [{"step": 0}],
        },
        {
            "schema_version": TRAINING_CURVES_SCHEMA_VERSION,
            "points": [{"step": 0, "metrics": "nope"}],
        },
    ],
)
def test_training_curves_from_payload_rejects_malformed_payloads(payload) -> None:
    with pytest.raises(ValueError):
        TrainingCurves.from_payload(payload)


def test_curves_json_round_trip(tmp_path) -> None:
    curves = TrainingCurves((TrainingCurvePoint(0, {"loss": 1.0}),))
    destination = tmp_path / "curves.json"

    write_curves_json(destination, curves)

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted == curves.to_payload()
    assert read_curves_json(destination) == curves


def test_read_curves_json_rejects_missing_and_malformed_files(tmp_path) -> None:
    with pytest.raises(ValueError):
        read_curves_json(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_curves_json(malformed)


def _module() -> nn.Module:
    return nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 2))


def test_checkpoint_round_trip_preserves_state_metadata_and_curves(tmp_path) -> None:
    set_reproducible_seed(42)
    module = _module()
    config = {"algorithm": "imitation", "lr": 0.001}
    curves = TrainingCurves((TrainingCurvePoint(0, {"loss": 1.0}),))
    destination = tmp_path / "checkpoints" / "ckpt.pt"

    summary = save_checkpoint(destination, module=module, seed=42, config=config, curves=curves)
    loaded = load_checkpoint(destination)

    assert destination.is_file()
    assert set(summary) == {
        "schema_version",
        "run_id",
        "seed",
        "config",
        "environment",
        "curves",
    }
    assert loaded["schema_version"] == TRAINING_CHECKPOINT_SCHEMA_VERSION
    assert loaded["seed"] == 42
    assert loaded["config"] == config
    assert TrainingCurves.from_payload(loaded["curves"]) == curves
    assert loaded["environment"]["torch_version"] == str(torch.__version__)
    saved_state = loaded["model_state_dict"]
    expected_state = module.state_dict()
    assert set(saved_state) == set(expected_state)
    for key, value in expected_state.items():
        assert saved_state[key].dtype == value.dtype
        assert torch.equal(saved_state[key], value)


def test_checkpoint_run_id_is_deterministic_and_config_sensitive(tmp_path) -> None:
    set_reproducible_seed(1)
    first = save_checkpoint(
        tmp_path / "a.pt",
        module=_module(),
        seed=1,
        config={"lr": 0.1},
        curves=TrainingCurves(),
    )
    second = save_checkpoint(
        tmp_path / "b.pt",
        module=_module(),
        seed=1,
        config={"lr": 0.1},
        curves=TrainingCurves(),
    )
    other_config = save_checkpoint(
        tmp_path / "c.pt",
        module=_module(),
        seed=1,
        config={"lr": 0.2},
        curves=TrainingCurves(),
    )
    explicit = save_checkpoint(
        tmp_path / "d.pt",
        module=_module(),
        seed=1,
        config={"lr": 0.1},
        curves=TrainingCurves(),
        run_id="run-abc",
    )

    assert first["run_id"] == second["run_id"]
    assert first["run_id"] != other_config["run_id"]
    assert explicit["run_id"] == "run-abc"


def test_checkpoint_restores_identical_weights_after_reseeding(tmp_path) -> None:
    set_reproducible_seed(99)
    trained = _module()
    destination = tmp_path / "ckpt.pt"
    save_checkpoint(
        destination,
        module=trained,
        seed=99,
        config={},
        curves=TrainingCurves(),
    )

    set_reproducible_seed(99)
    replayed = _module()

    reference_state = trained.state_dict()
    replayed_state = replayed.state_dict()
    for key, value in reference_state.items():
        assert torch.equal(replayed_state[key], value)


def test_restore_module_state_round_trip_and_incompatibility(tmp_path) -> None:
    set_reproducible_seed(11)
    source = _module()
    destination_file = tmp_path / "ckpt.pt"
    save_checkpoint(
        destination_file, module=source, seed=11, config={}, curves=TrainingCurves()
    )
    payload = load_checkpoint(destination_file)

    target = _module()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(1.0)
    restore_module_state(target, payload)

    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        assert torch.equal(target_state[key], value)

    incompatible = nn.Linear(3, 2)
    with pytest.raises(ValueError):
        restore_module_state(incompatible, payload)


def test_load_checkpoint_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_rejects_unsupported_schema(tmp_path) -> None:
    set_reproducible_seed(5)
    destination = tmp_path / "legacy.pt"
    save_checkpoint(
        destination, module=_module(), seed=5, config={}, curves=TrainingCurves()
    )
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload["schema_version"] = "neural-training-checkpoint-v0"
    legacy = tmp_path / "legacy-v0.pt"
    torch.save(payload, legacy)

    with pytest.raises(ValueError, match="schema_version"):
        load_checkpoint(legacy)


def test_load_checkpoint_rejects_non_mapping_payload(tmp_path) -> None:
    foreign = tmp_path / "foreign.pt"
    torch.save([1, 2, 3], foreign)

    with pytest.raises(ValueError, match="mapping"):
        load_checkpoint(foreign)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"config": "not-a-mapping"},
        {"curves": "not-curves"},
    ],
)
def test_save_checkpoint_validates_its_inputs(tmp_path, kwargs) -> None:
    call = {
        "module": _module(),
        "seed": 3,
        "config": {},
        "curves": TrainingCurves(),
    }
    call.update(kwargs)

    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "invalid.pt", **call)


def test_save_checkpoint_rejects_non_serializable_config(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_checkpoint(
            tmp_path / "invalid.pt",
            module=_module(),
            seed=3,
            config={"object": object()},
            curves=TrainingCurves(),
        )


def test_curves_are_shared_between_checkpoint_and_standalone_artifact(tmp_path) -> None:
    curves = TrainingCurves(
        (
            TrainingCurvePoint(0, {"loss": 2.0}),
            TrainingCurvePoint(1, {"loss": 1.0}),
        )
    )
    set_reproducible_seed(8)
    checkpoint_path = tmp_path / "ckpt.pt"
    curves_path = tmp_path / "curves.json"

    save_checkpoint(checkpoint_path, module=_module(), seed=8, config={}, curves=curves)
    write_curves_json(curves_path, curves)

    from_checkpoint = TrainingCurves.from_payload(load_checkpoint(checkpoint_path)["curves"])
    assert from_checkpoint == read_curves_json(curves_path)
