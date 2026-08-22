"""Deep REINFORCE training of the quality policy in the verified env."""

import math

import pytest
import torch

from neural_cutting_stock.learning import (
    ALGORITHM_IDENTIFIER,
    DEFAULT_BASELINE_MOMENTUM,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_STEPS,
    QUALITY_RL_POLICY_SCHEMA_VERSION,
    TRAINING_JOURNAL_SCHEMA_VERSION,
    QualityPolicyNetwork,
    train_quality_rl_policy,
    training_journal_payload,
)
from neural_cutting_stock.learning import rl_policy as rl_policy_module
from neural_cutting_stock.learning.imitation import (
    QualityAgentInput,
    imitation_candidate_features_batch,
)
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
)

EPOCHS = 20
TRAINING_CONFIG = {
    "epochs": EPOCHS,
    "learning_rate": 3e-3,
    "hidden_width": 16,
    "max_steps": 2,
}

TOY_INSTANCES = {
    "toy-a": CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 4)),
    "toy-b": CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 5)),
    "toy-multi": MultiFormatCuttingStockInstance((50.0, 120.0), 0.0, [30.0, 40.0], [5, 4]),
}


@pytest.fixture(scope="module")
def policy():
    return train_quality_rl_policy(TOY_INSTANCES, seed=7, **TRAINING_CONFIG)


def toy_observation(instance_id: str):
    instance = TOY_INSTANCES[instance_id]
    cg_result = ColumnGeneration(instance, instance_id=instance_id).solve()
    assert cg_result.status == "converged"
    return instance, cg_result


def observation_of(instance_id: str, instance, cg_result) -> QualityAgentInput:
    return QualityAgentInput(
        instance_id=instance_id,
        stock_length=instance.stock_length,
        kerf=instance.kerf,
        piece_lengths=instance.piece_lengths,
        demands=instance.demands,
        column_pool=cg_result.patterns,
        solution_patterns=cg_result.patterns,
        solution_column_values=cg_result.integer_master_result.column_values,
    )


def test_schema_versions_and_defaults_are_stable() -> None:
    assert QUALITY_RL_POLICY_SCHEMA_VERSION == "quality-rl-policy-v1"
    assert TRAINING_JOURNAL_SCHEMA_VERSION == "phase-9-training-journal-v1"
    assert ALGORITHM_IDENTIFIER == "reinforce-poisson-completion-v1"
    assert (DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_HIDDEN_WIDTH) == (300, 3e-3, 64)
    assert (DEFAULT_MAX_STEPS, DEFAULT_BASELINE_MOMENTUM) == (4, 0.8)


def test_trained_policy_carries_complete_provenance(policy) -> None:
    assert policy.schema_version == QUALITY_RL_POLICY_SCHEMA_VERSION
    assert policy.feature_width == 28
    assert policy.seed == 7
    assert policy.config["algorithm"] == ALGORITHM_IDENTIFIER
    assert policy.config["epochs"] == TRAINING_CONFIG["epochs"]
    assert policy.config["hidden_width"] == TRAINING_CONFIG["hidden_width"]
    assert policy.config["max_steps"] == TRAINING_CONFIG["max_steps"]
    assert policy.environment["seed"] == 7
    assert policy.environment["torch_version"]


def test_curves_cover_every_epoch_with_finite_metrics(policy) -> None:
    points = policy.curves.points

    assert [point.step for point in points] == list(range(TRAINING_CONFIG["epochs"]))
    expected_metrics = {
        "policy_loss",
        "mean_episode_return",
        "bars_saved_total",
        "accepted_steps",
        "invalid_steps",
    }
    for point in points:
        assert set(point.metrics) == expected_metrics
        assert all(math.isfinite(value) for value in point.metrics.values())


def test_episode_records_cover_the_full_run_and_never_hit_invalid_plans(policy) -> None:
    records = policy.episodes

    assert len(records) == TRAINING_CONFIG["epochs"] * len(TOY_INSTANCES)
    assert [record.episode_index for record in records] == list(range(len(records)))
    assert {record.instance_id for record in records} == set(TOY_INSTANCES)
    assert policy.trained_instance_ids == tuple(sorted(TOY_INSTANCES))
    for record in records:
        assert record.steps_taken == TRAINING_CONFIG["max_steps"]
        assert record.invalid_steps == 0
        assert record.final_bars == record.initial_bars - record.bars_saved
        assert math.isfinite(record.return_value)
    totals = policy.totals
    assert totals["episode_count"] == len(records)
    assert totals["step_count"] == sum(record.steps_taken for record in records)
    assert totals["bars_saved_total"] == sum(record.bars_saved for record in records)
    assert totals["accepted_step_count"] == sum(record.accepted_steps for record in records)
    assert totals["invalid_step_count"] == 0


def test_training_is_reproducible_for_a_fixed_seed(policy) -> None:
    replayed = train_quality_rl_policy(TOY_INSTANCES, seed=7, **TRAINING_CONFIG)

    for key, value in policy.module.state_dict().items():
        assert torch.equal(replayed.module.state_dict()[key], value)
    assert replayed.episodes == policy.episodes
    assert replayed.curves.to_payload() == policy.curves.to_payload()


def test_a_different_seed_produces_different_weights() -> None:
    first = train_quality_rl_policy(TOY_INSTANCES, seed=7, **TRAINING_CONFIG)
    second = train_quality_rl_policy(TOY_INSTANCES, seed=8, **TRAINING_CONFIG)

    keys = first.module.state_dict()
    assert any(
        not torch.equal(first.module.state_dict()[key], second.module.state_dict()[key])
        for key in keys
    )


def test_network_rates_stay_strictly_positive_under_extreme_inputs() -> None:
    network = QualityPolicyNetwork(28, 8)

    for scale in (-1e12, -1.0, 0.0, 1.0, 1e12):
        features = torch.full((5, 28), scale, dtype=torch.float32)

        assert bool((network(features) > 0).all())


def test_checkpoint_round_trip_preserves_the_emitted_rates(policy, tmp_path) -> None:
    from neural_cutting_stock.learning import (
        load_checkpoint,
        restore_module_state,
        save_checkpoint,
    )

    destination = tmp_path / "checkpoints" / "quality-policy.pt"
    save_checkpoint(
        destination,
        module=policy.module,
        seed=policy.seed,
        config=dict(policy.config),
        curves=policy.curves,
    )
    restored = QualityPolicyNetwork(policy.feature_width, policy.config["hidden_width"])
    restore_module_state(restored, load_checkpoint(destination))

    instance, cg_result = toy_observation("toy-a")
    rows = imitation_candidate_features_batch(
        observation_of("toy-a", instance, cg_result),
        ((4, 0), (3, 0), (2, 1), (1, 2), (0, 3)),
    )
    features = torch.tensor(rows, dtype=torch.float32)
    original = list(policy.module(features).tolist())
    replayed = list(restored(features).tolist())
    for left, right in zip(original, replayed, strict=True):
        assert left == right


def test_train_quality_rl_policy_validates_its_configuration() -> None:
    valid = {"seed": 7} | TRAINING_CONFIG

    for field in ("epochs", "hidden_width", "max_steps"):
        for bad_value in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match=field):
                train_quality_rl_policy(TOY_INSTANCES, **{**valid, field: bad_value})
    for bad_rate in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="learning_rate"):
            train_quality_rl_policy(TOY_INSTANCES, **{**valid, "learning_rate": bad_rate})
    for bad_momentum in (-0.1, 1.0, True, float("nan")):
        with pytest.raises(ValueError, match="baseline_momentum"):
            train_quality_rl_policy(TOY_INSTANCES, **{**valid, "baseline_momentum": bad_momentum})
    for bad_seed in (True, 7.0):
        with pytest.raises(ValueError, match="seed"):
            train_quality_rl_policy(TOY_INSTANCES, **{**valid, "seed": bad_seed})
    with pytest.raises(ValueError, match="non-empty mapping"):
        train_quality_rl_policy({}, **valid)


def test_training_requires_a_converged_verified_classical_start(monkeypatch) -> None:
    class LimitedColumnGeneration(ColumnGeneration):
        def __init__(self, instance, instance_id=None, **kwargs):
            super().__init__(instance, instance_id=instance_id, max_iterations=1)

    monkeypatch.setattr(rl_policy_module, "ColumnGeneration", LimitedColumnGeneration)
    multi_iteration_instance = CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))

    with pytest.raises(ValueError, match="toy-limited"):
        train_quality_rl_policy(
            {"toy-limited": multi_iteration_instance}, seed=7, **TRAINING_CONFIG
        )


def test_pattern_limits_guard_the_action_basis() -> None:
    limits = MaximalPatternLimits(max_search_space_size=1, max_patterns=1)

    with pytest.raises(PatternEnumerationLimitExceeded):
        train_quality_rl_policy(TOY_INSTANCES, seed=7, pattern_limits=limits, **TRAINING_CONFIG)


def test_journal_payload_records_the_complete_experiment(policy) -> None:
    source = {
        "partition_manifest": "data/phase-8-partitions/manifest.json",
        "plan_id": "b8ba8c2065d63bc2b1ba9a130e16751a2584fb2e24f107de055744aa1fc66ae9",
        "partition": "train",
        "instance_ids": list(policy.trained_instance_ids),
        "checkpoint_path": "models/phase-9-quality-policy.pt",
        "checkpoint_sha256": "0" * 64,
    }

    journal = training_journal_payload(policy, source=source)

    assert journal["schema_version"] == TRAINING_JOURNAL_SCHEMA_VERSION
    assert journal["policy_schema_version"] == QUALITY_RL_POLICY_SCHEMA_VERSION
    assert journal["algorithm"]["identifier"] == ALGORITHM_IDENTIFIER
    assert journal["algorithm"]["reference"] == "docs/phase-9-rl-algorithm.md"
    assert journal["source"] == source | {"instance_ids": sorted(source["instance_ids"])}
    assert journal["config"] == policy.config
    assert journal["environment"] == dict(policy.environment)
    assert journal["totals"] == policy.totals
    assert journal["curves"] == policy.curves.to_payload()
    assert len(journal["episodes"]) == len(policy.episodes)
    first = journal["episodes"][0]
    assert first["instance_id"] == policy.episodes[0].instance_id
    assert first["return_value"] == policy.episodes[0].return_value


def test_journal_payload_rejects_incomplete_or_mismatched_sources(policy) -> None:
    complete = {
        "partition_manifest": "manifest.json",
        "plan_id": "plan",
        "partition": "train",
        "instance_ids": list(policy.trained_instance_ids),
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="QualityRLPolicy"):
        training_journal_payload("not-a-policy", source=complete)
    for name in ("partition_manifest", "plan_id", "partition", "checkpoint_path",
                 "checkpoint_sha256"):
        broken = {key: value for key, value in complete.items() if key != name}
        with pytest.raises(ValueError, match=name):
            training_journal_payload(policy, source=broken)
    with pytest.raises(ValueError, match="instance_ids"):
        training_journal_payload(policy, source={**complete, "instance_ids": ["unknown"]})
    with pytest.raises(ValueError, match="instance_ids"):
        training_journal_payload(policy, source={**complete, "instance_ids": "not-a-list"})
