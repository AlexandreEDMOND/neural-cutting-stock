import math

import pytest
import torch

from neural_cutting_stock.learning import (
    IMITATION_BASELINE_SCHEMA_VERSION,
    QUALITY_AGENT_INTERFACE_SCHEMA_VERSION,
    ExactChoiceDemonstration,
    ImitationPolicy,
    ImitationPolicyNetwork,
    ImitationQualityAgent,
    QualityAgentInput,
    QualityAgentProposal,
    QualityRefinementEnv,
    TrainingCurves,
    collect_exact_choice_demonstrations,
    enumerated_candidates,
    imitation_candidate_features_batch,
    load_checkpoint,
    restore_module_state,
    save_checkpoint,
    train_imitation_policy,
)
from neural_cutting_stock.learning import imitation as imitation_module
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
    verify_plan,
)

EXPECTED_BARS = {
    "toy-a": (4, 3),
    "toy-b": (4, 3),
    "toy-c": (4, 3),
    "toy-multi": (4, 3),
}
TRAINING_CONFIG = {"epochs": 3000, "learning_rate": 0.01, "hidden_width": 32}


def toy_instances() -> dict:
    return {
        "toy-a": CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 4)),
        "toy-b": CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 5)),
        "toy-c": CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (6, 4)),
        "toy-multi": MultiFormatCuttingStockInstance((50.0, 120.0), 0.0, [30.0, 40.0], [5, 4]),
    }


def zero_margin_instance() -> CuttingStockInstance:
    return CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))


def observation_of(instance_id: str, instance) -> QualityAgentInput:
    cg_result = ColumnGeneration(instance, instance_id=instance_id).solve()
    assert cg_result.status == "converged"
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


@pytest.fixture(scope="module")
def demonstrations():
    return collect_exact_choice_demonstrations(toy_instances())


@pytest.fixture(scope="module")
def demonstration_by_id(demonstrations):
    return {item.instance_id: item for item in demonstrations}


@pytest.fixture(scope="module")
def policy(demonstrations):
    return train_imitation_policy(demonstrations, seed=42, **TRAINING_CONFIG)


@pytest.fixture(scope="module")
def agent(policy):
    return ImitationQualityAgent(policy)


@pytest.fixture(scope="module")
def zero_margin_setup():
    instances = toy_instances()
    instances["toy-zero"] = zero_margin_instance()
    demonstrations = collect_exact_choice_demonstrations(instances)
    policy = train_imitation_policy(demonstrations, seed=42, **TRAINING_CONFIG)
    by_id = {item.instance_id: item for item in demonstrations}
    return by_id["toy-zero"], ImitationQualityAgent(policy)


def test_schema_version_is_stable() -> None:
    assert IMITATION_BASELINE_SCHEMA_VERSION == "quality-imitation-baseline-v1"


def test_demonstrations_follow_sorted_instance_order(demonstrations) -> None:
    assert [item.instance_id for item in demonstrations] == sorted(EXPECTED_BARS)


@pytest.mark.parametrize("instance_id", sorted(EXPECTED_BARS))
def test_demonstration_carries_verified_classical_and_exact_endpoints(
    demonstration_by_id, instance_id
) -> None:
    demonstration = demonstration_by_id[instance_id]
    expected_baseline, expected_expert = EXPECTED_BARS[instance_id]
    verification = verify_plan(
        toy_instances()[instance_id],
        demonstration.expert_proposal.patterns,
        demonstration.expert_proposal.column_values,
    )

    assert demonstration.observation.schema_version == QUALITY_AGENT_INTERFACE_SCHEMA_VERSION
    assert demonstration.observation.column_pool == demonstration.observation.solution_patterns
    assert demonstration.baseline_bars == expected_baseline
    assert demonstration.expert_bars == expected_expert
    assert demonstration.bars_saved == 1
    assert verification.feasible
    assert verification.number_of_stock_bars == expected_expert


def test_expert_proposals_stay_within_the_enumerated_basis(demonstrations) -> None:
    for demonstration in demonstrations:
        basis = set(enumerated_candidates(demonstration.observation))

        assert set(demonstration.expert_proposal.patterns) <= basis


def test_zero_margin_instance_is_collected_without_a_fake_gain() -> None:
    demonstrations = collect_exact_choice_demonstrations({"toy-zero": zero_margin_instance()})
    demonstration = demonstrations[0]

    assert demonstration.baseline_bars == demonstration.expert_bars == 3
    assert demonstration.bars_saved == 0


def test_collect_requires_a_converged_classical_run(monkeypatch) -> None:
    class LimitedColumnGeneration(ColumnGeneration):
        def __init__(self, instance, instance_id=None, **kwargs):
            super().__init__(instance, instance_id=instance_id, max_iterations=1)

    monkeypatch.setattr(imitation_module, "ColumnGeneration", LimitedColumnGeneration)

    with pytest.raises(ValueError, match="toy-limited"):
        collect_exact_choice_demonstrations({"toy-limited": zero_margin_instance()})


def test_features_have_a_fixed_permutation_invariant_width(demonstrations) -> None:
    widths = set()
    for demonstration in demonstrations:
        candidates = enumerated_candidates(demonstration.observation)
        rows = imitation_candidate_features_batch(demonstration.observation, candidates)

        assert len(rows) == len(candidates)
        widths.update(len(row) for row in rows)

    assert widths == {28}


def test_features_are_invariant_under_a_joint_type_permutation() -> None:
    instance = CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 4))
    observation = observation_of("permuted", instance)
    reversed_observation = QualityAgentInput(
        instance_id="permuted",
        stock_length=observation.stock_length,
        kerf=observation.kerf,
        piece_lengths=tuple(reversed(observation.piece_lengths)),
        demands=tuple(reversed(observation.demands)),
        column_pool=tuple(tuple(reversed(pattern)) for pattern in observation.column_pool),
        solution_patterns=tuple(
            tuple(reversed(pattern)) for pattern in observation.solution_patterns
        ),
        solution_column_values=observation.solution_column_values,
    )
    forward_basis = enumerated_candidates(observation)
    backward_basis = [tuple(reversed(pattern)) for pattern in forward_basis]
    forward_rows = imitation_candidate_features_batch(observation, forward_basis)
    backward_rows = imitation_candidate_features_batch(reversed_observation, backward_basis)

    assert len(forward_rows) == len(backward_rows)
    for pattern, row in zip(forward_basis, forward_rows, strict=True):
        index = backward_basis.index(tuple(reversed(pattern)))

        assert backward_rows[index] == row


def test_training_records_a_complete_loss_curve(policy) -> None:
    points = policy.curves.points

    assert policy.schema_version == IMITATION_BASELINE_SCHEMA_VERSION
    assert policy.feature_width == 28
    assert policy.seed == 42
    assert policy.config["activation"] == "tanh"
    assert policy.config["epochs"] == TRAINING_CONFIG["epochs"]
    assert [point.step for point in points] == list(range(TRAINING_CONFIG["epochs"]))
    assert points[0].metrics["loss"] > points[-1].metrics["loss"] > 0.0
    assert all(math.isfinite(point.metrics["loss"]) for point in points)


def test_training_is_reproducible_for_a_fixed_seed(demonstrations, policy) -> None:
    replayed = train_imitation_policy(demonstrations, seed=42, **TRAINING_CONFIG)

    original_state = policy.module.state_dict()
    replayed_state = replayed.module.state_dict()
    assert set(original_state) == set(replayed_state)
    for key, value in original_state.items():
        assert torch.equal(replayed_state[key], value)


def test_trained_policy_clones_the_exact_choice_on_demonstrated_instances(
    demonstrations, agent
) -> None:
    for demonstration in demonstrations:

        assert agent.propose(demonstration.observation) == demonstration.expert_proposal


@pytest.mark.parametrize("instance_id", sorted(EXPECTED_BARS))
def test_refinement_env_accepts_the_imitated_plan_and_records_the_verified_gain(
    demonstration_by_id, agent, instance_id
) -> None:
    demonstration = demonstration_by_id[instance_id]
    env = QualityRefinementEnv(
        toy_instances()[instance_id],
        instance_id,
        demonstration.observation.column_pool,
        demonstration.observation.solution_patterns,
        demonstration.observation.solution_column_values,
        max_steps=1,
    )
    observation = env.reset()

    outcome = env.step(agent.propose(observation))

    assert outcome.accepted
    assert outcome.reward == float(demonstration.bars_saved) == 1.0
    assert outcome.review.accepted
    assert outcome.review.proposal_verification.feasible
    assert env.current_bars == demonstration.expert_bars == 3
    assert env.total_bars_saved == 1
    assert outcome.truncated


def test_zero_margin_clone_verifies_but_is_never_accepted(zero_margin_setup) -> None:
    demonstration, agent = zero_margin_setup
    proposal = agent.propose(demonstration.observation)
    env = QualityRefinementEnv(
        zero_margin_instance(),
        "toy-zero",
        demonstration.observation.column_pool,
        demonstration.observation.solution_patterns,
        demonstration.observation.solution_column_values,
        max_steps=2,
    )
    env.reset()
    verification = verify_plan(zero_margin_instance(), proposal.patterns, proposal.column_values)

    outcome = env.step(proposal)

    assert verification.feasible
    assert verification.number_of_stock_bars == 3
    assert not outcome.accepted
    assert outcome.reward == 0.0
    assert any("does not reduce" in error for error in outcome.review.errors)
    assert env.current_bars == env.initial_bars == 3
    assert env.total_bars_saved == 0


def test_unseen_instance_yields_a_wellformed_reviewed_proposal(agent) -> None:
    unseen = CuttingStockInstance(100.0, 0.0, (30.0, 45.0), (4, 4))
    observation = observation_of("unseen", unseen)
    env = QualityRefinementEnv(
        unseen,
        "unseen",
        observation.column_pool,
        observation.solution_patterns,
        observation.solution_column_values,
        max_steps=1,
    )
    env.reset()

    proposal = agent.propose(observation)
    outcome = env.step(proposal)

    assert len(proposal.patterns) == len(proposal.column_values)
    assert len(set(proposal.patterns)) == len(proposal.patterns)
    assert all(value >= 0 for value in proposal.column_values)
    assert outcome.review.baseline_verification.feasible
    if outcome.accepted:
        assert env.current_bars < env.initial_bars
        assert outcome.reward == float(env.initial_bars - env.current_bars)
    else:
        assert env.current_bars == env.initial_bars
        assert env.total_bars_saved == 0


def test_checkpoint_round_trip_preserves_the_decoded_plan(
    policy, demonstration_by_id, tmp_path
) -> None:
    destination = tmp_path / "checkpoints" / "imitation.pt"
    save_checkpoint(
        destination,
        module=policy.module,
        seed=policy.seed,
        config=dict(policy.config),
        curves=policy.curves,
    )
    restored_network = ImitationPolicyNetwork(policy.feature_width, policy.config["hidden_width"])
    restore_module_state(restored_network, load_checkpoint(destination))
    restored_agent = ImitationQualityAgent(
        ImitationPolicy(
            module=restored_network,
            feature_width=policy.feature_width,
            seed=policy.seed,
            config=policy.config,
            curves=policy.curves,
        )
    )
    demonstration = demonstration_by_id["toy-a"]

    reference = ImitationQualityAgent(policy).propose(demonstration.observation)
    assert restored_agent.propose(demonstration.observation) == reference


def test_network_rejects_invalid_widths() -> None:
    for name in ("feature_width", "hidden_width"):
        for bad_value in (0, -1, True, 2.5):
            kwargs = {"feature_width": 28, "hidden_width": 8}
            kwargs[name] = bad_value

            with pytest.raises(ValueError, match=name):
                ImitationPolicyNetwork(**kwargs)


def test_train_imitation_policy_validates_its_configuration(demonstrations) -> None:
    valid = {"seed": 42} | TRAINING_CONFIG

    for field in ("epochs", "hidden_width"):
        for bad_value in (0, -1, True, 2.5):
            broken = valid | {field: bad_value}

            with pytest.raises(ValueError):
                train_imitation_policy(demonstrations, **broken)

    for bad_rate in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            train_imitation_policy(demonstrations, **(valid | {"learning_rate": bad_rate}))

    with pytest.raises(ValueError, match="at least one demonstration"):
        train_imitation_policy((), **valid)
    with pytest.raises(ValueError, match="sequence of ExactChoiceDemonstration"):
        train_imitation_policy(("not-a-demonstration",), **valid)


def test_agent_and_enumeration_require_a_normalized_observation() -> None:
    unsorted_observation = QualityAgentInput(
        instance_id="unsorted",
        stock_length=120.0,
        kerf=0.0,
        piece_lengths=(40.0, 30.0),
        demands=(4, 5),
        column_pool=((0, 4), (3, 0)),
        solution_patterns=((0, 4), (3, 0)),
        solution_column_values=(2, 2),
    )
    network = ImitationPolicyNetwork(28, 8)
    agent = ImitationQualityAgent(
        ImitationPolicy(
            module=network,
            feature_width=network.feature_width,
            seed=1,
            config={},
            curves=TrainingCurves(),
        )
    )

    with pytest.raises(ValueError, match="normalized"):
        enumerated_candidates(unsorted_observation)
    with pytest.raises(ValueError, match="normalized"):
        agent.propose(unsorted_observation)


def test_pattern_limits_guard_the_action_basis(demonstration_by_id) -> None:
    limits = MaximalPatternLimits(max_search_space_size=1, max_patterns=1)

    with pytest.raises(PatternEnumerationLimitExceeded):
        enumerated_candidates(demonstration_by_id["toy-a"].observation, limits)


def test_demonstration_dataclass_validates_its_fields() -> None:
    observation = QualityAgentInput(
        instance_id="demo",
        stock_length=120.0,
        kerf=0.0,
        piece_lengths=(30.0, 40.0),
        demands=(5, 4),
        column_pool=((4, 0),),
        solution_patterns=((4, 0),),
        solution_column_values=(2,),
    )
    base = {
        "instance_id": "demo",
        "observation": observation,
        "expert_proposal": QualityAgentProposal(((4, 0),), (2,)),
        "baseline_bars": 4,
        "expert_bars": 3,
    }

    assert ExactChoiceDemonstration(**base).bars_saved == 1
    for field in ("baseline_bars", "expert_bars"):
        for bad_value in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match=field):
                ExactChoiceDemonstration(**(base | {field: bad_value}))
    with pytest.raises(ValueError, match="instance_id"):
        ExactChoiceDemonstration(**(base | {"instance_id": ""}))


def test_agent_refuses_foreign_policies() -> None:
    with pytest.raises(ValueError, match="ImitationPolicy"):
        ImitationQualityAgent("not-a-policy")
