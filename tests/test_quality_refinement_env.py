import pytest

from neural_cutting_stock.learning import (
    DEFAULT_INVALID_PLAN_PENALTY,
    QUALITY_AGENT_INTERFACE_SCHEMA_VERSION,
    QUALITY_REFINEMENT_ENV_SCHEMA_VERSION,
    QualityAgentProposal,
    QualityRefinementEnv,
)
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance

BASELINE_PATTERNS = ((3, 0), (0, 2))
BASELINE_VALUES = (2, 2)
IMPROVED_PROPOSAL = QualityAgentProposal(((2, 1),), (3,))
INSTANCE = CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))

CHAIN_INSTANCE = CuttingStockInstance(100.0, 0.0, (45.0, 30.0), (4, 4))
CHAIN_PATTERNS = ((1, 1), (2, 0), (0, 2))


def make_env(**overrides: object) -> QualityRefinementEnv:
    fields: dict[str, object] = {
        "instance": INSTANCE,
        "instance_id": "quality-env-instance",
        "column_pool": BASELINE_PATTERNS,
        "solution_patterns": BASELINE_PATTERNS,
        "solution_column_values": BASELINE_VALUES,
    }
    fields.update(overrides)
    return QualityRefinementEnv(**fields)


def make_chain_env(max_steps: int = 5) -> QualityRefinementEnv:
    return QualityRefinementEnv(
        CHAIN_INSTANCE,
        "chain-instance",
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
        (4, 4),
        max_steps=max_steps,
    )


def test_schema_version_is_stable() -> None:
    assert QUALITY_REFINEMENT_ENV_SCHEMA_VERSION == "quality-refinement-env-v1"


def test_reset_returns_the_pool_and_solution_observation() -> None:
    env = make_env()

    observation = env.reset()

    assert observation.schema_version == QUALITY_AGENT_INTERFACE_SCHEMA_VERSION
    assert observation.instance_id == "quality-env-instance"
    assert observation.stock_length == 100.0
    assert observation.kerf == 0.0
    assert observation.piece_lengths == (30.0, 40.0)
    assert observation.demands == (6, 3)
    assert observation.column_pool == BASELINE_PATTERNS
    assert observation.solution_patterns == BASELINE_PATTERNS
    assert observation.solution_column_values == BASELINE_VALUES
    assert env.steps_taken == 0
    assert env.reviews == ()
    assert env.initial_bars == 4
    assert env.current_bars == 4


def test_accepted_improvement_scores_the_verified_bar_reduction() -> None:
    env = make_env()

    outcome = env.step(IMPROVED_PROPOSAL)

    assert outcome.accepted
    assert outcome.reward == 1.0
    assert isinstance(outcome.reward, float)
    assert outcome.review.accepted
    assert outcome.total_bars_saved == 1
    assert outcome.steps_taken == 1
    assert not outcome.truncated
    assert outcome.observation.solution_patterns == ((2, 1),)
    assert outcome.observation.solution_column_values == (3,)
    assert env.current_bars == 3
    assert env.total_bars_saved == 1


def test_acceptance_strictly_reduces_the_documented_total_waste() -> None:
    env = make_env()
    requested_length = 6 * 30.0 + 3 * 40.0

    assert env.current_waste == pytest.approx(4 * 100.0 - requested_length)
    outcome = env.step(IMPROVED_PROPOSAL)

    assert outcome.reward == 1.0
    assert env.current_waste == pytest.approx(3 * 100.0 - requested_length)


def test_invalid_plan_receives_the_strict_penalty_and_keeps_the_incumbent() -> None:
    env = make_env()

    outcome = env.step(QualityAgentProposal(((2, 2),), (1,)))

    assert not outcome.accepted
    assert outcome.reward == DEFAULT_INVALID_PLAN_PENALTY
    assert outcome.reward < 0
    assert any("exceeds stock capacity" in error for error in outcome.review.errors)
    assert outcome.observation.solution_patterns == BASELINE_PATTERNS
    assert outcome.observation.solution_column_values == BASELINE_VALUES
    assert env.current_bars == 4
    assert env.total_bars_saved == 0
    assert env.reviews == (outcome.review,)


def test_valid_equal_or_worse_plans_score_their_signed_bar_reduction() -> None:
    env = make_env()

    equal_outcome = env.step(QualityAgentProposal(BASELINE_PATTERNS, BASELINE_VALUES))
    worse_outcome = env.step(QualityAgentProposal(BASELINE_PATTERNS, (3, 2)))

    assert not equal_outcome.accepted
    assert equal_outcome.reward == 0.0
    assert not worse_outcome.accepted
    assert worse_outcome.reward == -1.0
    assert worse_outcome.review.proposed_bars == 5
    assert env.current_bars == 4
    assert env.total_bars_saved == 0


def test_a_declared_penalty_is_honored_for_invalid_plans() -> None:
    env = make_env(invalid_plan_penalty=-2.5)

    outcome = env.step(QualityAgentProposal(((0, 0),), (1,)))

    assert not outcome.accepted
    assert outcome.reward == -2.5


@pytest.mark.parametrize("max_steps", [0, -1, True, 2.5])
def test_constructor_rejects_invalid_step_budgets(max_steps: object) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        make_env(max_steps=max_steps)


@pytest.mark.parametrize("penalty", [0.0, 1.0, float("nan"), float("inf"), True])
def test_constructor_requires_a_finite_strictly_negative_penalty(penalty: float) -> None:
    with pytest.raises(ValueError, match="invalid_plan_penalty"):
        make_env(invalid_plan_penalty=penalty)


@pytest.mark.parametrize("tolerance", [-1e-9, float("nan"), float("inf")])
def test_constructor_validates_the_verification_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        make_env(tolerance=tolerance)


def test_constructor_refuses_an_initial_solution_that_does_not_verify() -> None:
    with pytest.raises(ValueError, match="does not verify"):
        make_env(solution_column_values=(1, 1))


def test_episode_truncates_at_the_declared_budget_and_then_refuses_steps() -> None:
    env = make_env(max_steps=2)

    first = env.step(QualityAgentProposal(BASELINE_PATTERNS, BASELINE_VALUES))
    second = env.step(IMPROVED_PROPOSAL)

    assert not first.truncated
    assert second.truncated
    assert env.steps_taken == env.max_steps == 2
    assert env.total_bars_saved == 1
    with pytest.raises(RuntimeError, match="episode budget is exhausted"):
        env.step(IMPROVED_PROPOSAL)


def test_iterative_refinement_accumulates_verified_savings() -> None:
    env = make_chain_env()
    assert env.initial_bars == 8

    first = env.step(QualityAgentProposal(CHAIN_PATTERNS, (2, 1, 2)))
    second = env.step(QualityAgentProposal(CHAIN_PATTERNS, (2, 1, 1)))

    assert first.accepted
    assert first.reward == 3.0
    assert second.accepted
    assert second.reward == 1.0
    assert env.current_bars == 4
    assert env.total_bars_saved == 4
    assert env.steps_taken == 2
    assert second.observation.solution_column_values == (2, 1, 1)
    assert len(env.reviews) == 2


def test_reset_restores_the_classical_starting_point_after_an_episode() -> None:
    env = make_chain_env()
    env.step(QualityAgentProposal(CHAIN_PATTERNS, (2, 1, 2)))

    observation = env.reset()

    assert observation.solution_patterns == ((1, 0), (0, 1))
    assert observation.solution_column_values == (4, 4)
    assert env.steps_taken == 0
    assert env.reviews == ()
    assert env.current_bars == env.initial_bars == 8
    assert env.total_bars_saved == 0


def test_reset_reenables_stepping_after_truncation() -> None:
    env = make_env(max_steps=1)
    env.step(IMPROVED_PROPOSAL)

    env.reset()

    outcome = env.step(IMPROVED_PROPOSAL)
    assert outcome.accepted
    assert outcome.reward == 1.0


def test_the_column_pool_stays_constant_while_the_solution_improves() -> None:
    env = make_env()

    outcome = env.step(IMPROVED_PROPOSAL)

    assert outcome.observation.column_pool == BASELINE_PATTERNS
    assert ((2, 1),) not in outcome.observation.column_pool


def test_kerf_exercised_instances_flow_through_the_environment() -> None:
    instance = CuttingStockInstance(135.0, 1.0, (30.0, 40.0), (6, 3))
    env = QualityRefinementEnv(
        instance,
        "kerf-instance",
        ((2, 0), (0, 2)),
        ((2, 0), (0, 2)),
        (3, 2),
    )
    assert env.initial_bars == 5

    outcome = env.step(QualityAgentProposal(((2, 1),), (3,)))

    assert outcome.accepted
    assert outcome.reward == 2.0
    assert outcome.review.proposal_verification.kerf_loss == pytest.approx(9.0)
    assert env.current_bars == 3
    assert env.current_waste == pytest.approx(3 * 135.0 - 300.0)


def test_declared_multi_format_instances_are_supported() -> None:
    instance = MultiFormatCuttingStockInstance((50.0, 100.0), 0.0, [30.0, 40.0], [6, 3])
    env = QualityRefinementEnv(
        instance,
        "multi-format-instance",
        BASELINE_PATTERNS,
        BASELINE_PATTERNS,
        BASELINE_VALUES,
    )

    outcome = env.step(IMPROVED_PROPOSAL)

    assert outcome.accepted
    assert outcome.reward == 1.0
    assert env.current_bars == 3
