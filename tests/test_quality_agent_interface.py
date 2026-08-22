import math

import pytest

from neural_cutting_stock.learning import (
    QUALITY_AGENT_INTERFACE_SCHEMA_VERSION,
    ProposalReview,
    QualityAgent,
    QualityAgentInput,
    QualityAgentProposal,
    verify_proposal,
)
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance

BASELINE_PATTERNS = ((3, 0), (0, 2))
BASELINE_VALUES = (2, 2)
IMPROVED_PROPOSAL = QualityAgentProposal(((2, 1),), (3,))
INSTANCE = CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))


def make_observation(**overrides: object) -> QualityAgentInput:
    fields: dict[str, object] = {
        "instance_id": "quality-instance",
        "stock_length": 100.0,
        "kerf": 0.0,
        "piece_lengths": (30.0, 40.0),
        "demands": (6, 3),
        "column_pool": BASELINE_PATTERNS,
        "solution_patterns": BASELINE_PATTERNS,
        "solution_column_values": BASELINE_VALUES,
    }
    fields.update(overrides)
    return QualityAgentInput(**fields)


class FixedProposalAgent:
    def __init__(self, proposal: QualityAgentProposal) -> None:
        self.proposal = proposal

    def propose(self, observation: QualityAgentInput) -> QualityAgentProposal:
        del observation
        return self.proposal


def test_schema_version_is_stable() -> None:
    assert QUALITY_AGENT_INTERFACE_SCHEMA_VERSION == "quality-agent-interface-v1"


def test_input_defaults_to_the_versioned_schema() -> None:
    assert make_observation().schema_version == QUALITY_AGENT_INTERFACE_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "learning-interface-v1"),
        ("instance_id", ""),
        ("instance_id", "   "),
        ("stock_length", 0.0),
        ("stock_length", -1.0),
        ("stock_length", float("inf")),
        ("stock_length", float("nan")),
        ("kerf", -0.5),
        ("piece_lengths", ()),
        ("piece_lengths", (30.0,)),
        ("piece_lengths", (30.0, -4.0)),
        ("piece_lengths", (30.0, float("inf"))),
        ("demands", (6, 0)),
        ("demands", (6, -1)),
        ("demands", (6, True)),
        ("demands", (6, 1.5)),
        ("column_pool", ((3,), (0, 2))),
        ("column_pool", ((-1, 0),)),
        ("solution_patterns", ((3,), (0, 2))),
        ("solution_patterns", ((3, 0), (-1, 2))),
        ("solution_column_values", (-1, 2)),
        ("solution_column_values", (True, 2)),
        ("solution_column_values", (1.5, 2)),
    ],
)
def test_input_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_observation(**{field: value})


def test_input_rejects_mismatched_solution_shapes() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        make_observation(solution_column_values=(1,))


def test_proposal_rejects_duplicate_patterns() -> None:
    with pytest.raises(ValueError, match="distinct"):
        QualityAgentProposal(((2, 1), (2, 1)), (2, 1))


def test_proposal_rejects_shape_and_value_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        QualityAgentProposal(((2, 1),), (3, 1))
    with pytest.raises(ValueError, match="non-negative integers"):
        QualityAgentProposal(((2, 1),), (-3,))
    with pytest.raises(ValueError, match="non-negative integers"):
        QualityAgentProposal(((2, 1),), (True,))
    with pytest.raises(ValueError, match="non-negative integers"):
        QualityAgentProposal(((2, 1),), (3.0,))


def test_verify_proposal_accepts_a_verified_improvement() -> None:
    review = verify_proposal(INSTANCE, make_observation(), IMPROVED_PROPOSAL)

    assert isinstance(review, ProposalReview)
    assert review.accepted
    assert review.errors == ()
    assert review.baseline_bars == 4
    assert review.proposed_bars == 3
    assert review.bars_saved == 1
    assert review.baseline_verification.feasible
    assert review.proposal_verification.feasible


def test_verify_proposal_exercises_kerf_in_accepted_improvements() -> None:
    instance = CuttingStockInstance(103.0, 1.0, (30.0, 40.0), (6, 3))
    observation = make_observation(stock_length=103.0, kerf=1.0)

    review = verify_proposal(instance, observation, IMPROVED_PROPOSAL)

    assert review.accepted
    assert review.proposed_bars == 3
    assert review.proposal_verification.kerf_loss == 3 * 1.0 * 3


def test_verify_proposal_supports_declared_multi_format_instances() -> None:
    instance = MultiFormatCuttingStockInstance((50.0, 100.0), 0.0, [30.0, 40.0], [6, 3])

    review = verify_proposal(instance, make_observation(), IMPROVED_PROPOSAL)

    assert review.accepted
    assert review.baseline_bars == 4
    assert review.proposed_bars == 3
    assert review.bars_saved == 1


def test_verify_proposal_rejects_capacity_violating_plans_from_the_agent() -> None:
    proposal = QualityAgentProposal(((2, 2),), (1,))

    review = verify_proposal(INSTANCE, make_observation(), proposal)

    assert not review.accepted
    assert any("exceeds stock capacity" in error for error in review.errors)
    assert all(error.startswith("proposal: ") for error in review.errors)


def test_verify_proposal_rejects_incomplete_coverage() -> None:
    proposal = QualityAgentProposal(((1, 0),), (6,))

    review = verify_proposal(INSTANCE, make_observation(), proposal)

    assert not review.accepted
    assert any("does not cover every demand" in error for error in review.errors)


def test_verify_proposal_exercises_kerf_in_rejected_plans() -> None:
    instance = CuttingStockInstance(10.0, 1.0, (6.0, 4.0), (1, 2))
    observation = make_observation(
        stock_length=10.0,
        kerf=1.0,
        piece_lengths=(4.0, 6.0),
        demands=(2, 1),
        column_pool=((2, 0), (0, 1)),
        solution_patterns=((2, 0), (0, 1)),
        solution_column_values=(1, 1),
    )

    review = verify_proposal(instance, observation, QualityAgentProposal(((1, 1),), (1,)))

    assert not review.accepted
    assert any("exceeds stock capacity" in error for error in review.errors)


def test_verify_proposal_preserves_equal_objectives_as_rejections() -> None:
    proposal = QualityAgentProposal(BASELINE_PATTERNS, BASELINE_VALUES)

    review = verify_proposal(INSTANCE, make_observation(), proposal)

    assert not review.accepted
    assert review.errors == (
        "proposal does not reduce the number of stock bars",
    )
    assert review.bars_saved == 0


def test_verify_proposal_records_worse_plans_without_silent_filtering() -> None:
    proposal = QualityAgentProposal(BASELINE_PATTERNS, (3, 2))

    review = verify_proposal(INSTANCE, make_observation(), proposal)

    assert not review.accepted
    assert review.errors == ("proposal does not reduce the number of stock bars",)
    assert review.baseline_bars == 4
    assert review.proposed_bars == 5
    assert review.bars_saved == -1


def test_verify_proposal_refuses_to_compare_against_an_infeasible_baseline() -> None:
    observation = make_observation(solution_column_values=(1, 1))

    review = verify_proposal(INSTANCE, observation, IMPROVED_PROPOSAL)

    assert not review.accepted
    assert any(
        error.startswith("baseline: ") and "does not cover every demand" in error
        for error in review.errors
    )
    assert not review.baseline_verification.feasible


def test_verify_proposal_checks_instance_and_observation_coherence() -> None:
    with pytest.raises(ValueError, match="demands do not match"):
        verify_proposal(INSTANCE, make_observation(demands=(6, 4)), IMPROVED_PROPOSAL)
    with pytest.raises(ValueError, match="stock_length does not match"):
        verify_proposal(INSTANCE, make_observation(stock_length=90.0), IMPROVED_PROPOSAL)
    with pytest.raises(ValueError, match="kerf does not match"):
        verify_proposal(INSTANCE, make_observation(kerf=0.5), IMPROVED_PROPOSAL)
    with pytest.raises(ValueError, match="piece_lengths do not match"):
        verify_proposal(INSTANCE, make_observation(piece_lengths=(31.0, 40.0)), IMPROVED_PROPOSAL)


@pytest.mark.parametrize("tolerance", [-1e-9, float("nan"), float("inf")])
def test_verify_proposal_validates_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        verify_proposal(INSTANCE, make_observation(), IMPROVED_PROPOSAL, tolerance)


def test_quality_agent_protocol_flows_through_the_review_contract() -> None:
    agent: QualityAgent = FixedProposalAgent(IMPROVED_PROPOSAL)
    observation = make_observation()

    review = verify_proposal(INSTANCE, observation, agent.propose(observation))

    assert review.accepted
    assert math.isclose(review.baseline_verification.total_waste, 400.0 - 300.0)
