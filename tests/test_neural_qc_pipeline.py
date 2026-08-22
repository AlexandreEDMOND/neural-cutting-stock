"""Neural-QC pipeline integration on real classical starting points."""

import math
from dataclasses import fields as dataclass_fields

import pytest

from neural_cutting_stock.benchmarks.generator import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    SyntheticInstanceGenerator,
)
from neural_cutting_stock.learning import (
    NEURAL_QC_PIPELINE_SCHEMA_VERSION,
    STATUS_IMPROVED,
    STATUS_UNCHANGED,
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_CONVERGED,
    NeuralQCBudget,
    NeuralQCRefinementResult,
    ProposalReview,
    QualityAgentProposal,
    RLQualityAgent,
    run_neural_quality_refinement,
    train_quality_rl_policy,
)
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    CompleteIntegerMaster,
    iter_maximal_patterns,
    verify_plan,
)

GAP_INSTANCE_ID = "gap-instance"


def gap_instance() -> CuttingStockInstance:
    """A retained-family instance whose restricted master provably loses a bar."""

    return SyntheticInstanceGenerator(
        seed=13,
        stock_length=100.0,
        number_of_types=3,
        piece_length_range=(30, 49),
        demand_range=(4, 12),
        length_distribution=TIGHT_RATIO_LENGTH_DISTRIBUTION,
        demand_distribution=AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    ).generate()


@pytest.fixture(scope="module")
def classical():
    instance = gap_instance()
    result = ColumnGeneration(instance, instance_id=GAP_INSTANCE_ID).solve()
    assert result.status == "converged"
    return instance, result


@pytest.fixture(scope="module")
def exact_proposal(classical):
    instance, _ = classical
    exact = CompleteIntegerMaster(instance).solve()
    assert exact.status == 0 and exact.objective_value is not None
    usage = [
        (pattern, value)
        for pattern, value in zip(
            iter_maximal_patterns(instance), exact.column_values, strict=True
        )
        if value > 0
    ]
    return QualityAgentProposal(
        tuple(pattern for pattern, _ in usage), tuple(value for _, value in usage)
    )


class ConstantAgent:
    """Always propose the same plan; the verifier decides its fate."""

    def __init__(self, proposal: QualityAgentProposal | None) -> None:
        self.proposal = proposal

    def propose(self, observation) -> QualityAgentProposal:
        return self.proposal


class EchoAgent:
    """Never improve anything: re-propose the incumbent unchanged."""

    def propose(self, observation) -> QualityAgentProposal:
        return QualityAgentProposal(
            observation.solution_patterns, observation.solution_column_values
        )


class InvalidAgent:
    """Always propose a plan that violates stock capacity."""

    def propose(self, observation) -> QualityAgentProposal:
        return QualityAgentProposal(((2,) * len(observation.demands),), (1,))


def test_schema_version_is_stable() -> None:
    assert NEURAL_QC_PIPELINE_SCHEMA_VERSION == "neural-qc-pipeline-v1"


@pytest.mark.parametrize(
    ("max_steps", "stall_patience"), [(0, 1), (-1, 1), (1, 0), (2.5, 1), (3, True)]
)
def test_budget_rejects_non_positive_or_non_integer_fields(max_steps, stall_patience) -> None:
    with pytest.raises(ValueError):
        NeuralQCBudget(max_steps=max_steps, stall_patience=stall_patience)


def test_budget_rejects_patience_larger_than_the_step_budget() -> None:
    with pytest.raises(ValueError, match="stall_patience"):
        NeuralQCBudget(max_steps=2, stall_patience=3)


def test_pipeline_validates_its_inputs(classical) -> None:
    instance, _ = classical
    budget = NeuralQCBudget(max_steps=3)

    for bad_id in (None, "", "   ", 7):
        with pytest.raises(ValueError, match="instance_id"):
            run_neural_quality_refinement(instance, bad_id, ConstantAgent(None), budget=budget)
    for agent in (None, 42, object()):
        with pytest.raises(ValueError, match="propose"):
            run_neural_quality_refinement(instance, GAP_INSTANCE_ID, agent, budget=budget)
    for bad_budget in (None, (3, 1), "budget"):
        with pytest.raises(ValueError, match="budget"):
            run_neural_quality_refinement(instance, GAP_INSTANCE_ID, EchoAgent(), budget=bad_budget)
    with pytest.raises(ValueError, match="verification_tolerance"):
        run_neural_quality_refinement(
            instance, GAP_INSTANCE_ID, EchoAgent(), budget=budget, verification_tolerance=-1e-9
        )


def test_result_record_rejects_inconsistent_bookkeeping(classical, exact_proposal) -> None:
    instance, cg = classical
    rejected_review = ProposalReview(
        accepted=False,
        errors=("proposal does not reduce the number of stock bars",),
        baseline_verification=cg.verification,
        proposal_verification=cg.verification,
        baseline_bars=12,
        proposed_bars=12,
    )
    fields = {
        "instance_id": GAP_INSTANCE_ID,
        "status": STATUS_UNCHANGED,
        "termination_reason": TERMINATION_CONVERGED,
        "initial_bars": 12,
        "final_bars": 12,
        "steps_taken": 1,
        "accepted_steps": 0,
        "invalid_steps": 0,
        "initial_patterns": cg.patterns,
        "initial_column_values": cg.integer_master_result.column_values,
        "final_patterns": cg.patterns,
        "final_column_values": cg.integer_master_result.column_values,
        "final_verification": cg.verification,
        "reviews": (rejected_review,),
        "total_runtime_seconds": 0.0,
    }
    names = [field.name for field in dataclass_fields(NeuralQCRefinementResult)]

    def build(**overrides):
        return NeuralQCRefinementResult(**{**fields, **overrides})

    assert set(names) == set(fields) | {"schema_version"}
    build()
    with pytest.raises(ValueError, match="instance_id"):
        build(instance_id="")
    with pytest.raises(ValueError, match="status"):
        build(status="degraded")
    with pytest.raises(ValueError, match="termination"):
        build(termination_reason="agent_gave_up")
    with pytest.raises(ValueError, match="steps_taken"):
        build(steps_taken=0)
    with pytest.raises(ValueError, match="cannot exceed steps_taken"):
        build(accepted_steps=2)
    with pytest.raises(ValueError, match="more bars"):
        build(final_bars=13)
    with pytest.raises(ValueError, match="reviews must cover"):
        build(reviews=(rejected_review, rejected_review))
    with pytest.raises(ValueError, match="total_runtime_seconds"):
        build(total_runtime_seconds=-1.0)

    exact_verification = verify_plan(
        instance, exact_proposal.patterns, exact_proposal.column_values
    )
    improved = build(
        final_bars=11,
        status=STATUS_IMPROVED,
        final_patterns=exact_proposal.patterns,
        final_column_values=exact_proposal.column_values,
        final_verification=exact_verification,
    )
    assert improved.bars_saved == 1

    with pytest.raises(ValueError, match="contradicts"):
        build(final_bars=11, status=STATUS_UNCHANGED)


def test_pipeline_starts_from_the_verified_classical_solution(classical, exact_proposal) -> None:
    instance, cg = classical

    result = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, ConstantAgent(exact_proposal), budget=NeuralQCBudget(5)
    )

    assert result.schema_version == NEURAL_QC_PIPELINE_SCHEMA_VERSION
    assert result.initial_patterns == cg.patterns
    assert result.initial_column_values == cg.integer_master_result.column_values
    assert result.initial_bars == int(round(cg.integer_master_result.objective_value))
    assert result.final_verification.feasible
    assert result.final_verification.number_of_stock_bars == result.final_bars
    assert math.isfinite(result.total_runtime_seconds) and result.total_runtime_seconds >= 0


def test_agent_improvement_is_accepted_then_convergence_is_declared(
    classical, exact_proposal
) -> None:
    instance, _ = classical

    result = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, ConstantAgent(exact_proposal), budget=NeuralQCBudget(5)
    )

    assert result.status == STATUS_IMPROVED
    assert result.termination_reason == TERMINATION_CONVERGED
    assert result.initial_bars == 12
    assert result.final_bars == 11
    assert result.bars_saved == 1
    assert result.steps_taken == 2
    assert result.accepted_steps == 1
    assert result.invalid_steps == 0
    assert len(result.reviews) == result.steps_taken
    assert [review.accepted for review in result.reviews] == [True, False]
    assert result.final_patterns == exact_proposal.patterns
    assert result.final_column_values == exact_proposal.column_values
    assert result.final_verification.number_of_stock_bars == 11


def test_declared_improvement_budget_exhaustion_keeps_every_review(
    classical, exact_proposal
) -> None:
    instance, _ = classical

    result = run_neural_quality_refinement(
        instance,
        GAP_INSTANCE_ID,
        ConstantAgent(exact_proposal),
        budget=NeuralQCBudget(max_steps=3, stall_patience=3),
    )

    assert result.status == STATUS_IMPROVED
    assert result.termination_reason == TERMINATION_BUDGET_EXHAUSTED
    assert result.steps_taken == 3
    assert result.accepted_steps == 1
    assert len(result.reviews) == 3
    assert [review.accepted for review in result.reviews] == [True, False, False]
    assert result.final_bars == 11


def test_a_never_improving_agent_converges_without_saving_bars(classical) -> None:
    instance, _ = classical

    result = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, EchoAgent(), budget=NeuralQCBudget(max_steps=4)
    )

    assert result.status == STATUS_UNCHANGED
    assert result.termination_reason == TERMINATION_CONVERGED
    assert result.final_bars == result.initial_bars == 12
    assert result.bars_saved == 0
    assert result.steps_taken == 1
    assert result.accepted_steps == 0
    assert not any(review.accepted for review in result.reviews)


def test_invalid_plans_are_counted_and_never_silently_filtered(classical) -> None:
    instance, _ = classical

    result = run_neural_quality_refinement(
        instance,
        GAP_INSTANCE_ID,
        InvalidAgent(),
        budget=NeuralQCBudget(max_steps=5, stall_patience=2),
    )

    assert result.status == STATUS_UNCHANGED
    assert result.termination_reason == TERMINATION_CONVERGED
    assert result.final_bars == result.initial_bars
    assert result.invalid_steps == 2
    assert all(not review.accepted for review in result.reviews)
    assert all(len(review.errors) > 0 for review in result.reviews)
    assert result.final_verification.feasible


def test_pipeline_requires_a_converged_classical_start(monkeypatch) -> None:
    class LimitedColumnGeneration(ColumnGeneration):
        def __init__(self, instance, instance_id=None, **kwargs):
            super().__init__(instance, instance_id=instance_id, max_iterations=1)

    monkeypatch.setattr(
        "neural_cutting_stock.learning.neural_qc.ColumnGeneration", LimitedColumnGeneration
    )
    multi_iteration_instance = CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))

    with pytest.raises(ValueError, match="did not converge"):
        run_neural_quality_refinement(
            multi_iteration_instance,
            "limited-instance",
            EchoAgent(),
            budget=NeuralQCBudget(3),
        )


def test_pipeline_supports_declared_multi_format_instances() -> None:
    multi = MultiFormatCuttingStockInstance((50.0, 100.0), 0.0, [30.0, 40.0], [6, 3])

    result = run_neural_quality_refinement(
        multi, "multi-format-pipeline", EchoAgent(), budget=NeuralQCBudget(3)
    )

    classical = ColumnGeneration(multi, instance_id="multi-format-pipeline").solve()
    assert result.initial_bars == int(round(classical.integer_master_result.objective_value))
    assert result.status == STATUS_UNCHANGED
    assert result.termination_reason == TERMINATION_CONVERGED
    assert result.bars_saved == 0
    assert result.final_verification.feasible
    assert result.final_verification.number_of_stock_bars == result.final_bars


def test_trained_rl_policy_drives_the_full_pipeline(classical) -> None:
    instance, _ = classical
    training_instances = {
        GAP_INSTANCE_ID: instance,
        "toy-companion": CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3)),
    }

    policy = train_quality_rl_policy(
        training_instances, seed=11, epochs=5, hidden_width=8, max_steps=1
    )

    first = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, RLQualityAgent(policy), budget=NeuralQCBudget(max_steps=2)
    )
    second = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, RLQualityAgent(policy), budget=NeuralQCBudget(max_steps=2)
    )

    for result in (first, second):
        assert result.schema_version == NEURAL_QC_PIPELINE_SCHEMA_VERSION
        assert result.steps_taken >= 1
        assert result.invalid_steps == 0
        assert result.final_verification.feasible
        assert result.final_bars <= result.initial_bars
        assert result.status in (STATUS_IMPROVED, STATUS_UNCHANGED)
    assert (first.final_patterns, first.final_column_values) == (
        second.final_patterns,
        second.final_column_values,
    )


def test_rl_quality_agent_validates_its_policy() -> None:
    with pytest.raises(ValueError, match="QualityRLPolicy"):
        RLQualityAgent("not-a-policy")
