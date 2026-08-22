"""Publication guardrails of the Neural-QC pipeline on real refinement results."""

from dataclasses import replace

import pytest

from neural_cutting_stock.benchmarks.generator import (
    AWKWARD_DIVISIBILITY_DEMAND_DISTRIBUTION,
    TIGHT_RATIO_LENGTH_DISTRIBUTION,
    SyntheticInstanceGenerator,
)
from neural_cutting_stock.learning import (
    FAILURE_BASELINE_VERIFICATION,
    FAILURE_FINAL_VERIFICATION,
    FAILURE_REFINEMENT_ERROR,
    NEURAL_QC_PIPELINE_SCHEMA_VERSION,
    NEURAL_QC_PUBLICATION_SCHEMA_VERSION,
    OUTCOME_FAILURE,
    OUTCOME_SOLUTION,
    PUBLICATION_STATUS_DEGRADED,
    PUBLICATION_STATUS_EQUAL,
    PUBLICATION_STATUS_IMPROVED,
    STATUS_UNCHANGED,
    NeuralQCAttemptRecord,
    NeuralQCBudget,
    QualityAgentProposal,
    attempt_quality_refinement,
    classify_publication_status,
    publish_refinement_result,
    run_neural_quality_refinement,
)
from neural_cutting_stock.problem import CuttingStockInstance, MultiFormatCuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    CompleteIntegerMaster,
    PlanVerification,
    iter_maximal_patterns,
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


class EchoAgent:
    """Never improve anything: re-propose the incumbent unchanged."""

    def propose(self, observation) -> QualityAgentProposal:
        return QualityAgentProposal(
            observation.solution_patterns, observation.solution_column_values
        )


class ConstantAgent:
    """Always propose the same plan; the verifier decides its fate."""

    def __init__(self, proposal: QualityAgentProposal | None) -> None:
        self.proposal = proposal

    def propose(self, observation) -> QualityAgentProposal:
        return self.proposal


class CrashingAgent:
    """Simulate an agent whose inference crashes mid-campaign."""

    def propose(self, observation) -> QualityAgentProposal:
        raise RuntimeError("boom")


def feasible_verification(number_of_stock_bars: int) -> PlanVerification:
    return PlanVerification(
        feasible=True,
        errors=(),
        number_of_stock_bars=number_of_stock_bars,
        produced_counts=(0,),
        requested_length=1.0,
        produced_length=0.0,
        overproduction_length=-1.0,
        kerf_loss=0.0,
        trim_loss=0.0,
        total_waste=0.0,
    )


@pytest.fixture(scope="module")
def refined_unchanged(classical):
    instance, _ = classical
    result = run_neural_quality_refinement(
        instance, GAP_INSTANCE_ID, EchoAgent(), budget=NeuralQCBudget(3)
    )
    assert result.schema_version == NEURAL_QC_PIPELINE_SCHEMA_VERSION
    assert result.status == STATUS_UNCHANGED
    return instance, result


def test_schema_version_is_stable() -> None:
    assert NEURAL_QC_PUBLICATION_SCHEMA_VERSION == "neural-qc-publication-v1"


@pytest.mark.parametrize(
    ("initial_bars", "final_bars", "expected"),
    [
        (12, 11, PUBLICATION_STATUS_IMPROVED),
        (12, 12, PUBLICATION_STATUS_EQUAL),
        (12, 13, PUBLICATION_STATUS_DEGRADED),
    ],
)
def test_status_classification_is_honest_by_signed_bar_difference(
    initial_bars, final_bars, expected
) -> None:
    assert classify_publication_status(initial_bars, final_bars) == expected


@pytest.mark.parametrize(("initial_bars", "final_bars"), [(0, 1), (-2, -3), (12.0, 13), (12, True)])
def test_status_classification_validates_its_inputs(initial_bars, final_bars) -> None:
    with pytest.raises(ValueError):
        classify_publication_status(initial_bars, final_bars)
def test_published_solution_record_rejects_unverified_or_mislabeled_plans(
    refined_unchanged,
) -> None:
    _, result = refined_unchanged

    record = publish_refinement_result(gap_instance(), result)
    assert record.solution is not None
    solution = record.solution

    with pytest.raises(ValueError, match="instance_id"):
        replace(solution, instance_id="")
    with pytest.raises(ValueError, match="schema_version"):
        replace(solution, schema_version="neural-qc-publication-v2")
    with pytest.raises(ValueError, match="publication status"):
        replace(solution, status="faster")
    with pytest.raises(ValueError, match="positive integer"):
        replace(solution, final_bars=0)
    with pytest.raises(ValueError, match="unverified is publishable"):
        replace(solution, final_verification=replace(feasible_verification(12), feasible=False))
    with pytest.raises(ValueError, match="baseline_verification disagrees"):
        replace(solution, baseline_verification=feasible_verification(13))
    with pytest.raises(ValueError, match="final_verification disagrees"):
        replace(solution, final_verification=feasible_verification(13))
    with pytest.raises(ValueError, match="contradicts"):
        replace(solution, status=PUBLICATION_STATUS_DEGRADED)
    with pytest.raises(ValueError, match="contradicts"):
        replace(solution, status=PUBLICATION_STATUS_IMPROVED)


def test_unchanged_episode_publishes_an_honest_equal_solution(refined_unchanged, classical) -> None:
    _, cg = classical
    instance, result = refined_unchanged

    record = publish_refinement_result(instance, result)

    assert record.schema_version == NEURAL_QC_PUBLICATION_SCHEMA_VERSION
    assert record.outcome == OUTCOME_SOLUTION
    assert record.failure_reason is None
    assert record.failure_message is None
    solution = record.solution
    assert solution is not None
    assert solution.instance_id == GAP_INSTANCE_ID
    assert solution.status == PUBLICATION_STATUS_EQUAL
    assert solution.initial_bars == int(round(cg.integer_master_result.objective_value))
    assert solution.final_bars == solution.initial_bars
    assert solution.bars_saved == 0
    assert solution.baseline_verification.feasible
    assert solution.final_verification.feasible
    assert solution.final_verification.number_of_stock_bars == solution.final_bars


def test_improving_agent_publishes_an_honest_improved_solution(classical, exact_proposal) -> None:
    instance, _ = classical

    record = attempt_quality_refinement(
        instance, GAP_INSTANCE_ID, ConstantAgent(exact_proposal), budget=NeuralQCBudget(5)
    )

    assert record.outcome == OUTCOME_SOLUTION
    solution = record.solution
    assert solution is not None
    assert solution.status == PUBLICATION_STATUS_IMPROVED
    assert solution.initial_bars - solution.final_bars == 1
    assert solution.bars_saved == 1
    assert solution.final_patterns == exact_proposal.patterns
    assert solution.final_column_values == exact_proposal.column_values


def test_publication_reverifies_final_plan_from_persisted_patterns_not_from_the_record(
    refined_unchanged,
) -> None:
    """A tampered record whose embedded verification still looks coherent."""
    instance, result = refined_unchanged
    exact = CompleteIntegerMaster(instance).solve()
    usage = [
        (pattern, value)
        for pattern, value in zip(
            iter_maximal_patterns(instance), exact.column_values, strict=True
        )
        if value > 0
    ]
    forged = QualityAgentProposal(
        tuple(pattern for pattern, _ in usage), tuple(value for _, value in usage)
    )
    assert sum(forged.column_values) != result.final_bars

    tampered = replace(
        result, final_patterns=forged.patterns, final_column_values=forged.column_values
    )

    record = publish_refinement_result(instance, tampered)

    assert record.instance_id == GAP_INSTANCE_ID
    assert record.outcome == OUTCOME_FAILURE
    assert record.solution is None
    assert record.failure_reason == FAILURE_FINAL_VERIFICATION
    assert f"counted {sum(forged.column_values)} stock bars" in str(record.failure_message)


def test_a_baseline_that_fails_independent_verification_is_never_published(
    refined_unchanged,
) -> None:
    instance, result = refined_unchanged
    tampered = replace(
        result,
        initial_patterns=((2,) * len(instance.demands),),
        initial_column_values=(1,),
    )

    record = publish_refinement_result(instance, tampered)

    assert record.outcome == OUTCOME_FAILURE
    assert record.solution is None
    assert record.failure_reason == FAILURE_BASELINE_VERIFICATION
    assert "exceeds stock capacity" in str(record.failure_message)


def test_attempt_preserves_a_non_converged_classical_start(monkeypatch) -> None:
    class LimitedColumnGeneration(ColumnGeneration):
        def __init__(self, instance, instance_id=None, **kwargs):
            super().__init__(instance, instance_id=instance_id, max_iterations=1)

    monkeypatch.setattr(
        "neural_cutting_stock.learning.neural_qc.ColumnGeneration", LimitedColumnGeneration
    )
    multi_iteration_instance = CuttingStockInstance(100.0, 0.0, (30.0, 40.0), (6, 3))

    record = attempt_quality_refinement(
        multi_iteration_instance, "limited-instance", EchoAgent(), budget=NeuralQCBudget(3)
    )

    assert record.instance_id == "limited-instance"
    assert record.outcome == OUTCOME_FAILURE
    assert record.solution is None
    assert record.failure_reason == FAILURE_REFINEMENT_ERROR
    assert "did not converge" in str(record.failure_message)


def test_attempt_preserves_a_crashing_agent(classical) -> None:
    instance, _ = classical

    record = attempt_quality_refinement(
        instance, GAP_INSTANCE_ID, CrashingAgent(), budget=NeuralQCBudget(3)
    )

    assert record.outcome == OUTCOME_FAILURE
    assert record.solution is None
    assert record.failure_reason == FAILURE_REFINEMENT_ERROR
    assert "RuntimeError: boom" in str(record.failure_message)


def test_rejected_invalid_plans_still_publish_an_honest_equal_solution(classical) -> None:
    class InvalidAgent:
        """Always propose a plan that violates stock capacity."""

        def propose(self, observation) -> QualityAgentProposal:
            return QualityAgentProposal(((2,) * len(observation.demands),), (1,))

    instance, _ = classical

    record = attempt_quality_refinement(
        instance, GAP_INSTANCE_ID, InvalidAgent(), budget=NeuralQCBudget(5, stall_patience=2)
    )

    assert record.outcome == OUTCOME_SOLUTION
    solution = record.solution
    assert solution is not None
    assert solution.status == PUBLICATION_STATUS_EQUAL
    assert solution.final_verification.feasible


def test_degraded_is_a_first_class_status_never_relabeled() -> None:
    """The vocabulary keeps degraded reachable for any future publication path."""
    assert PUBLICATION_STATUS_DEGRADED == "degraded"
    assert classify_publication_status(12, 13) == PUBLICATION_STATUS_DEGRADED


def test_attempt_supports_declared_multi_format_instances() -> None:
    multi = MultiFormatCuttingStockInstance((50.0, 100.0), 0.0, [30.0, 40.0], [6, 3])

    record = attempt_quality_refinement(
        multi, "multi-format-publication", EchoAgent(), budget=NeuralQCBudget(3)
    )

    assert record.outcome == OUTCOME_SOLUTION
    solution = record.solution
    assert solution is not None
    assert solution.status == PUBLICATION_STATUS_EQUAL
    assert solution.final_verification.feasible


def test_attempt_raises_on_caller_misuse_instead_of_masking_it(classical) -> None:
    instance, _ = classical
    budget = NeuralQCBudget(3)

    for bad_id in (None, "", "   ", 7):
        with pytest.raises(ValueError, match="instance_id"):
            attempt_quality_refinement(instance, bad_id, EchoAgent(), budget=budget)
    for agent in (None, 42, object()):
        with pytest.raises(ValueError, match="propose"):
            attempt_quality_refinement(instance, GAP_INSTANCE_ID, agent, budget=budget)
    for bad_budget in (None, (3, 1)):
        with pytest.raises(ValueError, match="budget"):
            attempt_quality_refinement(instance, GAP_INSTANCE_ID, EchoAgent(), budget=bad_budget)
    with pytest.raises(ValueError, match="verification_tolerance"):
        attempt_quality_refinement(
            instance, GAP_INSTANCE_ID, EchoAgent(), budget=budget, verification_tolerance=-1e-9
        )


def test_attempt_records_validate_their_own_consistency() -> None:
    with pytest.raises(ValueError, match="unknown attempt outcome"):
        NeuralQCAttemptRecord(instance_id="i", outcome="dropped")
    with pytest.raises(ValueError, match="carries its solution"):
        NeuralQCAttemptRecord(instance_id="i", outcome=OUTCOME_SOLUTION)
    with pytest.raises(ValueError, match="failure_reason"):
        NeuralQCAttemptRecord(instance_id="i", outcome=OUTCOME_FAILURE, failure_message="x")
    with pytest.raises(ValueError, match="instance_id must be non-empty"):
        NeuralQCAttemptRecord(
            instance_id=" ", outcome=OUTCOME_FAILURE, failure_reason=FAILURE_REFINEMENT_ERROR
        )
