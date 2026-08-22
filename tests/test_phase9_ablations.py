"""Mandatory equal-budget ablations: greedy completion and random search."""

import pytest

from neural_cutting_stock.benchmarks import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    build_quality_partition_plan,
    materialize_partition_instances,
    phase8_family_specs,
)
from neural_cutting_stock.learning import (
    EXCLUSION_NO_PUBLISHED_SOLUTION,
    GREEDY_ABLATION_IDENTIFIER,
    NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION,
    OUTCOME_SOLUTION,
    QUALITY_ABLATION_EVAL_SCHEMA_VERSION,
    RANDOM_SEARCH_ABLATION_IDENTIFIER,
    GreedyQualityAgent,
    NeuralQCBudget,
    QualityAgentInput,
    QualityAgentProposal,
    RandomSearchQualityAgent,
    evaluate_quality_ablations_on_partition,
    evaluate_quality_agent_on_partition,
    summarize_ablation_deltas,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import (
    ColumnGeneration,
    CompleteIntegerMaster,
    iter_maximal_patterns,
    verify_plan,
)

FAMILY_T3 = "structured-tight-divisibility-t3-v1"
FAMILY_T4 = "structured-tight-divisibility-t4-v1"
BUDGET = NeuralQCBudget(3)

TOY_INSTANCE = CuttingStockInstance(120.0, 0.0, (30.0, 40.0), (5, 4))
TOY_INSTANCE_ID = "toy-ablation"
WIDE_INSTANCE = CuttingStockInstance(200.0, 0.0, (25.0, 35.0, 45.0), (9, 7, 5))
WIDE_INSTANCE_ID = "wide-ablation"


class EchoAgent:
    """Never improve anything: re-propose the incumbent unchanged."""

    def propose(self, observation) -> QualityAgentProposal:
        return QualityAgentProposal(
            observation.solution_patterns, observation.solution_column_values
        )


class CrashingAgent:
    """Simulate an agent whose inference crashes mid-campaign."""

    def propose(self, observation) -> QualityAgentProposal:
        raise RuntimeError("boom")


class ExactChoiceAgent:
    """Propose the certified exact optimum recorded for each instance."""

    def __init__(self, proposals: dict[str, QualityAgentProposal]) -> None:
        self.proposals = proposals

    def propose(self, observation) -> QualityAgentProposal:
        return self.proposals[observation.instance_id]


def _spec_of(label: str):
    return {spec.family_label: spec for spec in phase8_family_specs()}[label]


def _fabricated_report() -> dict:
    """A minimal family-margins-v1 payload retaining two small real families."""

    specs = (_spec_of(FAMILY_T3), _spec_of(FAMILY_T4))
    instances = [
        {
            "family_label": spec.family_label,
            "seed": generator.seed,
            "instance_id": generator.instance_id,
            "gap_available": True,
            "gap_bars": 1,
        }
        for spec in specs
        for generator in spec.generators
    ]
    return {
        "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
        "significant_positive_share": 0.5,
        "reference_method_limits": "maximal_patterns:test-limits",
        "families": [
            {
                "family_label": spec.family_label,
                "configuration": spec.configuration,
                "retained": True,
            }
            for spec in specs
        ],
        "instances": instances,
    }


def _observation_of(instance, instance_id: str) -> QualityAgentInput:
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
def manifest():
    return build_quality_partition_plan(_fabricated_report())


@pytest.fixture(scope="module")
def comparison(manifest):
    return evaluate_quality_ablations_on_partition(
        manifest,
        "validation",
        {"random": RandomSearchQualityAgent(17), "greedy": GreedyQualityAgent()},
        budget=BUDGET,
    )


@pytest.fixture(scope="module")
def exact_proposals(manifest):
    proposals = {}
    for instance_id, instance in materialize_partition_instances(
        manifest, "validation"
    ).items():
        exact = CompleteIntegerMaster(instance).solve()
        assert exact.status == 0 and exact.objective_value is not None
        usage = [
            (pattern, value)
            for pattern, value in zip(
                iter_maximal_patterns(instance), exact.column_values, strict=True
            )
            if value > 0
        ]
        proposals[instance_id] = QualityAgentProposal(
            tuple(pattern for pattern, _ in usage), tuple(value for _, value in usage)
        )
    return proposals


def test_schema_versions_and_identifiers_are_stable() -> None:
    assert QUALITY_ABLATION_EVAL_SCHEMA_VERSION == "neural-qc-ablation-eval-v1"
    assert GREEDY_ABLATION_IDENTIFIER == "greedy-basis-completion-v1"
    assert RANDOM_SEARCH_ABLATION_IDENTIFIER == "random-search-uniform-counts-v1"


def test_greedy_agent_is_deterministic_verified_and_inside_the_basis() -> None:
    observation = _observation_of(TOY_INSTANCE, TOY_INSTANCE_ID)
    basis = set(iter_maximal_patterns(TOY_INSTANCE))

    first = GreedyQualityAgent().propose(observation)
    second = GreedyQualityAgent().propose(observation)

    assert first == second
    assert first.patterns and set(first.patterns) <= basis
    verification = verify_plan(TOY_INSTANCE, first.patterns, first.column_values)
    assert verification.feasible


def test_random_search_is_reproducible_for_a_fixed_seed_and_stays_in_the_space() -> None:
    observation = _observation_of(TOY_INSTANCE, TOY_INSTANCE_ID)
    basis = set(iter_maximal_patterns(TOY_INSTANCE))

    forward = RandomSearchQualityAgent(seed=5)
    backward = RandomSearchQualityAgent(seed=5)
    forward_sequence = [forward.propose(observation) for _ in range(3)]
    backward_sequence = [backward.propose(observation) for _ in range(3)]

    assert forward_sequence == backward_sequence
    for proposal in forward_sequence:
        assert proposal.patterns and set(proposal.patterns) <= basis
        assert verify_plan(TOY_INSTANCE, proposal.patterns, proposal.column_values).feasible


def test_a_different_seed_explores_a_different_count_vector() -> None:
    observation = _observation_of(WIDE_INSTANCE, WIDE_INSTANCE_ID)

    left = RandomSearchQualityAgent(seed=1).propose(observation)
    right = RandomSearchQualityAgent(seed=2).propose(observation)

    left_counts = dict(zip(left.patterns, left.column_values, strict=True))
    right_counts = dict(zip(right.patterns, right.column_values, strict=True))
    assert any(
        left_counts.get(pattern, 0) != right_counts.get(pattern, 0)
        for pattern in iter_maximal_patterns(WIDE_INSTANCE)
    )


def test_random_streams_do_not_depend_on_the_evaluation_order() -> None:
    toy = _observation_of(TOY_INSTANCE, TOY_INSTANCE_ID)
    wide = _observation_of(WIDE_INSTANCE, WIDE_INSTANCE_ID)

    forward = RandomSearchQualityAgent(seed=9)
    backward = RandomSearchQualityAgent(seed=9)
    toy_forward, wide_forward = forward.propose(toy), forward.propose(wide)
    wide_backward, toy_backward = backward.propose(wide), backward.propose(toy)

    assert toy_forward == toy_backward
    assert wide_forward == wide_backward


def test_comparison_shares_one_budget_and_matches_single_agent_evaluations(
    manifest, comparison
) -> None:
    names = comparison["agent_names"]
    assert names == ["greedy", "random"]
    assert comparison["schema_version"] == QUALITY_ABLATION_EVAL_SCHEMA_VERSION
    assert comparison["plan_id"] == manifest["plan_id"]
    assert comparison["partition"] == "validation"
    assert comparison["budget"] == {"max_steps": 3, "stall_patience": 1}
    assert sorted(comparison["evaluations"]) == names
    agents_by_name = {
        "greedy": GreedyQualityAgent(),
        "random": RandomSearchQualityAgent(17),
    }
    for name in names:
        single = evaluate_quality_agent_on_partition(
            manifest, "validation", agents_by_name[name], budget=BUDGET
        )
        assert comparison["evaluations"][name] == single
        assert single["budget"] == comparison["budget"]


def test_per_instance_matrix_covers_every_instance_and_agent(manifest, comparison) -> None:
    assignments = {
        cell["instance_id"]: cell
        for cell in manifest["assignments"]
        if cell["partition"] == "validation"
    }

    cells = comparison["per_instance"]
    assert [cell["instance_id"] for cell in cells] == sorted(assignments)
    for cell in cells:
        assignment = assignments[cell["instance_id"]]
        assert cell["family_label"] == assignment["family_label"]
        assert isinstance(cell["number_of_piece_types"], int)
        assert set(cell["outcomes"]) == set(comparison["agent_names"])
        assert set(cell["bars_saved"]) == set(comparison["agent_names"])
        for name in comparison["agent_names"]:
            sub_entry = next(
                entry
                for entry in comparison["evaluations"][name]["instances"]
                if entry["instance_id"] == cell["instance_id"]
            )
            assert cell["outcomes"][name] == sub_entry["outcome"]
            assert cell["bars_saved"][name] == sub_entry["bars_saved"]
        if all(outcome == OUTCOME_SOLUTION for outcome in cell["outcomes"].values()):
            assert all(saved >= 0 for saved in cell["bars_saved"].values())


def test_invalid_inputs_are_rejected(manifest) -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        evaluate_quality_ablations_on_partition(manifest, "validation", {}, budget=BUDGET)
    with pytest.raises(ValueError, match="propose"):
        evaluate_quality_ablations_on_partition(
            manifest, "validation", {"broken": object()}, budget=BUDGET
        )
    with pytest.raises(ValueError, match="non-empty string name"):
        evaluate_quality_ablations_on_partition(
            manifest, "validation", {" ": GreedyQualityAgent()}, budget=BUDGET
        )
    with pytest.raises(ValueError, match="seed"):
        RandomSearchQualityAgent(seed="7")
    with pytest.raises(ValueError, match="unknown partition"):
        evaluate_quality_ablations_on_partition(
            manifest, "holdout", {"greedy": GreedyQualityAgent()}, budget=BUDGET
        )


def test_delta_summary_pairs_published_solutions_against_the_reference(
    manifest, exact_proposals
) -> None:
    report = evaluate_quality_ablations_on_partition(
        manifest,
        "validation",
        {"learned": ExactChoiceAgent(exact_proposals), "echo": EchoAgent()},
        budget=BUDGET,
    )

    deltas = summarize_ablation_deltas(report, reference_agent="learned")

    assert deltas["reference_agent"] == "learned"
    echo = deltas["comparisons"]["echo"]
    reference_total = report["evaluations"]["learned"]["overall"]["total_bars_saved"]
    assert echo["paired_instance_count"] == 2
    assert echo["excluded_instances"] == {}
    assert echo["delta_total_bars_saved"] == -reference_total
    assert echo["delta_mean_bars_saved"] == -reference_total / 2
    assert echo["instances_where_reference_saves_more"] == 2
    assert echo["equal_instances"] == 0
    assert echo["instances_where_candidate_saves_more"] == 0


def test_delta_summary_preserves_excluded_pairs_instead_of_dropping_them(
    manifest, exact_proposals
) -> None:
    report = evaluate_quality_ablations_on_partition(
        manifest,
        "validation",
        {"learned": ExactChoiceAgent(exact_proposals), "crashing": CrashingAgent()},
        budget=BUDGET,
    )

    deltas = summarize_ablation_deltas(report, reference_agent="learned")
    crashing = deltas["comparisons"]["crashing"]

    assert crashing["paired_instance_count"] == 0
    assert crashing["excluded_instances"] == {
        cell["instance_id"]: EXCLUSION_NO_PUBLISHED_SOLUTION
        for cell in report["per_instance"]
    }
    assert crashing["delta_total_bars_saved"] == 0
    assert crashing["delta_mean_bars_saved"] is None


def test_delta_summary_validates_its_inputs(manifest) -> None:
    report = evaluate_quality_ablations_on_partition(
        manifest, "validation", {"greedy": GreedyQualityAgent()}, budget=BUDGET
    )

    with pytest.raises(ValueError, match="ablation evaluation payload"):
        summarize_ablation_deltas(
            {"schema_version": NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION},
            reference_agent="greedy",
        )
    with pytest.raises(ValueError, match="unknown reference agent"):
        summarize_ablation_deltas(report, reference_agent="missing")
