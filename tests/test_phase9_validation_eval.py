"""Offline validation-partition evaluation of the trained quality policy."""

import hashlib

import pytest

from neural_cutting_stock.benchmarks import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    build_quality_partition_plan,
    materialize_partition_instances,
    phase8_family_specs,
)
from neural_cutting_stock.learning import (
    NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION,
    OUTCOME_FAILURE,
    OUTCOME_SOLUTION,
    PUBLICATION_STATUS_EQUAL,
    PUBLICATION_STATUS_IMPROVED,
    NeuralQCBudget,
    QualityAgentProposal,
    QualityPolicyNetwork,
    TrainingCurvePoint,
    TrainingCurves,
    checkpoint_sha256,
    evaluate_quality_agent_on_partition,
    quality_agent_from_checkpoint,
    save_checkpoint,
)
from neural_cutting_stock.solver import CompleteIntegerMaster, iter_maximal_patterns

FAMILY_T3 = "structured-tight-divisibility-t3-v1"
FAMILY_T4 = "structured-tight-divisibility-t4-v1"
BUDGET = NeuralQCBudget(3)


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


@pytest.fixture(scope="module")
def manifest():
    return build_quality_partition_plan(_fabricated_report())


@pytest.fixture(scope="module")
def validation(manifest):
    return materialize_partition_instances(manifest, "validation")


@pytest.fixture(scope="module")
def exact_proposals(validation):
    proposals = {}
    for instance_id, instance in validation.items():
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


@pytest.fixture(scope="module")
def echo_report(manifest):
    return evaluate_quality_agent_on_partition(
        manifest, "validation", EchoAgent(), budget=BUDGET
    )


@pytest.fixture(scope="module")
def exact_report(manifest, exact_proposals):
    return evaluate_quality_agent_on_partition(
        manifest, "validation", ExactChoiceAgent(exact_proposals), budget=BUDGET
    )


def test_schema_version_is_stable() -> None:
    assert NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION == "neural-qc-validation-eval-v1"


def test_echo_evaluation_covers_the_partition_with_honest_equal_solutions(
    manifest, echo_report
) -> None:
    assert echo_report["schema_version"] == NEURAL_QC_VALIDATION_EVAL_SCHEMA_VERSION
    assert echo_report["plan_id"] == manifest["plan_id"]
    assert echo_report["partition"] == "validation"
    assert echo_report["budget"] == {"max_steps": 3, "stall_patience": 1}

    counts = echo_report["counts"]
    assert counts == {
        "instance_count": 2,
        "published_solution_count": 2,
        "preserved_failure_count": 0,
    }
    overall = echo_report["overall"]
    assert overall["key"] is None
    assert overall["improved_count"] == 0
    assert overall["equal_count"] == 2
    assert overall["total_bars_saved"] == 0
    assert overall["mean_bars_saved"] == 0.0
    assert overall["failure_reasons"] == {}

    assert [group["key"] for group in echo_report["by_family"]] == [FAMILY_T3, FAMILY_T4]
    assert [group["key"] for group in echo_report["by_size"]] == [3, 4]
    for group in (*echo_report["by_family"], *echo_report["by_size"]):
        assert group["instance_count"] == 1
        assert group["published_solution_count"] == 1
        assert group["mean_bars_saved"] == 0.0


def test_entries_carry_the_frozen_assignment_metadata(manifest, echo_report) -> None:
    assignments = {
        cell["instance_id"]: cell
        for cell in manifest["assignments"]
        if cell["partition"] == "validation"
    }

    assert len(echo_report["instances"]) == len(assignments)
    for entry in echo_report["instances"]:
        assignment = assignments[entry["instance_id"]]
        assert entry["family_label"] == assignment["family_label"]
        assert entry["seed"] == assignment["seed"]
        assert entry["outcome"] == OUTCOME_SOLUTION
        assert entry["status"] == PUBLICATION_STATUS_EQUAL
        assert entry["initial_bars"] == entry["final_bars"]
        assert entry["bars_saved"] == 0
        assert entry["failure_reason"] is None
        assert entry["failure_message"] is None
        assert isinstance(entry["number_of_piece_types"], int)
        assert entry["total_demand"] > 0


def test_exact_choice_measures_verified_mean_bar_gains(exact_report) -> None:
    entries = exact_report["instances"]

    assert all(entry["outcome"] == OUTCOME_SOLUTION for entry in entries)
    assert all(entry["status"] == PUBLICATION_STATUS_IMPROVED for entry in entries)
    assert all(entry["bars_saved"] >= 1 for entry in entries)
    assert all(entry["final_bars"] < entry["initial_bars"] for entry in entries)

    overall = exact_report["overall"]
    expected_total = sum(entry["bars_saved"] for entry in entries)
    assert overall["total_bars_saved"] == expected_total
    assert overall["mean_bars_saved"] == expected_total / len(entries)
    assert overall["improved_count"] == len(entries)

    for group in exact_report["by_family"]:
        family_entries = [e for e in entries if e["family_label"] == group["key"]]
        assert group["total_bars_saved"] == sum(e["bars_saved"] for e in family_entries)
        assert group["mean_bars_saved"] == group["total_bars_saved"] / len(family_entries)
    for group in exact_report["by_size"]:
        size_entries = [e for e in entries if e["number_of_piece_types"] == group["key"]]
        assert group["total_bars_saved"] == sum(e["bars_saved"] for e in size_entries)


def test_preserved_failures_stay_visible_and_out_of_the_means(manifest) -> None:
    report = evaluate_quality_agent_on_partition(
        manifest, "validation", CrashingAgent(), budget=BUDGET
    )

    assert report["counts"] == {
        "instance_count": 2,
        "published_solution_count": 0,
        "preserved_failure_count": 2,
    }
    overall = report["overall"]
    assert overall["failure_reasons"] == {"refinement_error": 2}
    assert overall["improved_count"] == 0
    assert overall["equal_count"] == 0
    assert overall["total_bars_saved"] == 0
    assert overall["mean_bars_saved"] is None
    for entry in report["instances"]:
        assert entry["outcome"] == OUTCOME_FAILURE
        assert entry["initial_bars"] is None
        assert entry["final_bars"] is None
        assert entry["bars_saved"] is None
        assert entry["status"] is None
        assert "RuntimeError: boom" in entry["failure_message"]


def test_evaluation_is_deterministic(manifest, echo_report) -> None:
    replayed = evaluate_quality_agent_on_partition(
        manifest, "validation", EchoAgent(), budget=BUDGET
    )

    assert replayed == echo_report


def test_unknown_partitions_and_tampered_manifests_are_rejected(manifest) -> None:
    with pytest.raises(ValueError, match="unknown partition"):
        evaluate_quality_agent_on_partition(
            manifest, "holdout", EchoAgent(), budget=BUDGET
        )
    tampered = {**manifest, "plan_id": "0" * 64}
    with pytest.raises(ValueError, match="plan_id"):
        evaluate_quality_agent_on_partition(tampered, "train", EchoAgent(), budget=BUDGET)


def test_checkpoint_agent_rebuilt_from_provenance_evaluates_the_partition(
    manifest, tmp_path
) -> None:
    checkpoint = tmp_path / "checkpoints" / "quality-policy.pt"
    save_checkpoint(
        checkpoint,
        module=QualityPolicyNetwork(28, 8),
        seed=3,
        config={"hidden_width": 8},
        curves=TrainingCurves((TrainingCurvePoint(0, {"policy_loss": 0.0}),)),
    )

    agent = quality_agent_from_checkpoint(checkpoint)
    report = evaluate_quality_agent_on_partition(manifest, "validation", agent, budget=BUDGET)

    assert report["counts"]["preserved_failure_count"] == 0
    assert report["counts"]["published_solution_count"] == 2
    for entry in report["instances"]:
        assert entry["outcome"] == OUTCOME_SOLUTION
        assert entry["final_bars"] <= entry["initial_bars"]
    assert checkpoint_sha256(checkpoint) == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_checkpoint_loader_rejects_incomplete_or_foreign_artifacts(tmp_path) -> None:
    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="unreadable checkpoint"):
        quality_agent_from_checkpoint(foreign)

    missing_hidden = tmp_path / "missing-hidden-width.pt"
    save_checkpoint(
        missing_hidden,
        module=QualityPolicyNetwork(28, 8),
        seed=3,
        config={"epochs": 1},
        curves=TrainingCurves((TrainingCurvePoint(0, {"policy_loss": 0.0}),)),
    )
    with pytest.raises(ValueError, match="hidden_width"):
        quality_agent_from_checkpoint(missing_hidden)

    bad_hidden = tmp_path / "bad-hidden-width.pt"
    save_checkpoint(
        bad_hidden,
        module=QualityPolicyNetwork(28, 8),
        seed=3,
        config={"hidden_width": 0},
        curves=TrainingCurves((TrainingCurvePoint(0, {"policy_loss": 0.0}),)),
    )
    with pytest.raises(ValueError, match="hidden_width"):
        quality_agent_from_checkpoint(bad_hidden)
