from dataclasses import replace

import pytest

from neural_cutting_stock.benchmarks import (
    EnvironmentMetadata,
    ExactReferenceMethod,
    ExactReferenceRecord,
    ExactReferenceStatus,
    build_milp_exact_reference,
    verify_milp_exact_reference,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import CompleteIntegerMaster, CompleteMasterResult

_ENVIRONMENT = EnvironmentMetadata("abc123", "3.11.9", "numpy=1.26 scipy=1.17", "cpu/os")


def _reference(
    instance: CuttingStockInstance,
) -> tuple[CompleteMasterResult, ExactReferenceRecord]:
    outcome = CompleteIntegerMaster(instance).solve()
    record = build_milp_exact_reference(
        "instance-1",
        outcome,
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="maximal_patterns:max_search_space_size=10000000,max_patterns=100000",
    )
    return outcome, record


def test_valid_reference_passes_every_independent_check() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, record = _reference(instance)

    verification = verify_milp_exact_reference(
        "instance-1",
        instance,
        outcome,
        record,
        cross_check_with_enumeration=True,
    )

    assert verification.passed
    assert verification.errors == ()
    assert verification.plan_verification is not None
    assert verification.plan_verification.feasible
    assert verification.plan_verification.number_of_stock_bars == record.integer_optimum_bars
    assert verification.lp_bound_bars is not None
    assert verification.lp_bound_bars <= record.integer_optimum_bars + 1e-9
    assert verification.exhaustive_optimum_bars == record.integer_optimum_bars


def test_cross_check_left_out_reports_no_enumeration_optimum() -> None:
    instance = CuttingStockInstance(7, 0, [2, 3, 4], [2, 1, 2])
    outcome, record = _reference(instance)

    verification = verify_milp_exact_reference("instance-1", instance, outcome, record)

    assert verification.passed
    assert verification.exhaustive_optimum_bars is None


def test_inflated_optimum_is_caught_by_plan_and_cross_check() -> None:
    instance = CuttingStockInstance(10, 0, [2, 3], [5, 5])
    outcome, record = _reference(instance)
    tampered = replace(record, integer_optimum_bars=record.integer_optimum_bars + 1)

    verification = verify_milp_exact_reference(
        "instance-1",
        instance,
        outcome,
        tampered,
        cross_check_with_enumeration=True,
    )

    assert not verification.passed
    joined = "\n".join(verification.errors)
    assert "plan uses" in joined
    assert "disagrees with the claimed optimum" in joined
    assert "enumeration finds" in joined


def test_claim_below_the_lp_bound_is_caught_by_the_consistency_check() -> None:
    instance = CuttingStockInstance(10, 0, [2, 3], [5, 5])
    outcome, _ = _reference(instance)
    lying_record = build_milp_exact_reference(
        "instance-1",
        replace(outcome, objective_value=2.0, certified_lower_bound=2.0),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    verification = verify_milp_exact_reference(
        "instance-1",
        instance,
        outcome,
        lying_record,
        cross_check_with_enumeration=True,
    )

    assert not verification.passed
    joined = "\n".join(verification.errors)
    assert "LP relaxation bound" in joined
    assert "exceeds the integer optimum 2" in joined
    assert "plan uses" in joined
    assert "enumeration finds" in joined


def test_pattern_count_mismatch_stops_numeric_checks() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, record = _reference(instance)
    tampered = replace(outcome, number_of_patterns=outcome.number_of_patterns + 999)

    verification = verify_milp_exact_reference("instance-1", instance, tampered, record)

    assert not verification.passed
    assert verification.plan_verification is None
    assert verification.lp_bound_bars is None
    assert len(verification.errors) == 1
    assert "patterns while the outcome reports" in verification.errors[0]


def test_non_proven_outcome_is_reported_without_a_plan() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, record = _reference(instance)
    unproven = replace(
        outcome,
        status=1,
        objective_value=None,
        column_values=(),
        certified_lower_bound=2.5,
    )

    verification = verify_milp_exact_reference("instance-1", instance, unproven, record)

    assert not verification.passed
    assert verification.plan_verification is None
    assert verification.lp_bound_bars is None
    assert verification.errors == ("outcome is not a proven optimum",)


def test_verifier_refuses_empty_instance_id() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, record = _reference(instance)

    with pytest.raises(ValueError, match="instance_id"):
        verify_milp_exact_reference(" ", instance, outcome, record)


def test_verifier_refuses_a_record_from_another_instance() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, record = _reference(instance)

    with pytest.raises(ValueError, match="does not belong"):
        verify_milp_exact_reference("other-instance", instance, outcome, record)


def test_verifier_refuses_lower_bound_only_references() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    record = build_milp_exact_reference(
        "instance-1",
        CompleteMasterResult(1, None, (), 2.5, 5, "time limit reached"),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.LOWER_BOUND_ONLY

    with pytest.raises(ValueError, match="optimal reference"):
        verify_milp_exact_reference(
            "instance-1",
            instance,
            CompleteIntegerMaster(instance).solve(),
            record,
        )


def test_verifier_refuses_unsupported_reference_methods() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])
    outcome, _ = _reference(instance)
    record = ExactReferenceRecord(
        instance_id="instance-1",
        reference_method=ExactReferenceMethod.EXHAUSTIVE_PATTERN_ENUMERATION,
        status=ExactReferenceStatus.OPTIMAL,
        method_limits="limits",
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        integer_optimum_bars=4,
        certified_lower_bound_bars=4.0,
    )

    with pytest.raises(ValueError, match="MILP-on-enumerated-patterns"):
        verify_milp_exact_reference("instance-1", instance, outcome, record)
