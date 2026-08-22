import pytest

from neural_cutting_stock.benchmarks import (
    EnvironmentMetadata,
    ExactReferenceMethod,
    ExactReferenceRecord,
    ExactReferenceStatus,
    build_milp_exact_reference,
    solve_milp_exact_reference,
)
from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver import CompleteMasterResult, MaximalPatternLimits

_ENVIRONMENT = EnvironmentMetadata("abc123", "3.11.9", "numpy=1.26 scipy=1.17", "cpu/os")


def _outcome(**changes: object) -> CompleteMasterResult:
    values: dict[str, object] = {
        "status": 0,
        "objective_value": 3.0,
        "column_values": (1, 2),
        "certified_lower_bound": 3.0,
        "number_of_patterns": 7,
        "message": "Optimization terminated successfully",
    }
    values.update(changes)
    return CompleteMasterResult(**values)


def test_optimal_instance_yields_persisted_optimal_record() -> None:
    instance = CuttingStockInstance(10, 1, [4], [5])

    record = solve_milp_exact_reference(
        "instance-kerf-1",
        instance,
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
    )[1]

    assert record.schema_version == "exact-reference-v1"
    assert record.reference_method is ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS
    assert record.status is ExactReferenceStatus.OPTIMAL
    assert record.integer_optimum_bars == 3
    assert record.certified_lower_bound_bars is not None
    assert record.certified_lower_bound_bars <= record.integer_optimum_bars
    assert record.method_limits.startswith("maximal_patterns:max_search_space_size=")
    assert record.error_message is None


def test_optimal_record_round_trips_through_persisted_representation() -> None:
    instance = CuttingStockInstance(6, 1, [2, 3], [3, 2])

    record = solve_milp_exact_reference(
        "instance-1",
        instance,
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
    )[1]

    assert ExactReferenceRecord.from_dict(record.to_dict()) == record


def test_enumeration_guard_yields_failed_record_without_numerical_claims() -> None:
    instance = CuttingStockInstance(60, 0, [7, 11, 13], [30, 30, 30])

    record = solve_milp_exact_reference(
        "instance-huge",
        instance,
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        limits=MaximalPatternLimits(max_search_space_size=10),
    )[1]

    assert record.status is ExactReferenceStatus.FAILED
    assert record.reference_method is ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS
    assert record.integer_optimum_bars is None
    assert record.certified_lower_bound_bars is None
    assert record.error_message is not None
    assert "max_search_space_size=10" in record.error_message


def test_method_limits_derive_from_effective_guards() -> None:
    instance = CuttingStockInstance(10, 0, [2], [1])

    record = solve_milp_exact_reference(
        "instance-1",
        instance,
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        limits=MaximalPatternLimits(max_search_space_size=500, max_patterns=40),
    )[1]

    assert record.method_limits == "maximal_patterns:max_search_space_size=500,max_patterns=40"


def test_build_maps_non_integral_objective_to_failed_record() -> None:
    record = build_milp_exact_reference(
        "instance-1",
        _outcome(objective_value=2.5),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.FAILED
    assert record.integer_optimum_bars is None
    assert record.certified_lower_bound_bars is None
    assert "integrality_tolerance" in record.error_message


def test_build_requires_lower_bound_for_an_optimal_proof() -> None:
    record = build_milp_exact_reference(
        "instance-1",
        _outcome(certified_lower_bound=None),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.FAILED
    assert "certified lower bound" in record.error_message


def test_build_clamps_dual_bound_noise_to_the_proven_optimum() -> None:
    record = build_milp_exact_reference(
        "instance-1",
        _outcome(objective_value=3.0, certified_lower_bound=3.0 + 5e-10),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.OPTIMAL
    assert record.integer_optimum_bars == 3
    assert record.certified_lower_bound_bars == 3.0


def test_build_maps_solver_limit_to_lower_bound_only_record() -> None:
    record = build_milp_exact_reference(
        "instance-1",
        _outcome(
            status=1,
            objective_value=None,
            column_values=(),
            certified_lower_bound=2.5,
            message="time limit reached",
        ),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.LOWER_BOUND_ONLY
    assert record.integer_optimum_bars is None
    assert record.certified_lower_bound_bars == 2.5
    assert record.error_message is None


def test_build_maps_solver_failure_to_failed_record_with_diagnosis() -> None:
    record = build_milp_exact_reference(
        "instance-1",
        _outcome(
            status=2,
            objective_value=None,
            column_values=(),
            certified_lower_bound=None,
            message="infeasible",
        ),
        environment=_ENVIRONMENT,
        integrality_tolerance=1e-9,
        feasibility_tolerance=1e-9,
        method_limits="limits",
    )

    assert record.status is ExactReferenceStatus.FAILED
    assert record.error_message == "infeasible"
    assert record.integer_optimum_bars is None
    assert record.certified_lower_bound_bars is None


def test_build_rejects_empty_instance_id() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        build_milp_exact_reference(
            " ",
            _outcome(),
            environment=_ENVIRONMENT,
            integrality_tolerance=1e-9,
            feasibility_tolerance=1e-9,
            method_limits="limits",
        )
