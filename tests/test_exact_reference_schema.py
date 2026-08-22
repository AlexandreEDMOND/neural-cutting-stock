import pytest

from neural_cutting_stock.benchmarks import (
    EXACT_REFERENCE_SCHEMA_VERSION,
    EnvironmentMetadata,
    ExactReferenceMethod,
    ExactReferenceRecord,
    ExactReferenceStatus,
)


def _record(**changes: object) -> ExactReferenceRecord:
    values: dict[str, object] = {
        "instance_id": "instance-1",
        "reference_method": ExactReferenceMethod.EXHAUSTIVE_PATTERN_ENUMERATION,
        "status": ExactReferenceStatus.OPTIMAL,
        "method_limits": "bounded_types_and_demands_v1",
        "environment": EnvironmentMetadata("abc123", "3.11.9", "numpy=1.26 scipy=1.11", "cpu/os"),
        "integrality_tolerance": 1e-9,
        "feasibility_tolerance": 1e-9,
        "integer_optimum_bars": 4,
        "certified_lower_bound_bars": 4.0,
    }
    values.update(changes)
    return ExactReferenceRecord(**values)


def test_record_has_versioned_flat_json_ready_schema() -> None:
    record = _record()

    output = record.to_dict()

    assert output["schema_version"] == EXACT_REFERENCE_SCHEMA_VERSION
    assert output["instance_id"] == "instance-1"
    assert output["reference_method"] == "exhaustive_pattern_enumeration"
    assert output["status"] == "optimal"
    assert output["method_limits"] == "bounded_types_and_demands_v1"
    assert output["integer_optimum_bars"] == 4
    assert output["certified_lower_bound_bars"] == 4.0
    assert output["integrality_tolerance"] == 1e-9
    assert output["feasibility_tolerance"] == 1e-9
    assert output["code_commit"] == "abc123"
    assert output["python_version"] == "3.11.9"
    assert output["dependency_versions"] == "numpy=1.26 scipy=1.11"
    assert output["hardware_id"] == "cpu/os"
    assert "environment" not in output


def test_record_round_trips_through_persisted_representation() -> None:
    optimal = _record()
    bound_only = _record(
        status=ExactReferenceStatus.LOWER_BOUND_ONLY,
        reference_method=ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS,
        integer_optimum_bars=None,
        certified_lower_bound_bars=3.5,
    )

    assert ExactReferenceRecord.from_dict(optimal.to_dict()) == optimal
    assert ExactReferenceRecord.from_dict(bound_only.to_dict()) == bound_only


def test_lower_bound_only_record_carries_no_integer_claim() -> None:
    record = _record(
        status=ExactReferenceStatus.LOWER_BOUND_ONLY,
        certified_lower_bound_bars=3.5,
        integer_optimum_bars=None,
    )

    output = record.to_dict()

    assert record.status is ExactReferenceStatus.LOWER_BOUND_ONLY
    assert output["status"] == "lower_bound_only"
    assert output["integer_optimum_bars"] is None
    assert output["certified_lower_bound_bars"] == 3.5
    assert output["error_message"] is None


def test_failed_record_keeps_diagnosis_without_numerical_claim() -> None:
    record = _record(
        status=ExactReferenceStatus.FAILED,
        error_message="memory_guard_triggered",
        integer_optimum_bars=None,
        certified_lower_bound_bars=None,
    )

    output = record.to_dict()

    assert output["status"] == "failed"
    assert output["error_message"] == "memory_guard_triggered"
    assert output["integer_optimum_bars"] is None
    assert output["certified_lower_bound_bars"] is None


def test_optimal_reference_accepts_bound_within_feasibility_tolerance() -> None:
    record = _record(certified_lower_bound_bars=4.0 - 5e-10)

    assert record.certified_lower_bound_bars == pytest.approx(4.0 - 5e-10)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "exact-reference-v2"}, "unsupported schema_version"),
        ({"instance_id": ""}, "instance_id"),
        ({"instance_id": "  "}, "instance_id"),
        ({"method_limits": ""}, "method_limits"),
        ({"reference_method": "branch_and_price"}, "reference_method"),
        ({"status": "heuristic"}, "status"),
        ({"integrality_tolerance": float("nan")}, "integrality_tolerance"),
        ({"feasibility_tolerance": -1.0}, "feasibility_tolerance"),
        ({"certified_lower_bound_bars": float("inf")}, "certified_lower_bound_bars"),
        ({"integer_optimum_bars": 0}, "positive integer"),
        ({"integer_optimum_bars": True}, "positive integer"),
        ({"integer_optimum_bars": 4.0}, "positive integer"),
    ],
)
def test_record_rejects_invalid_fields(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _record(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"integer_optimum_bars": None}, "integer_optimum_bars is required"),
        ({"certified_lower_bound_bars": None}, "certified_lower_bound_bars is required"),
        ({"certified_lower_bound_bars": 4.0 + 1e-6}, "cannot exceed"),
    ],
)
def test_optimal_status_requires_consistent_proof(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _record(**changes)


def test_lower_bound_only_status_excludes_unproven_optimum() -> None:
    with pytest.raises(ValueError, match="requires an optimality proof"):
        _record(status=ExactReferenceStatus.LOWER_BOUND_ONLY)

    with pytest.raises(ValueError, match="optimality proof"):
        _record(status=ExactReferenceStatus.LOWER_BOUND_ONLY, integer_optimum_bars=4)


def test_failed_status_requires_diagnosis_and_rejects_claims() -> None:
    with pytest.raises(ValueError, match="error_message"):
        _record(
            status=ExactReferenceStatus.FAILED,
            integer_optimum_bars=None,
            certified_lower_bound_bars=None,
        )

    with pytest.raises(ValueError, match="numerical claims"):
        _record(status=ExactReferenceStatus.FAILED, error_message="oom", integer_optimum_bars=4)

    with pytest.raises(ValueError, match="numerical claims"):
        _record(
            status=ExactReferenceStatus.FAILED,
            error_message="oom",
            certified_lower_bound_bars=3.0,
        )


def test_from_dict_requires_complete_environment_and_supported_version() -> None:
    payload = _record().to_dict()
    del payload["hardware_id"]
    with pytest.raises(ValueError, match="complete environment fields"):
        ExactReferenceRecord.from_dict(payload)

    with pytest.raises(ValueError, match="unsupported schema_version"):
        ExactReferenceRecord.from_dict({**_record().to_dict(), "schema_version": "other-v9"})
