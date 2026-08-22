"""Versioned schema for exact cutting-stock quality references."""

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

from neural_cutting_stock.problem import CuttingStockInstance
from neural_cutting_stock.solver.complete_master import (
    CompleteIntegerMaster,
    CompleteMasterResult,
)
from neural_cutting_stock.solver.maximal_patterns import (
    MaximalPatternLimits,
    PatternEnumerationLimitExceeded,
)

from .schema import EnvironmentMetadata

EXACT_REFERENCE_SCHEMA_VERSION = "exact-reference-v1"

MILP_METHOD_LIMITS_PREFIX = "maximal_patterns"


class ExactReferenceMethod(StrEnum):
    """Exact methods allowed to produce a quality reference."""

    EXHAUSTIVE_PATTERN_ENUMERATION = "exhaustive_pattern_enumeration"
    MILP_ON_ENUMERATED_PATTERNS = "milp_on_enumerated_patterns"


class ExactReferenceStatus(StrEnum):
    """Stable outcomes retained in an exact-reference record."""

    OPTIMAL = "optimal"
    LOWER_BOUND_ONLY = "lower_bound_only"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExactReferenceRecord:
    """Certified quality reference for one instance under `exact-reference-v1`.

    An `optimal` record carries the proven integer optimum together with its
    associated certified lower bound. A `lower_bound_only` record carries only
    a certified bound. Failures keep their diagnosis and never carry numbers.
    """

    instance_id: str
    reference_method: ExactReferenceMethod
    status: ExactReferenceStatus
    method_limits: str
    environment: EnvironmentMetadata
    integrality_tolerance: float
    feasibility_tolerance: float
    integer_optimum_bars: int | None = None
    certified_lower_bound_bars: float | None = None
    error_message: str | None = None
    schema_version: str = EXACT_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("instance_id", self.instance_id)
        _require_text("method_limits", self.method_limits)
        if self.schema_version != EXACT_REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.reference_method, ExactReferenceMethod):
            try:
                object.__setattr__(
                    self, "reference_method", ExactReferenceMethod(self.reference_method)
                )
            except ValueError as error:
                raise ValueError("reference_method is not a supported exact method") from error
        if not isinstance(self.status, ExactReferenceStatus):
            try:
                object.__setattr__(self, "status", ExactReferenceStatus(self.status))
            except ValueError as error:
                raise ValueError("status is not a supported exact-reference status") from error
        for name in ("integrality_tolerance", "feasibility_tolerance"):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.integer_optimum_bars is not None and (
            not isinstance(self.integer_optimum_bars, int)
            or isinstance(self.integer_optimum_bars, bool)
            or self.integer_optimum_bars < 1
        ):
            raise ValueError("integer_optimum_bars must be a positive integer when present")
        if self.certified_lower_bound_bars is not None:
            _require_finite("certified_lower_bound_bars", self.certified_lower_bound_bars)
            if self.certified_lower_bound_bars <= 0:
                raise ValueError("certified_lower_bound_bars must be positive when present")
        if self.status is ExactReferenceStatus.OPTIMAL:
            if self.integer_optimum_bars is None:
                raise ValueError("integer_optimum_bars is required for an optimal reference")
            if self.certified_lower_bound_bars is None:
                raise ValueError("certified_lower_bound_bars is required for an optimal reference")
            if (
                self.certified_lower_bound_bars
                > self.integer_optimum_bars + self.feasibility_tolerance
            ):
                raise ValueError(
                    "certified_lower_bound_bars cannot exceed integer_optimum_bars "
                    "within the declared feasibility tolerance"
                )
        elif self.status is ExactReferenceStatus.LOWER_BOUND_ONLY:
            if self.certified_lower_bound_bars is None:
                raise ValueError(
                    "certified_lower_bound_bars is required for a lower-bound-only reference"
                )
            if self.integer_optimum_bars is not None:
                raise ValueError(
                    "integer_optimum_bars requires an optimality proof, not a bound alone"
                )
        elif self.integer_optimum_bars is not None or self.certified_lower_bound_bars is not None:
            raise ValueError("a failed reference must not carry numerical claims")
        if self.status is ExactReferenceStatus.FAILED and not self.error_message:
            raise ValueError("error_message is required for a failed reference")

    def to_dict(self) -> dict[str, Any]:
        """Return the flat, JSON-ready representation used by persisted references."""

        values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "environment"
        }
        values.update(
            {
                "code_commit": self.environment.code_commit,
                "python_version": self.environment.python_version,
                "dependency_versions": self.environment.dependency_versions,
                "hardware_id": self.environment.hardware_id,
            }
        )
        values["reference_method"] = self.reference_method.value
        values["status"] = self.status.value
        return values

    @classmethod
    def from_dict(cls, value: object) -> "ExactReferenceRecord":
        """Build a reference from its persisted JSON representation."""

        if not isinstance(value, dict):
            raise ValueError("exact reference must be a JSON object")
        if value.get("schema_version") != EXACT_REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        record_fields = dict(value)
        environment_fields = {
            name: record_fields.pop(name, None)
            for name in ("code_commit", "python_version", "dependency_versions", "hardware_id")
        }
        if any(item is None for item in environment_fields.values()):
            raise ValueError("exact reference must contain complete environment fields")
        return cls(
            **record_fields,
            environment=EnvironmentMetadata(**environment_fields),
        )


def build_milp_exact_reference(
    instance_id: str,
    outcome: CompleteMasterResult,
    *,
    environment: EnvironmentMetadata,
    integrality_tolerance: float,
    feasibility_tolerance: float,
    method_limits: str,
) -> ExactReferenceRecord:
    """Map a complete-master MILP outcome onto an `exact-reference-v1` record.

    A proven optimum carries the integer objective together with HiGHS'
    dual bound, clamped to the proven optimum so numeric noise never lifts
    the bound above it. A solver limit keeps its certified dual bound as a
    `lower_bound_only` reference. Every other outcome persists as `failed`
    with its diagnosis and without numerical claims.
    """

    def failed(message: str) -> ExactReferenceRecord:
        return ExactReferenceRecord(
            instance_id=instance_id,
            reference_method=ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS,
            status=ExactReferenceStatus.FAILED,
            method_limits=method_limits,
            environment=environment,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            error_message=message,
        )

    if not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")

    if outcome.status == 0:
        if outcome.objective_value is None:
            return failed("optimal termination reported no objective value")
        bars = round(outcome.objective_value)
        if abs(outcome.objective_value - bars) > integrality_tolerance or bars < 1:
            return failed("objective_value is not a positive integer within integrality_tolerance")
        if outcome.certified_lower_bound is None:
            return failed("optimal termination reported no certified lower bound")
        return ExactReferenceRecord(
            instance_id=instance_id,
            reference_method=ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS,
            status=ExactReferenceStatus.OPTIMAL,
            method_limits=method_limits,
            environment=environment,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            integer_optimum_bars=bars,
            certified_lower_bound_bars=min(outcome.certified_lower_bound, float(bars)),
        )

    if (
        outcome.status == 1
        and outcome.certified_lower_bound is not None
        and outcome.certified_lower_bound > 0
    ):
        return ExactReferenceRecord(
            instance_id=instance_id,
            reference_method=ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS,
            status=ExactReferenceStatus.LOWER_BOUND_ONLY,
            method_limits=method_limits,
            environment=environment,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            certified_lower_bound_bars=outcome.certified_lower_bound,
        )

    return failed(outcome.message or f"complete master MILP stopped with status {outcome.status}")


def compute_milp_exact_reference(
    instance_id: str,
    instance: CuttingStockInstance,
    *,
    environment: EnvironmentMetadata,
    integrality_tolerance: float,
    feasibility_tolerance: float,
    limits: MaximalPatternLimits | None = None,
) -> ExactReferenceRecord:
    """Solve the complete master by MILP and persist its proof or failure.

    Enumeration guards are captured as `failed` references so refused
    instances stay visible in persisted data instead of disappearing.
    """

    effective_limits = limits if limits is not None else MaximalPatternLimits()
    method_limits = (
        f"{MILP_METHOD_LIMITS_PREFIX}:max_search_space_size="
        f"{effective_limits.max_search_space_size},max_patterns="
        f"{effective_limits.max_patterns}"
    )
    try:
        outcome = CompleteIntegerMaster(instance, limits).solve()
    except PatternEnumerationLimitExceeded as error:
        return ExactReferenceRecord(
            instance_id=instance_id,
            reference_method=ExactReferenceMethod.MILP_ON_ENUMERATED_PATTERNS,
            status=ExactReferenceStatus.FAILED,
            method_limits=method_limits,
            environment=environment,
            integrality_tolerance=integrality_tolerance,
            feasibility_tolerance=feasibility_tolerance,
            error_message=str(error),
        )
    return build_milp_exact_reference(
        instance_id,
        outcome,
        environment=environment,
        integrality_tolerance=integrality_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        method_limits=method_limits,
    )


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
