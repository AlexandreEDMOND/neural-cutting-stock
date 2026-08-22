"""Versioned schema for exact cutting-stock quality references."""

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

from .schema import EnvironmentMetadata

EXACT_REFERENCE_SCHEMA_VERSION = "exact-reference-v1"


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


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
