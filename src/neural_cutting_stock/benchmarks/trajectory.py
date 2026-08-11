"""Versioned, replay-oriented schema for column-generation trajectories."""

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

from .schema import EnvironmentMetadata

TRAJECTORY_SCHEMA_VERSION = "cg-trajectory-v2"
DUAL_SIGN_CONVENTION = "nonnegative_covering_dual"


class TrajectoryStatus(StrEnum):
    """Stable terminal outcomes for a recorded classical trajectory."""

    CONVERGED = "converged"
    RESOURCE_LIMIT = "resource_limit"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TrajectoryMetadata:
    """Identity and numerical conventions shared by all iterations."""

    trajectory_id: str
    instance_id: str
    solver_version: str
    seed: int
    config_id: str
    environment: EnvironmentMetadata
    stock_length: float
    kerf: float
    piece_lengths: tuple[float, ...]
    demands: tuple[int, ...]
    reduced_cost_tolerance: float
    integrality_tolerance: float
    feasibility_tolerance: float
    dual_type_order: tuple[float, ...]
    dual_tolerance: float
    dual_sign_convention: str = DUAL_SIGN_CONVENTION
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "instance_id", "solver_version", "config_id"):
            _require_text(name, getattr(self, name))
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if len(self.piece_lengths) == 0 or len(self.piece_lengths) != len(self.demands):
            raise ValueError("piece_lengths and demands must have the same non-zero length")
        if self.dual_type_order != self.piece_lengths:
            raise ValueError("dual_type_order must match piece_lengths exactly")
        if any(length <= 0 or not math.isfinite(length) for length in self.piece_lengths):
            raise ValueError("piece_lengths must contain finite positive values")
        if any(
            not isinstance(demand, int) or isinstance(demand, bool) or demand <= 0
            for demand in self.demands
        ):
            raise ValueError("demands must contain positive integers")
        for name in ("stock_length", "kerf"):
            _require_finite(name, getattr(self, name))
        if self.stock_length <= 0 or self.kerf < 0:
            raise ValueError("stock_length must be positive and kerf must be non-negative")
        for name in (
            "reduced_cost_tolerance",
            "integrality_tolerance",
            "feasibility_tolerance",
            "dual_tolerance",
        ):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.dual_sign_convention != DUAL_SIGN_CONVENTION:
            raise ValueError(f"unsupported dual_sign_convention: {self.dual_sign_convention!r}")


@dataclass(frozen=True, slots=True)
class TrajectoryIteration:
    """One numbered observation; unavailable measurements remain null."""

    iteration_index: int
    rmp_status: str
    pricing_status: str | None = None
    rmp_objective_value: float | None = None
    dual_values: tuple[float, ...] | None = None
    initial_column_count: int | None = None
    final_column_count: int | None = None
    columns_added: int | None = None
    duplicate_column_count: int | None = None
    candidate_patterns: tuple[tuple[int, ...], ...] | None = None
    candidate_reduced_costs: tuple[float, ...] | None = None
    selected_patterns: tuple[tuple[int, ...], ...] | None = None
    exact_fallback: bool | None = None
    best_reduced_cost: float | None = None
    rmp_runtime_seconds: float | None = None
    pricing_runtime_seconds: float | None = None
    column_management_runtime_seconds: float | None = None
    instance_id: str | None = None
    rmp_column_values: tuple[float, ...] | None = None
    rmp_pattern_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.iteration_index, int) or isinstance(self.iteration_index, bool):
            raise ValueError("iteration_index must be an integer")
        if self.iteration_index < 1:
            raise ValueError("iteration_index must start at 1")
        _require_text("rmp_status", self.rmp_status)
        if self.instance_id is not None:
            _require_text("instance_id", self.instance_id)
        if self.pricing_status is not None:
            _require_text("pricing_status", self.pricing_status)
        if (
            self.candidate_reduced_costs is not None
            and self.candidate_patterns is not None
            and len(self.candidate_reduced_costs) != len(self.candidate_patterns)
        ):
            raise ValueError("candidate patterns and reduced costs must have the same length")
        if self.dual_values is not None:
            if len(self.dual_values) == 0:
                raise ValueError("dual_values must not be empty when present")
            if any(value < 0 for value in self.dual_values):
                raise ValueError("dual_values must be non-negative under the covering convention")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float):
                _require_finite(field.name, value)
            if (
                field.name.endswith("_count")
                and value is not None
                and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
            ):
                raise ValueError(f"{field.name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ColumnGenerationTrajectory:
    """Complete replay unit for one classical column-generation execution."""

    metadata: TrajectoryMetadata
    iterations: tuple[TrajectoryIteration, ...]
    status: TrajectoryStatus
    termination_reason: str
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TrajectoryStatus):
            try:
                object.__setattr__(self, "status", TrajectoryStatus(self.status))
            except ValueError as error:
                raise ValueError("status is not a supported trajectory status") from error
        if not self.iterations:
            raise ValueError("a trajectory must contain at least one iteration")
        expected = tuple(range(1, len(self.iterations) + 1))
        if tuple(item.iteration_index for item in self.iterations) != expected:
            raise ValueError("iteration_index values must be contiguous and start at 1")
        for iteration in self.iterations:
            if iteration.dual_values is not None and len(iteration.dual_values) != len(
                self.metadata.dual_type_order
            ):
                raise ValueError("dual_values must follow metadata.dual_type_order")
        _require_text("termination_reason", self.termination_reason)
        if self.status is not TrajectoryStatus.CONVERGED and not self.error_message:
            raise ValueError("error_message is required for a non-converged trajectory")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready nested representation."""

        metadata = {
            field.name: getattr(self.metadata, field.name)
            for field in fields(self.metadata)
            if field.name != "environment"
        }
        metadata.update(
            {
                "code_commit": self.metadata.environment.code_commit,
                "python_version": self.metadata.environment.python_version,
                "dependency_versions": self.metadata.environment.dependency_versions,
                "hardware_id": self.metadata.environment.hardware_id,
            }
        )
        output = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "metadata": metadata,
            "iterations": [_as_json_ready(item) for item in self.iterations],
            "status": self.status.value,
            "termination_reason": self.termination_reason,
            "error_message": self.error_message,
        }
        return output


def _as_json_ready(item: TrajectoryIteration) -> dict[str, Any]:
    return {field.name: getattr(item, field.name) for field in fields(item)}


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
