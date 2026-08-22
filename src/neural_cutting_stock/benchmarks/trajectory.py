"""Versioned, replay-oriented schema for column-generation trajectories."""

import json
import math
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any

from neural_cutting_stock.problem import CuttingStockInstance

from ._validation import require_finite as _require_finite
from ._validation import require_text as _require_text
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
        if (self.candidate_patterns is None) != (self.candidate_reduced_costs is None):
            raise ValueError("candidate patterns and reduced costs must be recorded together")
        if self.candidate_patterns is not None:
            if len(self.candidate_patterns) != len(self.candidate_reduced_costs):
                raise ValueError("candidate patterns and reduced costs must have the same length")
            for pattern in self.candidate_patterns:
                _require_pattern("candidate_patterns", pattern)
            for reduced_cost in self.candidate_reduced_costs:
                _require_finite("candidate_reduced_costs", reduced_cost)
        if self.selected_patterns is not None:
            for pattern in self.selected_patterns:
                _require_pattern("selected_patterns", pattern)
        if self.exact_fallback is not None and not isinstance(self.exact_fallback, bool):
            raise ValueError("exact_fallback must be a boolean when present")
        if self.dual_values is not None:
            if len(self.dual_values) == 0:
                raise ValueError("dual_values must not be empty when present")
            if any(value < 0 for value in self.dual_values):
                raise ValueError("dual_values must be non-negative under the covering convention")
        progress_fields = (
            "initial_column_count",
            "final_column_count",
            "columns_added",
            "duplicate_column_count",
        )
        for name in progress_fields:
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.initial_column_count is not None
            and self.final_column_count is not None
            and self.columns_added is not None
            and self.final_column_count != self.initial_column_count + self.columns_added
        ):
            raise ValueError(
                "final_column_count must equal initial_column_count plus columns_added"
            )
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float):
                _require_finite(field.name, value)
            if field.name.endswith("_runtime_seconds") and value is not None and value < 0:
                raise ValueError(f"{field.name} must be non-negative")


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

    @classmethod
    def from_dict(cls, value: object) -> "ColumnGenerationTrajectory":
        """Build a trajectory from the persisted JSON representation."""

        if not isinstance(value, dict):
            raise ValueError("trajectory must be a JSON object")
        if value.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        metadata_value = value.get("metadata")
        iterations_value = value.get("iterations")
        if not isinstance(metadata_value, dict):
            raise ValueError("metadata must be a JSON object")
        if not isinstance(iterations_value, list):
            raise ValueError("iterations must be a JSON array")

        metadata_value = dict(metadata_value)
        environment_fields = {
            "code_commit": metadata_value.pop("code_commit", None),
            "python_version": metadata_value.pop("python_version", None),
            "dependency_versions": metadata_value.pop("dependency_versions", None),
            "hardware_id": metadata_value.pop("hardware_id", None),
        }
        if any(value is None for value in environment_fields.values()):
            raise ValueError("metadata must contain complete environment fields")
        metadata = TrajectoryMetadata(
            **_tuple_fields(metadata_value, {"piece_lengths", "demands", "dual_type_order"}),
            environment=EnvironmentMetadata(**environment_fields),
        )
        iterations = tuple(
            TrajectoryIteration(
                **_tuple_fields(
                    item,
                    {
                        "dual_values",
                        "candidate_reduced_costs",
                        "selected_patterns",
                        "rmp_column_values",
                    },
                    nested_patterns={"candidate_patterns", "selected_patterns"},
                )
            )
            for item in iterations_value
        )
        return cls(
            metadata,
            iterations,
            TrajectoryStatus(value.get("status")),
            value.get("termination_reason"),
            value.get("error_message"),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryValidation:
    """Result of validating a trajectory and replaying its exact classical solve."""

    replayed_result: Any | None
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class TrajectoryCollectionMeasurement:
    """Measured cost of materializing and serializing one solver result."""

    trajectory: ColumnGenerationTrajectory
    collection_runtime_seconds: float
    serialized_size_bytes: int


@dataclass(frozen=True, slots=True)
class UsefulColumnTarget:
    """Counterfactual target for a column that reduces classical CG work."""

    pattern: tuple[int, ...]
    work_without_column_seconds: float
    work_with_column_seconds: float
    work_reduction_seconds: float
    useful: bool


def define_useful_column_target(
    without_column: ColumnGenerationTrajectory,
    with_column: ColumnGenerationTrajectory,
    pattern: tuple[int, ...],
    tolerance: float = 1e-12,
) -> UsefulColumnTarget:
    """Label a column from two comparable, measured classical trajectories.

    Work is the recorded RMP, pricing and column-management time. A missing
    component is rejected instead of being treated as zero.
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if without_column.metadata.instance_id != with_column.metadata.instance_id:
        raise ValueError("trajectories must describe the same instance")
    if without_column.metadata.config_id != with_column.metadata.config_id:
        raise ValueError("trajectories must use the same configuration")
    if len(pattern) == 0 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pattern
    ):
        raise ValueError("pattern must contain non-negative integers")
    selected_patterns = tuple(
        selected
        for iteration in with_column.iterations
        for selected in (iteration.selected_patterns or ())
    )
    if pattern not in selected_patterns:
        raise ValueError("pattern must be selected in the with-column trajectory")

    without_work = _trajectory_work_seconds(without_column)
    with_work = _trajectory_work_seconds(with_column)
    reduction = without_work - with_work
    return UsefulColumnTarget(
        pattern,
        without_work,
        with_work,
        reduction,
        reduction > tolerance,
    )


def _trajectory_work_seconds(trajectory: ColumnGenerationTrajectory) -> float:
    durations = (
        "rmp_runtime_seconds",
        "pricing_runtime_seconds",
        "column_management_runtime_seconds",
    )
    values = [getattr(iteration, name) for iteration in trajectory.iterations for name in durations]
    if any(value is None for value in values):
        raise ValueError("trajectory lacks complete column-generation work measurements")
    return sum(value for value in values if value is not None)


def collect_trajectory(
    result: Any,
    metadata: TrajectoryMetadata,
) -> TrajectoryCollectionMeasurement:
    """Collect a completed result without re-entering the solver loop."""

    started = perf_counter()
    iterations = []
    for index, state in enumerate(result.rmp_states):
        next_patterns = (
            result.rmp_states[index + 1].patterns
            if index + 1 < len(result.rmp_states)
            else result.patterns
        )
        selected_patterns = tuple(
            pattern for pattern in next_patterns if pattern not in state.patterns
        )
        iterations.append(
            TrajectoryIteration(
                iteration_index=state.iteration_index,
                rmp_status="optimal" if state.result.status == 0 else str(state.result.status),
                rmp_objective_value=state.result.objective_value,
                dual_values=state.result.dual_values,
                initial_column_count=len(state.patterns),
                final_column_count=len(next_patterns),
                columns_added=len(selected_patterns),
                duplicate_column_count=(
                    result.duplicate_columns if index == len(result.rmp_states) - 1 else 0
                ),
                selected_patterns=selected_patterns,
                exact_fallback=True,
                rmp_runtime_seconds=state.runtime_seconds,
                instance_id=state.instance_id,
                rmp_column_values=state.result.column_values,
                rmp_pattern_count=len(state.patterns),
            )
        )
    if not iterations:
        raise ValueError("cannot collect a result without RMP states")
    status = {
        "converged": TrajectoryStatus.CONVERGED,
        "limit_reached": TrajectoryStatus.RESOURCE_LIMIT,
    }.get(result.status, TrajectoryStatus.FAILED)
    trajectory = ColumnGenerationTrajectory(
        metadata,
        tuple(iterations),
        status,
        result.termination_reason,
        None if status is TrajectoryStatus.CONVERGED else result.termination_reason,
    )
    serialized_size = len(json.dumps(trajectory.to_dict(), sort_keys=True).encode("utf-8"))
    return TrajectoryCollectionMeasurement(
        trajectory,
        perf_counter() - started,
        serialized_size,
    )


class TrajectoryReader:
    """Read and replay JSON trajectories without changing solver decisions."""

    @staticmethod
    def read(path: str | Path) -> ColumnGenerationTrajectory:
        """Read one UTF-8 trajectory and validate its persisted schema."""

        with Path(path).open(encoding="utf-8") as stream:
            return ColumnGenerationTrajectory.from_dict(json.load(stream))

    @staticmethod
    def replay(trajectory: ColumnGenerationTrajectory) -> TrajectoryValidation:
        """Replay exact pricing and compare every recorded observation available."""

        from neural_cutting_stock.solver import ColumnGeneration

        metadata = trajectory.metadata
        instance = CuttingStockInstance(
            metadata.stock_length,
            metadata.kerf,
            metadata.piece_lengths,
            metadata.demands,
        )
        result = ColumnGeneration(
            instance,
            reduced_cost_tolerance=metadata.reduced_cost_tolerance,
            instance_id=metadata.instance_id,
        ).solve()
        errors: list[str] = []
        expected_statuses = {
            TrajectoryStatus.CONVERGED: {"converged"},
            TrajectoryStatus.RESOURCE_LIMIT: {"limit_reached"},
            TrajectoryStatus.FAILED: {"infeasible", "solver_error", "invalid_plan"},
        }
        if result.status not in expected_statuses[trajectory.status]:
            errors.append(
                f"status differs: recorded={trajectory.status.value}, replayed={result.status}"
            )
        if result.termination_reason != trajectory.termination_reason:
            errors.append("termination_reason differs")
        if len(result.rmp_states) != len(trajectory.iterations):
            errors.append("iteration count differs")
        for index, (recorded, state) in enumerate(
            zip(trajectory.iterations, result.rmp_states, strict=False)
        ):
            next_patterns = (
                result.rmp_states[index + 1].patterns
                if index + 1 < len(result.rmp_states)
                else result.patterns
            )
            _compare_iteration(
                recorded,
                state,
                next_patterns,
                result,
                index == len(trajectory.iterations) - 1,
                errors,
                metadata.dual_tolerance,
            )
        return TrajectoryValidation(result, tuple(errors))


def write_trajectory(path: str | Path, trajectory: ColumnGenerationTrajectory) -> None:
    """Write a trajectory as deterministic, human-readable UTF-8 JSON."""

    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(trajectory.to_dict(), stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")


def read_trajectory(path: str | Path) -> ColumnGenerationTrajectory:
    """Read one trajectory from disk."""

    return TrajectoryReader.read(path)


def replay_trajectory(trajectory: ColumnGenerationTrajectory) -> TrajectoryValidation:
    """Replay one trajectory through the exact classical solver."""

    return TrajectoryReader.replay(trajectory)


def _tuple_fields(
    value: object,
    fields_to_convert: set[str],
    nested_patterns: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("trajectory fields must be JSON objects")
    output = dict(value)
    for name in fields_to_convert:
        if output.get(name) is not None:
            output[name] = tuple(output[name])
    for name in nested_patterns or set():
        if output.get(name) is not None:
            output[name] = tuple(tuple(pattern) for pattern in output[name])
    return output


def _compare_iteration(
    recorded: TrajectoryIteration,
    state: Any,
    next_patterns: tuple[tuple[int, ...], ...],
    result: Any,
    is_final: bool,
    errors: list[str],
    tolerance: float,
) -> None:
    rmp_result = state.result
    if recorded.instance_id is not None and recorded.instance_id != state.instance_id:
        errors.append(f"iteration {recorded.iteration_index}: instance_id differs")
    if recorded.rmp_status == "optimal" and rmp_result.status != 0:
        errors.append(f"iteration {recorded.iteration_index}: RMP status differs")
    if recorded.rmp_objective_value is not None and not _close(
        recorded.rmp_objective_value, rmp_result.objective_value, tolerance
    ):
        errors.append(f"iteration {recorded.iteration_index}: RMP objective differs")
    if recorded.dual_values is not None and not _close_sequence(
        recorded.dual_values, rmp_result.dual_values, tolerance
    ):
        errors.append(f"iteration {recorded.iteration_index}: dual values differ")
    if recorded.rmp_column_values is not None and not _close_sequence(
        recorded.rmp_column_values, rmp_result.column_values, tolerance
    ):
        errors.append(f"iteration {recorded.iteration_index}: RMP column values differ")
    if recorded.rmp_pattern_count is not None and recorded.rmp_pattern_count != len(state.patterns):
        errors.append(f"iteration {recorded.iteration_index}: RMP pattern count differs")
    if recorded.initial_column_count is not None and recorded.initial_column_count != len(
        state.patterns
    ):
        errors.append(f"iteration {recorded.iteration_index}: initial column count differs")
    if recorded.final_column_count is not None and recorded.final_column_count != len(
        next_patterns
    ):
        errors.append(f"iteration {recorded.iteration_index}: final column count differs")
    if recorded.columns_added is not None:
        actual_added = len(next_patterns) - len(state.patterns)
        if recorded.columns_added != actual_added:
            errors.append(f"iteration {recorded.iteration_index}: columns added differs")
    if recorded.selected_patterns is not None:
        actual_selected = next_patterns[len(state.patterns) :]
        if recorded.selected_patterns != actual_selected:
            errors.append(f"iteration {recorded.iteration_index}: selected patterns differ")
    if recorded.pricing_status == "optimal" and (
        result.pricing_result is None or result.pricing_result.status != 0
    ):
        errors.append(f"iteration {recorded.iteration_index}: pricing status differs")
    if is_final and recorded.best_reduced_cost is not None and (
        result.pricing_result is None
        or not _close(recorded.best_reduced_cost, result.pricing_result.reduced_cost, tolerance)
    ):
        errors.append(f"iteration {recorded.iteration_index}: best reduced cost differs")


def _close(left: float, right: float | None, tolerance: float) -> bool:
    return right is not None and math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _close_sequence(left: tuple[float, ...], right: tuple[float, ...], tolerance: float) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right, strict=True)
    )


def _as_json_ready(item: TrajectoryIteration) -> dict[str, Any]:
    return {field.name: getattr(item, field.name) for field in fields(item)}


def _require_pattern(name: str, pattern: tuple[int, ...]) -> None:
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in pattern
    ):
        raise ValueError(f"{name} must contain non-negative integers")
