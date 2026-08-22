"""Versioned schema for raw benchmark execution records."""

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

from ._validation import require_text as _require_text

SCHEMA_VERSION = "benchmark-run-v1"


class SolverMode(StrEnum):
    """Solver modes that may be compared by the benchmark protocol."""

    CLASSICAL = "classical"
    NEURAL = "neural"


class RunStatus(StrEnum):
    """Stable top-level outcomes retained in raw benchmark data."""

    OPTIMAL_LP_RESTRICTED_IP = "optimal_lp_restricted_ip"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    SOLVER_ERROR = "solver_error"
    INVALID_PLAN = "invalid_plan"


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    """Environment information needed to reproduce an execution."""

    code_commit: str
    python_version: str
    dependency_versions: str
    hardware_id: str

    def __post_init__(self) -> None:
        for name in ("code_commit", "python_version", "dependency_versions", "hardware_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BenchmarkRunRecord:
    """One raw run, including failures and unavailable measurements as nulls."""

    run_id: str
    instance_id: str
    solver_mode: SolverMode
    solver_version: str
    seed: int
    config_id: str
    repetition: int
    environment: EnvironmentMetadata
    stock_length: float
    kerf: float
    number_of_piece_types: int
    total_demand: int
    requested_length: float
    length_distribution: str
    demand_distribution: str
    run_status: RunStatus
    master_status: str
    pricing_status: str
    integer_master_status: str
    termination_reason: str
    schema_version: str = SCHEMA_VERSION
    size_class: str | None = None
    objective_value: float | None = None
    number_of_stock_bars: int | None = None
    lp_objective_value: float | None = None
    restricted_integer_gap: float | None = None
    total_waste: float | None = None
    trim_loss: float | None = None
    kerf_loss: float | None = None
    overproduction_length: float | None = None
    plan_feasible: bool | None = None
    number_of_cg_iterations: int | None = None
    number_of_generated_columns: int | None = None
    number_of_columns_added: int | None = None
    initial_column_count: int | None = None
    final_column_count: int | None = None
    duplicate_column_count: int | None = None
    final_reduced_cost: float | None = None
    total_runtime_seconds: float | None = None
    master_problem_runtime: float | None = None
    pricing_runtime: float | None = None
    integer_master_runtime: float | None = None
    column_management_runtime: float | None = None
    verification_runtime: float | None = None
    unattributed_runtime: float | None = None
    peak_memory_bytes: int | None = None
    exact_pricing_calls: int | None = None
    error_message: str | None = None
    model_id: str | None = None
    neural_inference_runtime: float | None = None
    feature_preparation_runtime: float | None = None
    number_of_candidates: int | None = None
    number_of_selected_columns: int | None = None
    exact_fallback_calls: int | None = None
    speedup_vs_classical: float | None = None
    objective_difference_vs_classical: float | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "instance_id", "solver_version", "config_id"):
            _require_text(name, getattr(self, name))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.solver_mode, SolverMode):
            try:
                object.__setattr__(self, "solver_mode", SolverMode(self.solver_mode))
            except ValueError as error:
                raise ValueError("solver_mode must be 'classical' or 'neural'") from error
        if not isinstance(self.run_status, RunStatus):
            try:
                object.__setattr__(self, "run_status", RunStatus(self.run_status))
            except ValueError as error:
                raise ValueError("run_status is not a supported raw-run status") from error
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(self.repetition, int)
            or isinstance(self.repetition, bool)
            or self.repetition < 0
        ):
            raise ValueError("repetition must be a non-negative integer")
        if self.solver_mode is SolverMode.NEURAL and not self.model_id:
            raise ValueError("model_id is required for neural runs")
        if self.solver_mode is SolverMode.CLASSICAL and any(
            value is not None
            for value in (
                self.model_id,
                self.neural_inference_runtime,
                self.feature_preparation_runtime,
                self.number_of_candidates,
                self.number_of_selected_columns,
                self.exact_fallback_calls,
                self.speedup_vs_classical,
                self.objective_difference_vs_classical,
            )
        ):
            raise ValueError("neural-only fields must be null for classical runs")
        if self.run_status is not RunStatus.OPTIMAL_LP_RESTRICTED_IP and not self.error_message:
            raise ValueError("error_message is required for non-successful runs")
        for field in fields(self):
            if field.name == "environment":
                continue
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite when present")
            if field.name in {
                "peak_memory_bytes",
                "exact_pricing_calls",
                "number_of_candidates",
                "number_of_selected_columns",
            } and value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field.name} must be a non-negative integer when present")

    def to_dict(self) -> dict[str, Any]:
        """Return the flat, JSON-ready representation used by raw tables."""

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
        values["solver_mode"] = self.solver_mode.value
        values["run_status"] = self.run_status.value
        return values
