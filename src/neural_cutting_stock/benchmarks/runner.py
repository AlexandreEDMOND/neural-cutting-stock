"""Execution and persistence of configured classical benchmark matrices."""

import csv
import hashlib
import json
import math
import tracemalloc
from dataclasses import dataclass, fields
from pathlib import Path

from neural_cutting_stock.solver import ColumnGeneration, ColumnGenerationResult

from .generator import SyntheticInstanceGenerator
from .profile import classify_runtime
from .schema import (
    BenchmarkRunRecord,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
)


@dataclass(frozen=True, slots=True)
class ClassicalBenchmarkConfig:
    """Configuration for one deterministic matrix of classical runs."""

    generators: tuple[SyntheticInstanceGenerator, ...]
    environment: EnvironmentMetadata
    repetitions: int = 1
    reduced_cost_tolerance: float = 1e-9
    max_runtime_seconds: float | None = None
    max_cg_iterations: int | None = None
    solver_version: str = "classical-cg-v1"

    def __post_init__(self) -> None:
        if not self.generators:
            raise ValueError("generators must not be empty")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions <= 0
        ):
            raise ValueError("repetitions must be a positive integer")
        if self.reduced_cost_tolerance < 0:
            raise ValueError("reduced_cost_tolerance must be non-negative")
        if self.max_runtime_seconds is not None and (
            not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0
        ):
            raise ValueError("max_runtime_seconds must be finite and positive when present")
        if self.max_cg_iterations is not None and (
            not isinstance(self.max_cg_iterations, int)
            or isinstance(self.max_cg_iterations, bool)
            or self.max_cg_iterations <= 0
        ):
            raise ValueError("max_cg_iterations must be a positive integer when present")
        if not self.solver_version.strip():
            raise ValueError("solver_version must not be empty")

    @property
    def config_id(self) -> str:
        """Return the stable identifier of the complete solver matrix."""

        payload = {
            "generators": sorted(
                (_generator_payload(generator) for generator in self.generators),
                key=lambda generator: json.dumps(generator, sort_keys=True, separators=(",", ":")),
            ),
            "repetitions": self.repetitions,
            "reduced_cost_tolerance": self.reduced_cost_tolerance,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_cg_iterations": self.max_cg_iterations,
            "solver_version": self.solver_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


class ClassicalBenchmarkRunner:
    """Run every configured generator and repetition in deterministic order."""

    def __init__(self, configuration: ClassicalBenchmarkConfig) -> None:
        self.configuration = configuration

    def run(self, output_path: str | Path | None = None) -> tuple[BenchmarkRunRecord, ...]:
        """Return and optionally persist one raw record for every matrix cell."""

        records: list[BenchmarkRunRecord] = []
        for generator in self.configuration.generators:
            instance = generator.generate()
            for repetition in range(self.configuration.repetitions):
                records.append(self._run_one(generator, instance, repetition))
        if output_path is not None:
            write_raw_runs(output_path, records)
        return tuple(records)

    def _run_one(
        self,
        generator: SyntheticInstanceGenerator,
        instance,
        repetition: int,
    ) -> BenchmarkRunRecord:
        run_key = f"{self.configuration.config_id}:{generator.instance_id}:{repetition}"
        run_id = hashlib.sha256(run_key.encode("ascii")).hexdigest()
        try:
            result, peak_memory_bytes = solve_with_peak_memory(ColumnGeneration(
                instance,
                self.configuration.reduced_cost_tolerance,
                self.configuration.max_runtime_seconds,
                self.configuration.max_cg_iterations,
                generator.instance_id,
            ))
        except Exception as error:  # Keep a matrix cell visible when a solver call fails.
            return self._failed_record(generator, instance, repetition, run_id, str(error))
        return self._record_from_result(
            generator, instance, repetition, run_id, result, peak_memory_bytes=peak_memory_bytes
        )

    def _record_from_result(
        self,
        generator: SyntheticInstanceGenerator,
        instance,
        repetition: int,
        run_id: str,
        result: ColumnGenerationResult,
        solver_mode: SolverMode = SolverMode.CLASSICAL,
        solver_version: str | None = None,
        model_id: str | None = None,
        neural_profile=None,
        peak_memory_bytes: int | None = None,
    ) -> BenchmarkRunRecord:
        verification = result.verification
        integer = result.integer_master_result
        rmp = result.rmp_result
        pricing = result.pricing_result
        status = _run_status(result.status)
        error_message = (
            None
            if status is RunStatus.OPTIMAL_LP_RESTRICTED_IP
            else result.termination_reason
        )
        return BenchmarkRunRecord(
            **self._record_identity(
                generator, instance, repetition, run_id, solver_mode=solver_mode,
                solver_version=solver_version,
            ),
            run_status=status,
            master_status=_component_status(rmp.status if rmp else None),
            pricing_status=_component_status(pricing.status if pricing else None),
            integer_master_status=_component_status(integer.status if integer else None),
            termination_reason=result.termination_reason,
            size_class=(
                classify_runtime(result.total_runtime_seconds)
                if result.total_runtime_seconds is not None
                else None
            ),
            objective_value=integer.objective_value if integer else None,
            number_of_stock_bars=verification.number_of_stock_bars if verification else None,
            lp_objective_value=rmp.objective_value if rmp else None,
            restricted_integer_gap=result.integrality_gap,
            total_waste=verification.total_waste if verification else None,
            trim_loss=verification.trim_loss if verification else None,
            kerf_loss=verification.kerf_loss if verification else None,
            overproduction_length=verification.overproduction_length if verification else None,
            plan_feasible=verification.feasible if verification else None,
            number_of_cg_iterations=result.iterations,
            number_of_generated_columns=result.columns_added + result.duplicate_columns,
            number_of_columns_added=result.columns_added,
            initial_column_count=len(instance.initial_patterns()),
            final_column_count=len(result.patterns),
            duplicate_column_count=result.duplicate_columns,
            final_reduced_cost=pricing.reduced_cost if pricing else None,
            total_runtime_seconds=result.total_runtime_seconds,
            master_problem_runtime=result.master_problem_runtime,
            pricing_runtime=result.pricing_runtime,
            integer_master_runtime=result.integer_master_runtime,
            column_management_runtime=result.column_management_runtime,
            verification_runtime=result.verification_runtime,
            unattributed_runtime=result.unattributed_runtime,
            peak_memory_bytes=peak_memory_bytes,
            exact_pricing_calls=result.exact_pricing_calls,
            error_message=error_message,
            model_id=model_id,
            neural_inference_runtime=(
                neural_profile.neural_inference_runtime if neural_profile else None
            ),
            feature_preparation_runtime=(
                neural_profile.feature_preparation_runtime if neural_profile else None
            ),
            number_of_candidates=(neural_profile.number_of_candidates if neural_profile else None),
            number_of_selected_columns=(
                neural_profile.number_of_selected_columns if neural_profile else None
            ),
            exact_fallback_calls=(
                neural_profile.exact_fallback_calls if neural_profile else None
            ),
        )

    def _failed_record(
        self,
        generator,
        instance,
        repetition,
        run_id,
        message,
        *,
        solver_mode: SolverMode = SolverMode.CLASSICAL,
        solver_version: str | None = None,
        model_id: str | None = None,
    ):
        return BenchmarkRunRecord(
            **self._record_identity(
                generator,
                instance,
                repetition,
                run_id,
                solver_mode=solver_mode,
                solver_version=solver_version,
            ),
            run_status=RunStatus.SOLVER_ERROR,
            master_status="not_run",
            pricing_status="not_run",
            integer_master_status="not_run",
            termination_reason="solver_exception",
            error_message=message or "solver call failed",
            model_id=model_id,
        )

    def _record_identity(
        self,
        generator,
        instance,
        repetition,
        run_id,
        *,
        solver_mode: SolverMode = SolverMode.CLASSICAL,
        solver_version: str | None = None,
    ):
        return {
            "run_id": run_id,
            "instance_id": generator.instance_id,
            "solver_mode": solver_mode,
            "solver_version": solver_version or self.configuration.solver_version,
            "seed": generator.seed,
            "config_id": self.configuration.config_id,
            "repetition": repetition,
            "environment": self.configuration.environment,
            "stock_length": instance.stock_length,
            "kerf": instance.kerf,
            "number_of_piece_types": instance.number_of_types,
            "total_demand": sum(instance.demands),
            "requested_length": sum(
                length * demand
                for length, demand in zip(instance.piece_lengths, instance.demands, strict=True)
            ),
            "length_distribution": generator.length_distribution,
            "demand_distribution": generator.demand_distribution,
        }


def _component_status(status: int | None) -> str:
    if status is None:
        return "not_run"
    return "optimal" if status == 0 else str(status)


def _run_status(status: str) -> RunStatus:
    if status == "converged":
        return RunStatus.OPTIMAL_LP_RESTRICTED_IP
    if status == "infeasible":
        return RunStatus.INFEASIBLE
    if status == "limit_reached":
        return RunStatus.TIMEOUT
    if status == "invalid_plan":
        return RunStatus.INVALID_PLAN
    return RunStatus.SOLVER_ERROR


def solve_with_peak_memory(solver) -> tuple[ColumnGenerationResult, int]:
    """Run one solver and measure its peak Python allocation footprint."""

    tracemalloc.start()
    try:
        result = solver.solve()
        _current, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak_memory_bytes


def write_raw_runs(path: str | Path, records: tuple[BenchmarkRunRecord, ...]) -> None:
    """Write every supplied raw run in canonical order without filtering."""

    fieldnames = tuple(records[0].to_dict()) if records else _raw_run_fieldnames()
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            record.to_dict() for record in sorted(records, key=lambda item: item.run_id)
        )


def _raw_run_fieldnames() -> tuple[str, ...]:
    """Return the complete flat schema for an empty raw-run table."""

    record_fields = tuple(
        field.name for field in fields(BenchmarkRunRecord) if field.name != "environment"
    )
    return record_fields + tuple(field.name for field in fields(EnvironmentMetadata))


def _generator_payload(generator: SyntheticInstanceGenerator) -> dict[str, object]:
    return {
        "seed": generator.seed,
        "stock_length": generator.stock_length,
        "kerf": generator.kerf,
        "number_of_types": generator.number_of_types,
        "piece_length_range": generator.piece_length_range,
        "demand_range": generator.demand_range,
        "length_distribution": generator.length_distribution,
        "demand_distribution": generator.demand_distribution,
    }
