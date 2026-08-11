"""Reproducible validation runner for paired Classical and Neural CG runs."""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from .comparison import compare_paired_runs
from .generator import SyntheticInstanceGenerator
from .runner import ClassicalBenchmarkRunner, write_raw_runs
from .schema import BenchmarkRunRecord, EnvironmentMetadata, SolverMode


@dataclass(frozen=True, slots=True)
class PairedBenchmarkConfig:
    """Common validation configuration for both solver modes."""

    generators: tuple[SyntheticInstanceGenerator, ...]
    environment: EnvironmentMetadata
    policy: object
    model_id: str
    repetitions: int = 1
    reduced_cost_tolerance: float = 1e-9
    candidate_budget: int | None = None
    max_runtime_seconds: float | None = None
    max_cg_iterations: int | None = None
    solver_version: str = "paired-cg-v1"

    def __post_init__(self) -> None:
        if not self.generators:
            raise ValueError("generators must not be empty")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")

    @property
    def config_id(self) -> str:
        payload = {
            "generators": sorted(generator.instance_id for generator in self.generators),
            "repetitions": self.repetitions,
            "reduced_cost_tolerance": self.reduced_cost_tolerance,
            "candidate_budget": self.candidate_budget,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_cg_iterations": self.max_cg_iterations,
            "model_id": self.model_id,
            "solver_version": self.solver_version,
        }
        return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


class PairedBenchmarkRunner:
    """Execute both modes on each identical generated instance and repetition."""

    def __init__(self, configuration: PairedBenchmarkConfig) -> None:
        self.configuration = configuration

    def run(self, output_path: str | Path | None = None) -> tuple[BenchmarkRunRecord, ...]:
        classical = ClassicalBenchmarkRunner(
            _classical_config(self.configuration)
        )
        records: list[BenchmarkRunRecord] = []
        for generator in self.configuration.generators:
            instance = generator.generate()
            for repetition in range(self.configuration.repetitions):
                records.append(classical._run_one(generator, instance, repetition))
                records.append(self._run_neural(generator, instance, repetition))

        comparisons = compare_paired_runs(records)
        by_neural_id = {comparison.neural_run_id: comparison for comparison in comparisons}
        records = [
            replace(
                record,
                speedup_vs_classical=by_neural_id[record.run_id].speedup_vs_classical,
                objective_difference_vs_classical=(
                    by_neural_id[record.run_id].objective_difference_vs_classical
                ),
            )
            if record.solver_mode is SolverMode.NEURAL
            else record
            for record in records
        ]
        result = tuple(records)
        if output_path is not None:
            write_raw_runs(output_path, result)
        return result

    def _run_neural(self, generator, instance, repetition) -> BenchmarkRunRecord:
        from neural_cutting_stock.learning import NeuralColumnGeneration

        run_key = f"{self.configuration.config_id}:{generator.instance_id}:neural:{repetition}"
        run_id = hashlib.sha256(run_key.encode("ascii")).hexdigest()
        runner = ClassicalBenchmarkRunner(_classical_config(self.configuration))
        solver = NeuralColumnGeneration(
            instance,
            self.configuration.policy,
            candidate_budget=self.configuration.candidate_budget,
            reduced_cost_tolerance=self.configuration.reduced_cost_tolerance,
            max_runtime_seconds=self.configuration.max_runtime_seconds,
            max_iterations=self.configuration.max_cg_iterations,
            instance_id=generator.instance_id,
        )
        try:
            result = solver.solve()
        except Exception as error:
            return runner._failed_record(
                generator,
                instance,
                repetition,
                run_id,
                str(error),
                solver_mode=SolverMode.NEURAL,
                solver_version=self.configuration.solver_version,
                model_id=self.configuration.model_id,
            )
        return runner._record_from_result(
            generator,
            instance,
            repetition,
            run_id,
            result,
            solver_mode=SolverMode.NEURAL,
            solver_version=self.configuration.solver_version,
            model_id=self.configuration.model_id,
            neural_profile=solver.runtime_profile,
        )


def _classical_config(configuration: PairedBenchmarkConfig):
    from .runner import ClassicalBenchmarkConfig

    return ClassicalBenchmarkConfig(
        generators=configuration.generators,
        environment=configuration.environment,
        repetitions=configuration.repetitions,
        reduced_cost_tolerance=configuration.reduced_cost_tolerance,
        max_runtime_seconds=configuration.max_runtime_seconds,
        max_cg_iterations=configuration.max_cg_iterations,
        solver_version=configuration.solver_version,
    )
