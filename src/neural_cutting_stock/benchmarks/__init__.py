"""Reproducible instance generation and benchmark execution."""

from .generator import SyntheticInstanceGenerator
from .matrix import BenchmarkMatrix, DistributionSpec
from .profile import (
    PROFILE_SCHEMA_VERSION,
    SIZE_CLASS_RUNTIME_THRESHOLDS_SECONDS,
    SIZE_CLASS_SCHEMA_VERSION,
    SizeClass,
    classify_runtime,
    profile_classical_runs,
)
from .runner import ClassicalBenchmarkConfig, ClassicalBenchmarkRunner, write_raw_runs
from .schema import (
    SCHEMA_VERSION,
    BenchmarkRunRecord,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
)
from .trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    ColumnGenerationTrajectory,
    TrajectoryIteration,
    TrajectoryMetadata,
    TrajectoryStatus,
)

__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkRunRecord",
    "EnvironmentMetadata",
    "RunStatus",
    "SolverMode",
    "SyntheticInstanceGenerator",
    "BenchmarkMatrix",
    "DistributionSpec",
    "ClassicalBenchmarkConfig",
    "ClassicalBenchmarkRunner",
    "write_raw_runs",
    "PROFILE_SCHEMA_VERSION",
    "profile_classical_runs",
    "SIZE_CLASS_SCHEMA_VERSION",
    "SIZE_CLASS_RUNTIME_THRESHOLDS_SECONDS",
    "SizeClass",
    "classify_runtime",
    "TRAJECTORY_SCHEMA_VERSION",
    "ColumnGenerationTrajectory",
    "TrajectoryIteration",
    "TrajectoryMetadata",
    "TrajectoryStatus",
]
