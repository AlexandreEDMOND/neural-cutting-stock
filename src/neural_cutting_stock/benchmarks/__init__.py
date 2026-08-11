"""Reproducible instance generation and benchmark execution."""

from .generator import SyntheticInstanceGenerator
from .matrix import BenchmarkMatrix, DistributionSpec
from .runner import ClassicalBenchmarkConfig, ClassicalBenchmarkRunner, write_raw_runs
from .schema import (
    SCHEMA_VERSION,
    BenchmarkRunRecord,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
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
]
