"""Reproducible instance generation and benchmark execution."""

from .dataset import DATASET_SCHEMA_VERSION, DatasetExample, TrajectoryDataset, build_dataset
from .generator import SyntheticInstanceGenerator
from .matrix import BenchmarkMatrix, DistributionSpec
from .partitions import (
    PARTITION_SCHEMA_VERSION,
    DatasetPartition,
    PartitionAssignment,
    PartitionPlan,
)
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
    DUAL_SIGN_CONVENTION,
    TRAJECTORY_SCHEMA_VERSION,
    ColumnGenerationTrajectory,
    TrajectoryCollectionMeasurement,
    TrajectoryIteration,
    TrajectoryMetadata,
    TrajectoryReader,
    TrajectoryStatus,
    TrajectoryValidation,
    UsefulColumnTarget,
    collect_trajectory,
    define_useful_column_target,
    read_trajectory,
    replay_trajectory,
    write_trajectory,
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
    "DATASET_SCHEMA_VERSION",
    "DatasetExample",
    "TrajectoryDataset",
    "build_dataset",
    "PARTITION_SCHEMA_VERSION",
    "DatasetPartition",
    "PartitionAssignment",
    "PartitionPlan",
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
    "DUAL_SIGN_CONVENTION",
    "ColumnGenerationTrajectory",
    "TrajectoryCollectionMeasurement",
    "TrajectoryReader",
    "TrajectoryIteration",
    "TrajectoryMetadata",
    "TrajectoryStatus",
    "TrajectoryValidation",
    "UsefulColumnTarget",
    "collect_trajectory",
    "define_useful_column_target",
    "read_trajectory",
    "replay_trajectory",
    "write_trajectory",
]
