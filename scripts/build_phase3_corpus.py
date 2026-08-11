"""Build the small Phase 3 trajectory corpus from its fixed generator plan."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    CORPUS_SCHEMA_VERSION,
    EnvironmentMetadata,
    PartitionPlan,
    SyntheticInstanceGenerator,
    TrajectoryMetadata,
    collect_trajectory,
    corpus_statistics,
    trajectory_sha256,
    write_trajectory,
)
from neural_cutting_stock.solver import ColumnGeneration

CODE_COMMIT = "ce3d85831e2802374a6bf18f762015dd9d49493f"
CORPUS_ID = "phase-3-small-v1"


def build(output_dir: Path, code_commit: str = CODE_COMMIT) -> dict[str, object]:
    generators = (
        SyntheticInstanceGenerator(seed=11, number_of_types=2),
        SyntheticInstanceGenerator(seed=12, number_of_types=3),
        SyntheticInstanceGenerator(seed=13, number_of_types=4),
    )
    plan = PartitionPlan(
        train_seeds=(11,),
        validation_seeds=(12,),
        test_seeds=(13,),
        train_families=(generators[0].family_id,),
        validation_families=(generators[1].family_id,),
        test_families=(generators[2].family_id,),
    )
    assignments = plan.assignments(generators)
    environment = EnvironmentMetadata(
        code_commit,
        platform.python_version(),
        "numpy,scipy",
        f"{platform.system()}-{platform.machine()}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    entries = []
    trajectories = []
    partitions = {}
    config_id = hashlib.sha256(
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    for generator, assignment in zip(generators, assignments, strict=True):
        instance = generator.generate()
        trajectory_id = hashlib.sha256(
            f"{CORPUS_ID}:{generator.instance_id}".encode("ascii")
        ).hexdigest()
        result = ColumnGeneration(instance, instance_id=generator.instance_id).solve()
        metadata = TrajectoryMetadata(
            trajectory_id,
            generator.instance_id,
            "classical-cg-v1",
            generator.seed,
            config_id,
            environment,
            instance.stock_length,
            instance.kerf,
            instance.piece_lengths,
            instance.demands,
            1e-9,
            1e-9,
            1e-9,
            instance.piece_lengths,
            1e-9,
        )
        trajectory = collect_trajectory(result, metadata).trajectory
        relative_path = f"trajectories/{trajectory_id}.json"
        write_trajectory(output_dir / relative_path, trajectory)
        trajectories.append(trajectory)
        partitions[trajectory_id] = assignment.partition
        entries.append(
            {
                "trajectory_id": trajectory_id,
                "instance_id": generator.instance_id,
                "seed": generator.seed,
                "family_id": generator.family_id,
                "partition": assignment.partition.value,
                "path": relative_path,
                "sha256": trajectory_sha256(trajectory),
                "stock_length": instance.stock_length,
                "kerf": instance.kerf,
                "piece_lengths": list(instance.piece_lengths),
                "demands": list(instance.demands),
            }
        )
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "generator": {
            "name": SyntheticInstanceGenerator.name,
            "version": SyntheticInstanceGenerator.version,
        },
        "partition_plan": plan.to_dict(),
        "environment": {
            "code_commit": environment.code_commit,
            "python_version": environment.python_version,
            "dependency_versions": environment.dependency_versions,
            "hardware_id": environment.hardware_id,
        },
        "trajectories": sorted(entries, key=lambda item: item["trajectory_id"]),
        "statistics": corpus_statistics(tuple(trajectories), partitions),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase-3-corpus"))
    parser.add_argument("--code-commit", default=CODE_COMMIT)
    args = parser.parse_args()
    build(args.output_dir, args.code_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
