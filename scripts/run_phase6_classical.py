"""Run and persist the final Classical CG baseline."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    ClassicalBenchmarkConfig,
    ClassicalBenchmarkRunner,
    collect_environment,
    generators_from_final_manifest,
    write_campaign_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument("--runs", type=Path, default=Path("results/phase-6-classical-runs.csv"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("results/phase-6-classical-campaign.json")
    )
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = root / config["files"]["final_instance_manifest"]
    phase3_path = root / config["partitions"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase3_manifest = json.loads(phase3_path.read_text(encoding="utf-8"))
    environment = collect_environment(root)
    generators = generators_from_final_manifest(manifest, phase3_manifest)
    protocol = config["protocol"]
    benchmark_config = ClassicalBenchmarkConfig(
        generators=generators,
        environment=environment,
        repetitions=protocol["repetitions"],
        reduced_cost_tolerance=protocol["reduced_cost_tolerance"],
        max_runtime_seconds=protocol["max_runtime_seconds"],
        max_cg_iterations=protocol["max_cg_iterations"],
    )
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    records = ClassicalBenchmarkRunner(benchmark_config).run(args.runs)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    write_campaign_metadata(
        args.metadata,
        config=config,
        manifest=manifest,
        environment=environment,
        benchmark_config_id=benchmark_config.config_id,
        run_count=len(records),
    )


if __name__ == "__main__":
    main()
