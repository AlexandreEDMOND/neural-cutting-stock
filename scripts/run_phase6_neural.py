"""Run and persist the final Neural CG campaign."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    PairedBenchmarkConfig,
    PairedBenchmarkRunner,
    collect_environment,
    generators_from_final_manifest,
    write_neural_campaign_metadata,
)
from neural_cutting_stock.learning import LearnedColumnSelectionPolicy, load_training_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument("--runs", type=Path, default=Path("results/phase-6-neural-runs.csv"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("results/phase-6-neural-campaign.json")
    )
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / config["files"]["final_instance_manifest"]).read_text(encoding="utf-8")
    )
    phase3_manifest = json.loads(
        (root / config["partitions"]["manifest"]).read_text(encoding="utf-8")
    )
    environment = collect_environment(root)
    generators = generators_from_final_manifest(manifest, phase3_manifest)
    protocol = config["protocol"]
    model_path = root / config["model"]["artifact"]
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    candidate_budget = artifact["config"]["candidate_budget"]
    model = load_training_artifact(model_path)
    policy = LearnedColumnSelectionPolicy(model, candidate_budget)
    benchmark_config = PairedBenchmarkConfig(
        generators=generators,
        environment=environment,
        policy=policy,
        model_id=config["model"]["model_id"],
        repetitions=protocol["repetitions"],
        reduced_cost_tolerance=protocol["reduced_cost_tolerance"],
        candidate_budget=candidate_budget,
        max_runtime_seconds=protocol["max_runtime_seconds"],
        max_cg_iterations=protocol["max_cg_iterations"],
    )
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    records = PairedBenchmarkRunner(benchmark_config).run_neural(args.runs)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    write_neural_campaign_metadata(
        args.metadata,
        config=config,
        manifest=manifest,
        environment=environment,
        benchmark_config_id=benchmark_config.config_id,
        run_count=len(records),
        candidate_budget=candidate_budget,
    )


if __name__ == "__main__":
    main()
