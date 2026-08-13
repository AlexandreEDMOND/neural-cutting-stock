import json
from pathlib import Path

from neural_cutting_stock.benchmarks import (
    EnvironmentMetadata,
    PairedBenchmarkConfig,
    PairedBenchmarkRunner,
    generators_from_final_manifest,
)
from neural_cutting_stock.learning import LearnedColumnSelectionPolicy, load_training_artifact

ROOT = Path(__file__).parents[1]


def test_final_baseline_reconstructs_manifest_instances_in_manifest_order() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["files"]["final_instance_manifest"]).read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())

    generators = generators_from_final_manifest(manifest, phase3)

    assert [generator.instance_id for generator in generators] == [
        entry["instance_id"] for entry in manifest["instances"]
    ]
    assert [generator.seed for generator in generators] == [
        entry["seed"] for entry in manifest["instances"]
    ]


def test_neural_campaign_runner_can_persist_neural_records_only(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["files"]["final_instance_manifest"]).read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())
    generators = generators_from_final_manifest(manifest, phase3)[:1]
    protocol = config["protocol"]
    policy = LearnedColumnSelectionPolicy(
        load_training_artifact(ROOT / config["model"]["artifact"]),
        config["model"].get("candidate_budget"),
    )
    benchmark_config = PairedBenchmarkConfig(
        generators=generators,
        environment=EnvironmentMetadata("a" * 40, "3.11", "numpy==test", "test"),
        policy=policy,
        model_id=config["model"]["model_id"],
        repetitions=1,
        reduced_cost_tolerance=protocol["reduced_cost_tolerance"],
        candidate_budget=config["model"].get("candidate_budget"),
    )

    records = PairedBenchmarkRunner(benchmark_config).run_neural(tmp_path / "runs.csv")

    assert len(records) == 1
    assert records[0].solver_mode.value == "neural"
    assert records[0].instance_id == generators[0].instance_id
    assert (tmp_path / "runs.csv").is_file()
