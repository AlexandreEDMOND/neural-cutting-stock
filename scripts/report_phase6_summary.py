"""Produce the final Phase 6 publication summary from validated results."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import validate_final_manifest
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import (
    phase6_runtime_comparison_data,
    write_phase6_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument(
        "--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase-6-summary.md"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["files"]["final_instance_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase3_manifest = json.loads(Path(config["partitions"]["manifest"]).read_text(encoding="utf-8"))
    validate_final_manifest(manifest, phase3_manifest)
    size_class_by_instance = {
        entry["instance_id"]: entry["target_size_class"] for entry in manifest["instances"]
    }
    data = phase6_runtime_comparison_data(
        load_phase4_runs(args.classical_runs),
        load_phase4_runs(args.neural_runs),
        size_class_by_instance,
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_phase6_summary(
        data,
        args.output,
        classical_source=str(args.classical_runs),
        neural_source=str(args.neural_runs),
        config_path=str(args.config),
        config=config,
        manifest=manifest,
        final_manifest_path=config["files"]["final_instance_manifest"],
        model_artifact_path=config["model"]["artifact"],
    )
    report = data["report"]
    print(f"wrote {args.output}")
    print(
        f"runs: {report['run_count']}, pairs: {report['pair_count']},"
        f" admissible: {report['admissible_pair_count']}"
    )


if __name__ == "__main__":
    main()
