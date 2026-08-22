"""Generate the final runtime and speedup figures from validated Phase 6 runs."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import validate_final_manifest
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import (
    phase6_runtime_comparison_data,
    write_phase6_runtime_comparison,
    write_phase6_speedup_by_size,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument("--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["files"]["final_instance_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase3_manifest = json.loads(
        Path(config["partitions"]["manifest"]).read_text(encoding="utf-8")
    )
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_phase6_runtime_comparison(data, args.output_dir)
    write_phase6_speedup_by_size(data, args.output_dir)
    print(f"wrote {args.output_dir / 'runtime_comparison.png'}")
    print(f"wrote {args.output_dir / 'speedup_by_size.png'}")
    report = data["report"]
    print(f"pairs: {report['pair_count']}, admissible: {report['admissible_pair_count']}")


if __name__ == "__main__":
    main()
