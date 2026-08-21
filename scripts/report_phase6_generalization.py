"""Report out-of-distribution size generalization from persisted Phase 6 runs."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import evaluate_size_generalization, training_size_frontier
from neural_cutting_stock.visualization.phase4 import load_phase4_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument(
        "--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase-6-generalization.json"))
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frontier = training_size_frontier(root / config["partitions"]["manifest"])
    report = evaluate_size_generalization(
        load_phase4_runs(args.classical_runs),
        load_phase4_runs(args.neural_runs),
        frontier,
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
        output_path=args.output,
    )
    print(f"wrote {args.output}")
    print(
        "above-training instances:"
        f" {report['coverage']['instance_count_above_training']},"
        f" admissible pairs: {report['coverage']['admissible_pair_count_above_training']}"
    )


if __name__ == "__main__":
    main()
