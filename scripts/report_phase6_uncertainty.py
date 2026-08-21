"""Report uncertainty from persisted Phase 6 paired raw runs."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import summarize_runtime_variability
from neural_cutting_stock.visualization.phase4 import load_phase4_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument(
        "--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("results/phase-6-uncertainty.json"))
    args = parser.parse_args()
    report = {
        "schema_version": "phase-6-runtime-uncertainty-v1",
        "classical": summarize_runtime_variability(load_phase4_runs(args.classical_runs)),
        "neural": summarize_runtime_variability(load_phase4_runs(args.neural_runs)),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
