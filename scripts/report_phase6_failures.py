"""Report failures, violations, fallbacks and timeouts from Phase 6 runs."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import analyze_campaign_failures
from neural_cutting_stock.visualization.phase4 import load_phase4_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument("--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-6-failures.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = analyze_campaign_failures(
        load_phase4_runs(args.classical_runs),
        load_phase4_runs(args.neural_runs),
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
        output_path=args.output,
    )
    print(f"wrote {args.output}")
    for mode in ("classical", "neural"):
        item = report["modes"][mode]
        print(
            f"{mode}: failures={item['failure_count']}, timeouts={item['timeout_count']},"
            f" plan_violations={item['plan_violation_count']}"
        )
    print(f"quality-violating pairs: {report['pairs']['quality_violation_pair_count']}")


if __name__ == "__main__":
    main()
