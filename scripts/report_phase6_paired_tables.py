"""Produce paired quality, runtime, memory, iteration and column tables."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import build_paired_tables
from neural_cutting_stock.visualization.phase4 import load_phase4_runs
from neural_cutting_stock.visualization.phase6 import write_paired_tables_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument(
        "--classical-runs", type=Path, default=Path("results/phase-6-classical-runs.csv")
    )
    parser.add_argument(
        "--neural-runs", type=Path, default=Path("results/phase-6-neural-runs.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_paired_tables(
        load_phase4_runs(args.classical_runs),
        load_phase4_runs(args.neural_runs),
        quality_tolerance=config["protocol"]["quality_tolerance_bars"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase-6-paired-tables.json"
    markdown_path = args.output_dir / "phase-6-paired-tables.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_paired_tables_markdown(
        report, markdown_path, str(args.classical_runs), str(args.neural_runs)
    )
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        f"pairs: {report['pair_count']}, admissible: {report['admissible_pair_count']},"
        f" instances: {len(report['instances'])}"
    )


if __name__ == "__main__":
    main()
