"""Generate the Phase 5 decision report and figures from raw paired runs."""

import argparse
from pathlib import Path

from neural_cutting_stock.visualization.phase5 import (
    load_phase5_runs,
    phase5_report_data,
    write_phase5_figures,
    write_phase5_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = phase5_report_data(load_phase5_runs(args.runs), args.candidate_id)
    write_phase5_figures(data, args.output_dir)
    write_phase5_summary(data, args.output_dir / "phase-5-summary.md", str(args.runs))


if __name__ == "__main__":
    main()
