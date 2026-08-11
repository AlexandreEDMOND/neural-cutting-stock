"""Generate the validated Phase 4 report and figures."""

import argparse

from neural_cutting_stock.visualization.phase4 import (
    load_phase4_runs,
    phase4_report_data,
    write_phase4_figures,
    write_phase4_summary,
)

parser = argparse.ArgumentParser()
parser.add_argument("--runs", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

data = phase4_report_data(load_phase4_runs(args.runs))
write_phase4_figures(data, args.output_dir)
write_phase4_summary(data, f"{args.output_dir}/phase-4-summary.md", args.runs)
