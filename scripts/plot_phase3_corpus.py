"""Generate the persisted-data figures and Markdown report for Phase 3."""

import argparse
from pathlib import Path

from neural_cutting_stock.visualization.phase3 import (
    load_phase3_corpus,
    write_phase3_figures,
    write_phase3_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, trajectories = load_phase3_corpus(args.manifest)
    write_phase3_figures(manifest, trajectories, args.output_dir)
    write_phase3_summary(manifest, trajectories, args.output_dir / "phase-3-summary.md")


if __name__ == "__main__":
    main()
