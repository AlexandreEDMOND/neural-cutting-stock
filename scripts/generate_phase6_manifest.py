"""Generate the frozen Phase 6 instance manifest from its configuration."""

import argparse
import json
from pathlib import Path

from neural_cutting_stock.benchmarks import build_final_manifest, write_final_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase-6-final.json"))
    parser.add_argument("--output", type=Path, default=Path("data/phase-6-final/manifest.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    phase3_path = Path(config["partitions"]["manifest"])
    phase3_manifest = json.loads(phase3_path.read_text(encoding="utf-8"))
    manifest = build_final_manifest(tuple(config["final_instances"]), phase3_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_final_manifest(args.output, manifest)


if __name__ == "__main__":
    main()
