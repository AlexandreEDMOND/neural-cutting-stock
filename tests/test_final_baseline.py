import json
from pathlib import Path

from neural_cutting_stock.benchmarks import generators_from_final_manifest

ROOT = Path(__file__).parents[1]


def test_final_baseline_reconstructs_manifest_instances_in_manifest_order() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["files"]["final_instance_manifest"]).read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())

    generators = generators_from_final_manifest(manifest, phase3)

    assert [generator.instance_id for generator in generators] == [
        entry["instance_id"] for entry in manifest["instances"]
    ]
    assert [generator.seed for generator in generators] == [
        entry["seed"] for entry in manifest["instances"]
    ]
