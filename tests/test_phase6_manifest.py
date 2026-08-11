import json
from copy import deepcopy
from pathlib import Path

import pytest

from neural_cutting_stock.benchmarks import (
    FINAL_MANIFEST_SCHEMA_VERSION,
    SIZE_CLASSES,
    build_final_manifest,
    validate_final_manifest,
)

ROOT = Path(__file__).parents[1]


def test_final_manifest_is_materialized_and_disjoint_from_phase3() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())
    manifest = json.loads((ROOT / config["files"]["final_instance_manifest"]).read_text())

    validate_final_manifest(manifest, phase3)
    assert manifest["schema_version"] == FINAL_MANIFEST_SCHEMA_VERSION
    assert manifest["statistics"]["target_size_class_counts"] == dict.fromkeys(SIZE_CLASSES, 3)
    assert all(entry["target_size_class"] in SIZE_CLASSES for entry in manifest["instances"])


def test_final_manifest_generation_is_reproducible() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())

    first = build_final_manifest(tuple(config["final_instances"]), phase3)
    second = build_final_manifest(tuple(config["final_instances"]), phase3)

    assert first == second


def test_final_manifest_rejects_materialization_changes() -> None:
    config = json.loads((ROOT / "configs/phase-6-final.json").read_text())
    phase3 = json.loads((ROOT / config["partitions"]["manifest"]).read_text())
    manifest = json.loads((ROOT / config["files"]["final_instance_manifest"]).read_text())
    changed = deepcopy(manifest)
    changed["instances"][0]["demands"][0] += 1

    with pytest.raises(ValueError, match="materialized data"):
        validate_final_manifest(changed, phase3)
