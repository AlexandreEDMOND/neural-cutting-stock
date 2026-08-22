"""Frozen train/validation/test partitions of the retained Phase 8 families.

Building crosses every measured cell of a retained ``family-margins-v1``
family with its declared generator and assigns it by seed alone, so no seed
is ever shared between two partitions. Validation re-materializes every cell
from its recorded configuration alone, keeping the frozen manifest checkable
independently of the live specs. A regression pins the committed manifest to
the committed measurement.
"""

import copy
import json
from pathlib import Path

import pytest

from neural_cutting_stock.benchmarks import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    QUALITY_PARTITIONS_SCHEMA_VERSION,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    build_quality_partition_plan,
    phase8_family_specs,
    read_quality_partition_manifest,
    validate_quality_partition_manifest,
    write_quality_partition_manifest,
)

ROOT = Path(__file__).parents[1]
MARGINS_PATH = ROOT / "results/phase-8-family-margins.json"
MANIFEST_PATH = ROOT / "data/phase-8-partitions/manifest.json"

FAMILY = "structured-tight-divisibility-t3-v1"
NOT_RETAINED = "structured-tight-divisibility-t4-v1"


def _spec_of(label: str):
    return {spec.family_label: spec for spec in phase8_family_specs()}[label]


def _fabricated_report() -> dict:
    """A minimal but faithful family-margins-v1 payload built on real cells."""

    retained = _spec_of(FAMILY)
    rejected = _spec_of(NOT_RETAINED)
    instances = []
    for generator in retained.generators:
        instances.append(
            {
                "family_label": FAMILY,
                "seed": generator.seed,
                "instance_id": generator.instance_id,
                "gap_available": True,
                "gap_bars": 1,
            }
        )
    for generator in rejected.generators:
        instances.append(
            {
                "family_label": NOT_RETAINED,
                "seed": generator.seed,
                "instance_id": generator.instance_id,
                "gap_available": True,
                "gap_bars": 0,
            }
        )
    return {
        "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
        "significant_positive_share": 0.5,
        "reference_method_limits": "maximal_patterns:test-limits",
        "families": [
            {
                "family_label": NOT_RETAINED,
                "configuration": rejected.configuration,
                "retained": False,
            },
            {
                "family_label": FAMILY,
                "configuration": retained.configuration,
                "retained": True,
            },
        ],
        "instances": instances,
    }


def _tampered(manifest: dict, mutation) -> dict:
    target = copy.deepcopy(manifest)
    mutation(target)
    return target


def test_frozen_seed_split_is_declared_disjoint_and_non_empty() -> None:
    assert QUALITY_PARTITIONS_SCHEMA_VERSION == "phase-8-quality-partitions-v1"
    assert TRAIN_SEEDS == (1, 2, 3)
    assert VALIDATION_SEEDS == (4,)
    assert TEST_SEEDS == (5, 6)
    assert not ({*TRAIN_SEEDS} & {*VALIDATION_SEEDS})
    assert not ({*TRAIN_SEEDS} | {*VALIDATION_SEEDS}) & {*TEST_SEEDS}


def test_build_keeps_only_retained_families_and_assigns_by_frozen_seeds() -> None:
    manifest = build_quality_partition_plan(_fabricated_report())

    assert manifest["schema_version"] == QUALITY_PARTITIONS_SCHEMA_VERSION
    assert [family["family_label"] for family in manifest["families"]] == [FAMILY]
    assert manifest["source"]["schema_version"] == FAMILY_MARGINS_SCHEMA_VERSION
    assert manifest["seed_partitions"] == {
        "train": [1, 2, 3],
        "validation": [4],
        "test": [5, 6],
    }

    partition_by_seed = {
        **{seed: "train" for seed in TRAIN_SEEDS},
        **{seed: "validation" for seed in VALIDATION_SEEDS},
        **{seed: "test" for seed in TEST_SEEDS},
    }
    generators = {generator.seed: generator for generator in _spec_of(FAMILY).generators}
    cells = manifest["assignments"]
    assert {(cell["family_label"], cell["seed"]) for cell in cells} == {
        (FAMILY, seed) for seed in generators
    }
    for cell in cells:
        assert cell["partition"] == partition_by_seed[cell["seed"]]
        assert cell["instance_id"] == generators[cell["seed"]].instance_id

    assert manifest["statistics"] == {
        "instance_count": 6,
        "family_count": 1,
        "partition_instance_counts": {"train": 3, "validation": 1, "test": 2},
        "partition_family_counts": {"train": 1, "validation": 1, "test": 1},
        "seeds_per_partition": {"train": 3, "validation": 1, "test": 2},
    }


def test_build_is_deterministic_including_the_plan_hash() -> None:
    first = build_quality_partition_plan(_fabricated_report())
    second = build_quality_partition_plan(_fabricated_report())

    assert first == second
    assert len(first["plan_id"]) == 64


def test_build_rejects_unusable_or_drifting_reports() -> None:
    report = _fabricated_report()

    with pytest.raises(ValueError, match="unsupported margin report"):
        build_quality_partition_plan({**report, "schema_version": "family-margins-v0"})
    with pytest.raises(ValueError, match="documented retention share"):
        build_quality_partition_plan({**report, "significant_positive_share": 0.9})

    nothing_retained = {
        **report,
        "families": [{**family, "retained": False} for family in report["families"]],
    }
    with pytest.raises(ValueError, match="retains no family"):
        build_quality_partition_plan(nothing_retained)

    ghost = [
        {**family, "family_label": "ghost-family", "retained": True}
        if family["family_label"] == FAMILY
        else family
        for family in report["families"]
    ]
    with pytest.raises(ValueError, match="without a declared spec"):
        build_quality_partition_plan({**report, "families": ghost})

    drifted_configuration = [
        {**family, "configuration": {**family["configuration"], "kerf": 1.0}}
        if family["family_label"] == FAMILY
        else family
        for family in report["families"]
    ]
    with pytest.raises(ValueError, match="configuration drift"):
        build_quality_partition_plan({**report, "families": drifted_configuration})

    instances = [dict(entry) for entry in report["instances"]]
    instances[0]["seed"] = 99
    with pytest.raises(ValueError, match="measured cells differ from declared cells"):
        build_quality_partition_plan({**report, "instances": instances})

    instances = [dict(entry) for entry in report["instances"]]
    instances[0]["instance_id"] = "0" * 64
    with pytest.raises(ValueError, match="instance_id drift"):
        build_quality_partition_plan({**report, "instances": instances})

    instances = [dict(entry) for entry in report["instances"]]
    instances[0]["gap_available"] = False
    with pytest.raises(ValueError, match="unavailable gap"):
        build_quality_partition_plan({**report, "instances": instances})


def test_validation_is_self_sufficient_and_loud_about_tampering() -> None:
    manifest = build_quality_partition_plan(_fabricated_report())
    validate_quality_partition_manifest(json.loads(json.dumps(manifest)))

    def drop_one_cell(target: dict) -> None:
        target["assignments"].pop()

    with pytest.raises(ValueError, match="missing from the plan"):
        validate_quality_partition_manifest(_tampered(manifest, drop_one_cell))

    def duplicate_a_cell(target: dict) -> None:
        target["assignments"].append(dict(target["assignments"][0]))

    with pytest.raises(ValueError, match="assigned more than once"):
        validate_quality_partition_manifest(_tampered(manifest, duplicate_a_cell))

    def move_seed_out_of_its_partition(target: dict) -> None:
        target["assignments"][0]["partition"] = "validation"

    with pytest.raises(ValueError, match="not frozen into partition"):
        validate_quality_partition_manifest(_tampered(manifest, move_seed_out_of_its_partition))

    def forge_an_instance_id(target: dict) -> None:
        target["assignments"][0]["instance_id"] = "f" * 64

    with pytest.raises(ValueError, match="does not match the recorded configuration"):
        validate_quality_partition_manifest(_tampered(manifest, forge_an_instance_id))

    def record_an_unknown_family(target: dict) -> None:
        target["assignments"].append(
            {
                "partition": "train",
                "family_label": "ghost",
                "seed": 1,
                "instance_id": "a" * 64,
            }
        )
        target["plan_id"] = "b" * 64
        target["statistics"] = {"instance_count": 7}

    with pytest.raises(ValueError, match="unrecorded family|unrecognized recorded"):
        validate_quality_partition_manifest(_tampered(manifest, record_an_unknown_family))

    def add_a_stray_configuration_field(target: dict) -> None:
        family = next(item for item in target["families"] if item["family_label"] == FAMILY)
        family["configuration"]["stock_lengths"] = [50.0]

    with pytest.raises(ValueError, match="configuration for"):
        validate_quality_partition_manifest(_tampered(manifest, add_a_stray_configuration_field))

    def alter_the_frozen_split(target: dict) -> None:
        target["seed_partitions"]["train"] = [1, 2]

    with pytest.raises(ValueError, match="frozen Phase 8 split"):
        validate_quality_partition_manifest(_tampered(manifest, alter_the_frozen_split))

    with pytest.raises(ValueError, match="unsupported quality partition manifest"):
        validate_quality_partition_manifest({**manifest, "schema_version": "quality-partitions-v0"})


def test_multi_format_configurations_materialize_from_the_recording(tmp_path) -> None:
    report = _fabricated_report()
    multi = _spec_of("multi-stock-formats-t4-v1")
    report["families"] = [
        {
            "family_label": multi.family_label,
            "configuration": multi.configuration,
            "retained": True,
        }
    ]
    report["instances"] = [
        {
            "family_label": multi.family_label,
            "seed": generator.seed,
            "instance_id": generator.instance_id,
            "gap_available": True,
            "gap_bars": 2,
        }
        for generator in multi.generators
    ]

    manifest = build_quality_partition_plan(report)
    path = tmp_path / "nested" / "manifest.json"
    write_quality_partition_manifest(path, manifest)
    reloaded = read_quality_partition_manifest(path)

    assert reloaded == manifest
    assert reloaded["families"][0]["configuration"]["stock_lengths"] == [50.0, 100.0]


def test_committed_manifest_matches_the_committed_measurement() -> None:
    margins_report = json.loads(MARGINS_PATH.read_text(encoding="utf-8"))
    rebuilt = build_quality_partition_plan(margins_report)
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert rebuilt == committed
    labels = sorted(family["family_label"] for family in committed["families"])
    assert labels == [
        "scaled-tight-divisibility-t12-v1",
        "structured-tight-divisibility-t3-v1",
        "structured-tight-divisibility-t4-v1",
    ]
    assert committed["statistics"]["partition_instance_counts"] == {
        "train": 9,
        "validation": 3,
        "test": 6,
    }
    assert committed["statistics"]["partition_family_counts"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }

    cells = committed["assignments"]
    seen_cells = [(item["family_label"], item["seed"]) for item in cells]
    instance_ids = [item["instance_id"] for item in cells]
    assert len(set(seen_cells)) == len(cells) == len(set(instance_ids)) == 18
    for label in labels:
        for partition in ("train", "validation", "test"):
            present = any(
                item["family_label"] == label and item["partition"] == partition for item in cells
            )
            assert present, f"{label} must populate the {partition} partition"
