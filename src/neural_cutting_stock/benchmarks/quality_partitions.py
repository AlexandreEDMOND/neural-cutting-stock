"""Frozen train/validation/test partitions of the retained Phase 8 families.

The retained families come from the persisted ``family-margins-v1`` report,
never from an opinion: a family enters the plan only when its measured summary
is marked retained, and building crosses every measured cell with its declared
generator before accepting it. The partition unit is the measured cell
``(family_label, seed)`` and the seed alone decides the partition, so no seed
— and therefore no random draw — is ever shared between two partitions.
Validation is self-sufficient: each family records its full configuration and
every cell re-materializes its declared ``instance_id`` from that recording,
so the frozen manifest stays checkable even if the canonical specs evolve.
Every materialized ``instance_id`` must be globally unique and each partition
must receive at least one family, which makes any leakage or silent drop loud.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_cutting_stock.problem import AnyCuttingStockInstance

from ._validation import require_text as _require_text
from .family_margins import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    SIGNIFICANT_POSITIVE_SHARE,
    phase8_family_specs,
)
from .generator import (
    GENERATOR_CONFIGURATION_FIELDS,
    MultiFormatSyntheticGenerator,
    SyntheticInstanceGenerator,
)
from .partitions import DatasetPartition

QUALITY_PARTITIONS_SCHEMA_VERSION = "phase-8-quality-partitions-v1"

TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3)
VALIDATION_SEEDS: tuple[int, ...] = (4,)
TEST_SEEDS: tuple[int, ...] = (5, 6)

SEED_PARTITIONS: dict[DatasetPartition, tuple[int, ...]] = {
    DatasetPartition.TRAIN: TRAIN_SEEDS,
    DatasetPartition.VALIDATION: VALIDATION_SEEDS,
    DatasetPartition.TEST: TEST_SEEDS,
}

_PARTITION_ORDER = (
    DatasetPartition.TRAIN,
    DatasetPartition.VALIDATION,
    DatasetPartition.TEST,
)


def build_quality_partition_plan(margins_report: dict[str, Any]) -> dict[str, Any]:
    """Freeze the leakage-safe partitions of every family retained by the report.

    The source must be a persisted ``family-margins-v1`` measurement. The
    returned manifest is deterministic: identical inputs produce an identical
    manifest including its ``plan_id`` hash.
    """

    if not isinstance(margins_report, dict):
        raise ValueError("the margin report must be a mapping")
    if margins_report.get("schema_version") != FAMILY_MARGINS_SCHEMA_VERSION:
        raise ValueError(f"unsupported margin report: {margins_report.get('schema_version')!r}")
    share = margins_report.get("significant_positive_share")
    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or float(share) != SIGNIFICANT_POSITIVE_SHARE
    ):
        raise ValueError("the margin report must declare the documented retention share")
    limits = margins_report.get("reference_method_limits")
    if not isinstance(limits, str) or not limits.strip():
        raise ValueError("the margin report must record the reference method limits")

    summaries = margins_report.get("families")
    instances = margins_report.get("instances")
    if not isinstance(summaries, list) or not isinstance(instances, list):
        raise ValueError("the margin report must contain families and instances")
    retained = sorted(
        summary["family_label"]
        for summary in summaries
        if isinstance(summary, dict) and summary.get("retained") is True
    )
    if not retained:
        raise ValueError("the margin report retains no family to partition")

    generators_by_family = {
        spec.family_label: {generator.seed: generator for generator in spec.generators}
        for spec in phase8_family_specs()
    }
    configurations: dict[str, Any] = {}
    assignments: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    for family_label in retained:
        spec_cells = generators_by_family.get(family_label)
        if spec_cells is None:
            raise ValueError(f"retained family without a declared spec: {family_label!r}")
        configuration = _configuration_of_summary(summaries, family_label)
        _require_same_configuration(family_label, configuration, next(iter(spec_cells.values())))
        configurations[family_label] = json.loads(json.dumps(configuration))
        entries = _entries_of_family(instances, family_label)
        if set(entries) != set(spec_cells):
            raise ValueError(
                f"measured cells differ from declared cells for {family_label!r}: "
                f"measured seeds {sorted(entries)}, declared seeds {sorted(spec_cells)}"
            )
        for seed, generator in sorted(spec_cells.items()):
            entry = entries[seed]
            if entry.get("gap_available") is not True:
                raise ValueError(
                    f"retained family {family_label!r} holds an unavailable gap at seed {seed}"
                )
            instance_id = entry.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError(
                    f"retained family {family_label!r} lacks an instance_id at seed {seed}"
                )
            if instance_id != generator.instance_id:
                raise ValueError(
                    f"instance_id drift for {family_label!r} at seed {seed}: "
                    "the declared generator no longer reproduces the measured instance"
                )
            if instance_id in seen_instance_ids:
                raise ValueError(
                    f"instance {instance_id[:12]}… appears more than once across families"
                )
            seen_instance_ids.add(instance_id)
            assignments.append(
                {
                    "partition": _partition_of_seed(seed).value,
                    "family_label": family_label,
                    "seed": seed,
                    "instance_id": instance_id,
                }
            )

    assignments.sort(key=lambda item: (item["partition"], item["family_label"], item["seed"]))
    families_payload = [
        {"family_label": label, "configuration": configurations[label]} for label in retained
    ]
    seed_partitions_payload = {
        partition.value: list(SEED_PARTITIONS[partition]) for partition in _PARTITION_ORDER
    }
    manifest: dict[str, Any] = {
        "schema_version": QUALITY_PARTITIONS_SCHEMA_VERSION,
        "plan_id": _plan_id(seed_partitions_payload, families_payload, assignments),
        "seed_partitions": seed_partitions_payload,
        "source": {
            "schema_version": FAMILY_MARGINS_SCHEMA_VERSION,
            "significant_positive_share": float(share),
            "reference_method_limits": limits,
        },
        "families": families_payload,
        "assignments": assignments,
        "statistics": _statistics(seed_partitions_payload, families_payload, assignments),
    }
    validate_quality_partition_manifest(manifest)
    return manifest


def validate_quality_partition_manifest(manifest: object) -> None:
    """Validate structure, coverage, materialization and absence of leakage."""

    if not isinstance(manifest, dict):
        raise ValueError("the quality partition manifest must be a mapping")
    if manifest.get("schema_version") != QUALITY_PARTITIONS_SCHEMA_VERSION:
        raise ValueError("unsupported quality partition manifest")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != FAMILY_MARGINS_SCHEMA_VERSION
        or source.get("significant_positive_share") != SIGNIFICANT_POSITIVE_SHARE
    ):
        raise ValueError("the manifest must cite the family-margins-v1 retention source")

    seed_partitions = manifest.get("seed_partitions")
    if not isinstance(seed_partitions, dict) or set(seed_partitions) != {
        partition.value for partition in _PARTITION_ORDER
    }:
        raise ValueError("the manifest must declare train, validation and test seed sets")
    seen_seeds: set[int] = set()
    for value in seed_partitions.values():
        if not isinstance(value, list) or not value:
            raise ValueError("each partition needs a non-empty seed set")
        for seed in value:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen_seeds:
                raise ValueError("partition seed sets must hold distinct integers")
            seen_seeds.add(seed)
    if seed_partitions != {
        partition.value: list(SEED_PARTITIONS[partition]) for partition in _PARTITION_ORDER
    }:
        raise ValueError("seed_partitions differ from the frozen Phase 8 split")

    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("the manifest must retain at least one family")
    recorded_configurations: dict[str, Any] = {}
    for family in families:
        if (
            not isinstance(family, dict)
            or not isinstance(family.get("family_label"), str)
            or not isinstance(family.get("configuration"), dict)
        ):
            raise ValueError("each retained family needs a label and a configuration")
        label = family["family_label"]
        if label in recorded_configurations:
            raise ValueError(f"duplicate retained family: {label!r}")
        recorded_configurations[label] = family["configuration"]

    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("the manifest must contain assignments")
    covered: set[tuple[str, int]] = set()
    instance_ids: set[str] = set()
    partition_families: dict[str, set[str]] = {
        partition.value: set() for partition in _PARTITION_ORDER
    }
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("each assignment must be a mapping")
        try:
            partition_value = assignment["partition"]
            label = assignment["family_label"]
            seed = assignment["seed"]
            instance_id = assignment["instance_id"]
        except KeyError as error:
            raise ValueError(f"incomplete assignment: missing {error.args[0]}") from error
        _require_text("family_label", label)
        if isinstance(label, str) and label not in recorded_configurations:
            raise ValueError(f"assignment of an unrecorded family: {label!r}")
        if partition_value not in seed_partitions:
            raise ValueError(f"unknown partition in assignment: {partition_value!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("assignment seeds must be integers")
        if seed not in seed_partitions[partition_value]:
            raise ValueError(f"seed {seed} is not frozen into partition {partition_value!r}")
        _require_text("instance_id", instance_id)
        key = (label, seed)
        if key in covered:
            raise ValueError(f"cell ({label!r}, {seed}) assigned more than once")
        covered.add(key)
        generator = _generator_from_record(label, recorded_configurations[label], seed)
        if generator.instance_id != instance_id:
            raise ValueError(
                f"instance_id does not match the recorded configuration for ({label!r}, {seed})"
            )
        if instance_id in instance_ids:
            raise ValueError(f"instance {instance_id[:12]}… leaks across partitions")
        instance_ids.add(instance_id)
        partition_families[partition_value].add(label)

    expected_cells = {(label, seed) for label in recorded_configurations for seed in seen_seeds}
    if covered != expected_cells:
        missing = sorted(expected_cells - covered)
        raise ValueError(f"declared family cells are missing from the plan: {missing}")
    for value in partition_families.values():
        if not value:
            raise ValueError("each partition must receive at least one retained family")

    if manifest.get("statistics") != _statistics(seed_partitions, families, assignments):
        raise ValueError("manifest statistics differ from its assignments")
    if manifest.get("plan_id") != _plan_id(seed_partitions, families, assignments):
        raise ValueError("plan_id does not match the frozen content")


def read_quality_partition_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a versioned quality partition manifest."""

    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_quality_partition_manifest(manifest)
    return manifest


def materialize_partition_instances(
    manifest: dict[str, Any], partition: str
) -> dict[str, AnyCuttingStockInstance]:
    """Re-materialize every instance assigned to one frozen partition.

    The manifest must be valid, and each assignment of ``partition`` is
    rebuilt through its recorded family configuration and seed exactly like
    plan validation does, so downstream consumers can never drift away from
    the frozen cells. Instances are keyed by their declared ``instance_id``.
    """

    validate_quality_partition_manifest(manifest)
    seed_partitions = manifest["seed_partitions"]
    if not isinstance(partition, str) or partition not in seed_partitions:
        raise ValueError(f"unknown partition: {partition!r}")
    configurations = {
        family["family_label"]: family["configuration"] for family in manifest["families"]
    }
    instances: dict[str, AnyCuttingStockInstance] = {}
    for assignment in manifest["assignments"]:
        if assignment["partition"] != partition:
            continue
        generator = _generator_from_record(
            assignment["family_label"],
            configurations[assignment["family_label"]],
            assignment["seed"],
        )
        if generator.instance_id != assignment["instance_id"]:
            raise ValueError(
                f"instance_id drift for ({assignment['family_label']!r}, {assignment['seed']})"
            )
        instances[assignment["instance_id"]] = generator.generate()
    if not instances:
        raise ValueError(f"partition {partition!r} holds no assigned instance")
    return instances


def write_quality_partition_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Validate then persist a canonical quality partition manifest."""

    validate_quality_partition_manifest(manifest)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _generator_from_record(family_label: str, configuration: dict[str, Any], seed: int) -> Any:
    """Re-materialize the declared generator of one cell from its recording."""

    fields = set(configuration)
    if fields == set(GENERATOR_CONFIGURATION_FIELDS):
        try:
            generator = SyntheticInstanceGenerator(
                seed=seed,
                stock_length=configuration["stock_length"],
                kerf=configuration["kerf"],
                number_of_types=configuration["number_of_types"],
                piece_length_range=_pair(configuration["piece_length_range"]),
                demand_range=_pair(configuration["demand_range"]),
                length_distribution=configuration["length_distribution"],
                demand_distribution=configuration["demand_distribution"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid recorded configuration for family {family_label!r}: {error}"
            ) from error
    elif "stock_lengths" in fields:
        expected = {
            "stock_lengths",
            "kerf",
            "number_of_types",
            "piece_length_range",
            "demand_range",
        }
        if fields != expected or not isinstance(configuration["stock_lengths"], list):
            raise ValueError(f"invalid multi-format configuration for {family_label!r}")
        try:
            generator = MultiFormatSyntheticGenerator(
                seed=seed,
                stock_lengths=tuple(configuration["stock_lengths"]),
                kerf=configuration["kerf"],
                number_of_types=configuration["number_of_types"],
                piece_length_range=_pair(configuration["piece_length_range"]),
                demand_range=_pair(configuration["demand_range"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid recorded configuration for family {family_label!r}: {error}"
            ) from error
    else:
        raise ValueError(f"unrecognized recorded configuration for {family_label!r}")
    if _canonical(configuration) != _canonical(generator.configuration):
        raise ValueError(f"recorded configuration is not canonical for family {family_label!r}")
    return generator


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, list) and len(value) == 2:
        return (value[0], value[1])
    if isinstance(value, tuple):
        return value
    raise ValueError("range fields must be pairs")


def _partition_of_seed(seed: int) -> DatasetPartition:
    matches = [partition for partition, seeds in SEED_PARTITIONS.items() if seed in seeds]
    if len(matches) != 1:
        raise ValueError(f"no frozen partition holds seed {seed}")
    return matches[0]


def _entries_of_family(instances: list[Any], family_label: str) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    for entry in instances:
        if not isinstance(entry, dict) or entry.get("family_label") != family_label:
            continue
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"measured cell of {family_label!r} lacks an integer seed")
        if seed in entries:
            raise ValueError(f"family {family_label!r} measures seed {seed} twice")
        entries[seed] = entry
    return entries


def _configuration_of_summary(summaries: list[Any], family_label: str) -> dict[str, Any]:
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("family_label") == family_label:
            configuration = summary.get("configuration")
            if not isinstance(configuration, dict) or not configuration:
                raise ValueError(f"retained family {family_label!r} lacks a configuration")
            return configuration
    raise ValueError(f"the margin report lacks the summary of {family_label!r}")


def _require_same_configuration(
    family_label: str, configuration: dict[str, Any], generator: Any
) -> None:
    if _canonical(configuration) != _canonical(generator.configuration):
        raise ValueError(f"configuration drift for retained family {family_label!r}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _plan_id(seed_partitions: Any, families: Any, assignments: Any) -> str:
    payload = _canonical(
        {
            "assignments": assignments,
            "families": families,
            "seed_partitions": seed_partitions,
        }
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _statistics(seed_partitions: Any, families: Any, assignments: Any) -> dict[str, Any]:
    labels = [family["family_label"] for family in families]
    counts = {partition.value: 0 for partition in _PARTITION_ORDER}
    family_counts = {partition.value: set() for partition in _PARTITION_ORDER}
    for assignment in assignments:
        counts[assignment["partition"]] += 1
        family_counts[assignment["partition"]].add(assignment["family_label"])
    return {
        "instance_count": len(assignments),
        "family_count": len(labels),
        "partition_instance_counts": counts,
        "partition_family_counts": {
            partition.value: len(family_counts[partition.value]) for partition in _PARTITION_ORDER
        },
        "seeds_per_partition": {
            partition.value: len(seed_partitions[partition.value]) for partition in _PARTITION_ORDER
        },
    }


__all__ = [
    "QUALITY_PARTITIONS_SCHEMA_VERSION",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "VALIDATION_SEEDS",
    "build_quality_partition_plan",
    "materialize_partition_instances",
    "read_quality_partition_manifest",
    "validate_quality_partition_manifest",
    "write_quality_partition_manifest",
]
