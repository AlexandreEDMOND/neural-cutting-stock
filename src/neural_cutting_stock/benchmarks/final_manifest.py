"""Generation and validation of the frozen Phase 6 instance manifest."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .generator import SyntheticInstanceGenerator
from .profile import SIZE_CLASSES

FINAL_MANIFEST_SCHEMA_VERSION = "phase-6-instance-manifest-v1"


def build_final_manifest(
    strata: tuple[dict[str, Any], ...],
    phase3_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Materialize deterministic, pre-evaluation instances for the final campaign."""

    entries = []
    for stratum in strata:
        target = stratum["target_size_class"]
        for seed in stratum["seeds"]:
            generator = SyntheticInstanceGenerator(
                seed=seed,
                stock_length=stratum["stock_length"],
                kerf=stratum["kerf"],
                number_of_types=stratum["number_of_types"],
                piece_length_range=tuple(stratum["piece_length_range"]),
                demand_range=tuple(stratum["demand_range"]),
                length_distribution=stratum["length_distribution"],
                demand_distribution=stratum["demand_distribution"],
            )
            instance = generator.generate()
            entries.append(
                {
                    "instance_id": generator.instance_id,
                    "seed": seed,
                    "family_id": generator.family_id,
                    "target_size_class": target,
                    "generator": _generator_dict(generator),
                    "stock_length": instance.stock_length,
                    "kerf": instance.kerf,
                    "piece_lengths": list(instance.piece_lengths),
                    "demands": list(instance.demands),
                }
            )

    entries = sorted(entries, key=lambda entry: entry["instance_id"])
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "manifest_id": _manifest_id(entries),
        "classification": {
            "version": "size-class-v1",
            "field": "target_size_class",
            "basis": "pre_evaluation_workload_stratum",
            "measured_field": "size_class",
        },
        "instances": entries,
        "statistics": _statistics(entries),
        "source": {"phase3_manifest_schema_version": phase3_manifest.get("schema_version")},
    }
    validate_final_manifest(manifest, phase3_manifest)
    return manifest


def validate_final_manifest(
    manifest: dict[str, Any], phase3_manifest: dict[str, Any]
) -> None:
    """Validate identities, materialized data, strata and Phase 3 separation."""

    if manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported final instance manifest")
    entries = manifest.get("instances")
    if not isinstance(entries, list) or not entries:
        raise ValueError("final manifest must contain instances")
    if manifest.get("classification", {}).get("version") != "size-class-v1":
        raise ValueError("final manifest must declare size-class-v1")

    previous_ids = {
        entry.get("instance_id")
        for entry in phase3_manifest.get("trajectories", [])
        if isinstance(entry, dict)
    }
    seen_ids: set[str] = set()
    for entry in entries:
        _validate_entry(entry)
        instance_id = entry["instance_id"]
        if instance_id in seen_ids:
            raise ValueError(f"duplicate final instance_id: {instance_id}")
        if instance_id in previous_ids:
            raise ValueError(f"final instance overlaps Phase 3: {instance_id}")
        seen_ids.add(instance_id)
        generator = generator_from_entry(entry)
        instance = generator.generate()
        if generator.instance_id != instance_id:
            raise ValueError(f"instance_id does not match materialized data: {instance_id}")
        materialized_matches = (
            list(instance.piece_lengths) == entry["piece_lengths"]
            and list(instance.demands) == entry["demands"]
        )
        if not materialized_matches:
            raise ValueError(f"materialized data does not match generator: {instance_id}")

    if manifest.get("manifest_id") != _manifest_id(entries):
        raise ValueError("manifest_id does not match instances")
    if manifest.get("statistics") != _statistics(entries):
        raise ValueError("final manifest statistics differ from instances")
    if set(_statistics(entries)["target_size_class_counts"]) != set(SIZE_CLASSES):
        raise ValueError("final manifest must cover all size classes")


def write_final_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Validate and write a canonical final manifest."""

    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _generator_dict(generator: SyntheticInstanceGenerator) -> dict[str, Any]:
    return {
        "name": generator.name,
        "version": generator.version,
        "seed": generator.seed,
        "stock_length": generator.stock_length,
        "kerf": generator.kerf,
        "number_of_types": generator.number_of_types,
        "piece_length_range": list(generator.piece_length_range),
        "demand_range": list(generator.demand_range),
        "length_distribution": generator.length_distribution,
        "demand_distribution": generator.demand_distribution,
    }


def generator_from_entry(entry: dict[str, Any]) -> SyntheticInstanceGenerator:
    """Reconstruct the deterministic generator recorded for one manifest entry."""
    config = entry["generator"]
    return SyntheticInstanceGenerator(
        seed=config["seed"],
        stock_length=config["stock_length"],
        kerf=config["kerf"],
        number_of_types=config["number_of_types"],
        piece_length_range=tuple(config["piece_length_range"]),
        demand_range=tuple(config["demand_range"]),
        length_distribution=config["length_distribution"],
        demand_distribution=config["demand_distribution"],
    )


def _validate_entry(entry: dict[str, Any]) -> None:
    if entry.get("target_size_class") not in SIZE_CLASSES:
        raise ValueError("each final instance needs a valid target_size_class")
    for field in ("instance_id", "family_id", "generator", "piece_lengths", "demands"):
        if field not in entry:
            raise ValueError(f"final instance is missing {field}")


def _manifest_id(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _statistics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {size_class: 0 for size_class in SIZE_CLASSES}
    for entry in entries:
        counts[entry["target_size_class"]] += 1
    return {"instance_count": len(entries), "target_size_class_counts": counts}
