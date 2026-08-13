"""Utilities for the reproducible Phase 6 classical baseline campaign."""

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .final_manifest import validate_final_manifest
from .generator import SyntheticInstanceGenerator
from .schema import EnvironmentMetadata


def generators_from_final_manifest(
    manifest: dict[str, Any], phase3_manifest: dict[str, Any]
) -> tuple[SyntheticInstanceGenerator, ...]:
    """Reconstruct the exact generators recorded in the frozen manifest."""

    validate_final_manifest(manifest, phase3_manifest)
    generators = tuple(
        _generator_from_entry(entry) for entry in manifest["instances"]
    )
    if tuple(generator.instance_id for generator in generators) != tuple(
        entry["instance_id"] for entry in manifest["instances"]
    ):
        raise ValueError("final manifest generator order or identities changed")
    return generators


def collect_environment(repo_root: str | Path) -> EnvironmentMetadata:
    """Collect runtime, dependency, hardware and repository identity metadata."""

    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to record the repository commit") from error
    dependencies = ",".join(
        f"{name}=={_installed_version(name)}" for name in ("numpy", "scipy")
    )
    thread_settings = ",".join(
        f"{name}={os.environ.get(name, 'unset')}"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    hardware = ";".join(
        (
            f"platform={platform.platform()}",
            f"machine={platform.machine() or 'unknown'}",
            f"processor={platform.processor() or 'unknown'}",
            f"cpu_count={os.cpu_count() or 'unknown'}",
            f"threads={thread_settings}",
        )
    )
    return EnvironmentMetadata(
        code_commit=commit,
        python_version=platform.python_version(),
        dependency_versions=dependencies,
        hardware_id=hardware,
    )


def write_campaign_metadata(
    path: str | Path,
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    environment: EnvironmentMetadata,
    benchmark_config_id: str,
    run_count: int,
) -> None:
    """Persist campaign inputs and observed environment beside the raw CSV."""

    payload = {
        "schema_version": "phase-6-classical-baseline-v1",
        "config_schema_version": config["schema_version"],
        "config_id": benchmark_config_id,
        "final_manifest_id": manifest["manifest_id"],
        "final_manifest_sha256": _file_sha256(config["files"]["final_instance_manifest"]),
        "environment": {
            "code_commit": environment.code_commit,
            "python_version": environment.python_version,
            "dependency_versions": environment.dependency_versions,
            "hardware_id": environment.hardware_id,
        },
        "solver": {
            "mode": "classical",
            "solver_version": "classical-cg-v1",
            "repetitions": config["protocol"]["repetitions"],
            "reduced_cost_tolerance": config["protocol"]["reduced_cost_tolerance"],
            "max_runtime_seconds": config["protocol"]["max_runtime_seconds"],
            "max_cg_iterations": config["protocol"]["max_cg_iterations"],
        },
        "run_count": run_count,
    }
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generator_from_entry(entry: dict[str, Any]) -> SyntheticInstanceGenerator:
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


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required dependency is not installed: {name}") from error


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
