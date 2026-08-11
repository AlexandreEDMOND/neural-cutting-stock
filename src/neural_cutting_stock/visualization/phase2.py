"""Figures and report data derived from the persisted Phase 2 profile."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks.profile import (
    PROFILE_SCHEMA_VERSION,
    SIZE_CLASS_SCHEMA_VERSION,
    SizeClass,
)

SIZE_CLASSES = tuple(size_class.value for size_class in SizeClass)
COMPONENTS = (
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
)


def load_phase2_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate the persisted classical profile used for publication."""

    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("profile_schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported profile_schema_version")
    if profile.get("size_class_schema_version") != SIZE_CLASS_SCHEMA_VERSION:
        raise ValueError("unsupported size_class_schema_version")
    runs = profile.get("runs")
    if not isinstance(runs, list) or len({run.get("run_id") for run in runs}) != len(runs):
        raise ValueError("profile must contain unique runs")
    if any(run.get("solver_mode") != "classical" for run in runs):
        raise ValueError("Phase 2 profile must contain classical runs only")
    return profile


def phase2_report_data(profile: dict[str, Any]) -> dict[str, Any]:
    """Compute report aggregates from successful profile records."""

    runs = [run for run in profile["runs"] if run["run_status"] == "optimal_lp_restricted_ip"]
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["size_class"] in SIZE_CLASSES:
            by_size[run["size_class"]].append(run)
    size_data = {}
    for size_class in SIZE_CLASSES:
        runtimes = sorted(run["total_runtime_seconds"] for run in by_size[size_class])
        iterations = sorted(run["number_of_cg_iterations"] for run in by_size[size_class])
        size_data[size_class] = {
            "count": len(runtimes),
            "runtime_min_seconds": runtimes[0] if runtimes else None,
            "runtime_median_seconds": _median(runtimes),
            "runtime_max_seconds": runtimes[-1] if runtimes else None,
            "iterations_median": _median(iterations),
        }
    return {"successful_runs": runs, "size_data": size_data}


def write_phase2_figures(profile: dict[str, Any], output_dir: str | Path) -> None:
    """Write classical Phase 2 figures using only values in ``profile``."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_data = phase2_report_data(profile)
    sizes = [size for size in SIZE_CLASSES if report_data["size_data"][size]["count"]]
    medians = [report_data["size_data"][size]["runtime_median_seconds"] for size in sizes]
    lows = [
        report_data["size_data"][size]["runtime_median_seconds"]
        - report_data["size_data"][size]["runtime_min_seconds"]
        for size in sizes
    ]
    highs = [
        report_data["size_data"][size]["runtime_max_seconds"]
        - report_data["size_data"][size]["runtime_median_seconds"]
        for size in sizes
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(sizes, medians, yerr=[lows, highs], fmt="o-", capsize=4)
    axis.set_xlabel("size-class-v1")
    axis.set_ylabel("Median total runtime (s)")
    axis.set_title("Classical CG runtime by measured difficulty")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "classical_runtime_by_size.png", dpi=160)
    plt.close(figure)

    totals = [profile["component_totals_seconds"][component] for component in COMPONENTS]
    labels = [component.removesuffix("_runtime").replace("_", " ") for component in COMPONENTS]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, totals)
    axis.set_ylabel("Accumulated runtime (s)")
    axis.set_title("Classical CG measured runtime decomposition")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "classical_runtime_components.png", dpi=160)
    plt.close(figure)


def _median(values: list[int | float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (values[middle - 1] + values[middle]) / 2
