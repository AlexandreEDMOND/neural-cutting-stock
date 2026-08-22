"""Publication data and figures derived from validated Phase 4 runs."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks import SIZE_CLASSES, compare_paired_runs
from neural_cutting_stock.benchmarks.schema import (
    BenchmarkRunRecord,
    EnvironmentMetadata,
    RunStatus,
    SolverMode,
)
from neural_cutting_stock.benchmarks.stats import median
from neural_cutting_stock.visualization._shared import seconds

_INT_FIELDS = {
    "seed",
    "repetition",
    "number_of_piece_types",
    "number_of_stock_formats",
    "total_demand",
    "peak_memory_bytes",
    "exact_pricing_calls",
    "number_of_candidates",
    "number_of_selected_columns",
    "exact_fallback_calls",
}
_BOOL_FIELDS = {"plan_feasible"}
_FLOAT_FIELDS = {
    "stock_length",
    "kerf",
    "requested_length",
    "objective_value",
    "number_of_stock_bars",
    "lp_objective_value",
    "restricted_integer_gap",
    "total_waste",
    "trim_loss",
    "kerf_loss",
    "overproduction_length",
    "number_of_cg_iterations",
    "number_of_generated_columns",
    "number_of_columns_added",
    "initial_column_count",
    "final_column_count",
    "duplicate_column_count",
    "final_reduced_cost",
    "total_runtime_seconds",
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
    "neural_inference_runtime",
    "feature_preparation_runtime",
    "speedup_vs_classical",
    "objective_difference_vs_classical",
}
_NULLABLE_FIELDS = {
    "size_class",
    "number_of_stock_formats",
    "stock_lengths",
    "objective_value",
    "number_of_stock_bars",
    "lp_objective_value",
    "restricted_integer_gap",
    "total_waste",
    "trim_loss",
    "kerf_loss",
    "overproduction_length",
    "plan_feasible",
    "number_of_cg_iterations",
    "number_of_generated_columns",
    "number_of_columns_added",
    "initial_column_count",
    "final_column_count",
    "duplicate_column_count",
    "final_reduced_cost",
    "total_runtime_seconds",
    "master_problem_runtime",
    "pricing_runtime",
    "integer_master_runtime",
    "column_management_runtime",
    "verification_runtime",
    "unattributed_runtime",
    "peak_memory_bytes",
    "exact_pricing_calls",
    "error_message",
    "model_id",
    "neural_inference_runtime",
    "feature_preparation_runtime",
    "number_of_candidates",
    "number_of_selected_columns",
    "exact_fallback_calls",
    "speedup_vs_classical",
    "objective_difference_vs_classical",
}


def load_phase4_runs(path: str | Path) -> tuple[BenchmarkRunRecord, ...]:
    """Load raw runs and validate every row through the versioned record schema."""

    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values: dict[str, Any] = dict(row)
            for field in _NULLABLE_FIELDS:
                if values.get(field, "") == "":
                    values[field] = None
            for field in _INT_FIELDS:
                if values.get(field) is not None:
                    values[field] = int(values[field])
            for field in _BOOL_FIELDS:
                if values[field] is not None:
                    values[field] = values[field].lower() == "true"
            for field in _FLOAT_FIELDS:
                if values.get(field) is not None:
                    values[field] = float(values[field])
            if values.get("stock_lengths"):
                values["stock_lengths"] = tuple(
                    float(length) for length in str(values["stock_lengths"]).split(";")
                )
            values["solver_mode"] = SolverMode(values["solver_mode"])
            values["run_status"] = RunStatus(values["run_status"])
            values["environment"] = EnvironmentMetadata(
                values.pop("code_commit"),
                values.pop("python_version"),
                values.pop("dependency_versions"),
                values.pop("hardware_id"),
            )
            records.append(BenchmarkRunRecord(**values))
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("raw Phase 4 runs must contain unique run_id values")
    return tuple(records)


def phase4_report_data(records: tuple[BenchmarkRunRecord, ...]) -> dict[str, Any]:
    """Recompute paired metrics and aggregate only quality-preserved pairs."""

    comparisons = compare_paired_runs(records)
    quality_pairs = [
        item
        for item in comparisons
        if item.quality_preserved and item.speedup_vs_classical is not None
    ]
    neural = {
        record.run_id: record for record in records if record.solver_mode is SolverMode.NEURAL
    }
    by_size: dict[str, list[Any]] = defaultdict(list)
    for item in quality_pairs:
        size = neural[item.neural_run_id].size_class
        if size in SIZE_CLASSES:
            by_size[size].append(item)
    size_data = {
        size: {
            "pair_count": len(by_size[size]),
            "classical_median_seconds": median(
                [
                    records_by_id(records, item.classical_run_id).total_runtime_seconds
                    for item in by_size[size]
                ]
            ),
            "neural_median_seconds": median(
                [neural[item.neural_run_id].total_runtime_seconds for item in by_size[size]]
            ),
            "speedup_median": median([item.speedup_vs_classical for item in by_size[size]]),
        }
        for size in SIZE_CLASSES
    }
    return {
        "records": records,
        "comparisons": comparisons,
        "quality_pair_count": len(quality_pairs),
        "comparable_pair_count": sum(item.comparable for item in comparisons),
        "size_data": size_data,
        "status_counts": {
            mode.value: {
                status.value: sum(
                    record.solver_mode is mode and record.run_status is status for record in records
                )
                for status in RunStatus
            }
            for mode in SolverMode
        },
        "objective_differences": [item.objective_difference_vs_classical for item in comparisons],
    }


def write_phase4_figures(data: dict[str, Any], output_dir: str | Path) -> None:
    """Write figures using only quality-preserved, validated paired measurements."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = [size for size in SIZE_CLASSES if data["size_data"][size]["pair_count"]]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        sizes,
        [data["size_data"][size]["classical_median_seconds"] for size in sizes],
        "o-",
        label="Classical CG",
    )
    axis.plot(
        sizes,
        [data["size_data"][size]["neural_median_seconds"] for size in sizes],
        "o-",
        label="Neural CG",
    )
    axis.set_xlabel("size-class-v1")
    axis.set_ylabel("Median total runtime (s)")
    axis.set_title("Measured paired runtime at preserved quality")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "runtime_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.plot(sizes, [data["size_data"][size]["speedup_median"] for size in sizes], "o-")
    axis.set_xlabel("size-class-v1")
    axis.set_ylabel("Median speedup (Classical / Neural)")
    axis.set_title("Measured paired speedup at preserved quality")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "speedup_by_size.png", dpi=160)
    plt.close(figure)


def write_phase4_summary(data: dict[str, Any], path: str | Path, runs_path: str) -> None:
    """Write the factual Phase 4 publication summary."""

    lines = [
        "# Bilan de publication de la Phase 4",
        "",
        f"Source brute validée : `{runs_path}` (schéma `benchmark-run-v1`).",
        "",
        "## Couverture",
        "",
        f"- Exécutions : **{len(data['records'])}** ; paires : **{len(data['comparisons'])}**.",
        f"- Paires comparables : **{data['comparable_pair_count']}** ; paires à qualité préservée : **{data['quality_pair_count']}**.",
        "- Les différences d'objectif et les speedups ont été recalculés depuis les enregistrements bruts ; aucune valeur dérivée persistée n'est utilisée.",
        "",
        "| Taille | Paires admissibles | Médiane Classical (s) | Médiane Neural (s) | Médiane speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in SIZE_CLASSES:
        item = data["size_data"][size]
        lines.append(
            f"| {size} | {item['pair_count']} | {seconds(item['classical_median_seconds'])} | {seconds(item['neural_median_seconds'])} | {seconds(item['speedup_median'])} |"
        )
    lines += [
        "",
        "## Garanties et limites",
        "",
        "Toutes les exécutions restent représentées, y compris les statuts non réussis. Une paire n'alimente les figures que si les deux plans sont faisables, convergés et ont une différence d'objectif nulle. Le contrôle exact du pricing et la vérification indépendante du plan restent ceux du solveur classique.",
        "",
        "Le corpus Phase 3 ne contient aucun candidat sélectionné : cette publication ne prétend donc pas mesurer la qualité d'un modèle entraîné sur ce corpus. Les mesures publiées décrivent uniquement la politique et le modèle identifiés dans les enregistrements bruts.",
        "",
        "## Figures",
        "",
        "- [`runtime_comparison.png`](runtime_comparison.png) : temps mur-à-mur médian des paires admissibles.",
        "- [`speedup_by_size.png`](speedup_by_size.png) : speedup médian des mêmes paires, avec référence `1x`.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def records_by_id(records: tuple[BenchmarkRunRecord, ...], run_id: str) -> BenchmarkRunRecord:
    return next(record for record in records if record.run_id == run_id)
