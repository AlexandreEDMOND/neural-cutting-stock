"""Phase 5 publication data and figures derived from raw paired runs."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks import compare_paired_runs, freeze_candidate_on_validation
from neural_cutting_stock.benchmarks.schema import BenchmarkRunRecord, SolverMode
from neural_cutting_stock.visualization.phase4 import SIZE_CLASSES, load_phase4_runs


def load_phase5_runs(path: str | Path) -> tuple[BenchmarkRunRecord, ...]:
    """Load the versioned raw paired-run source used for the Phase 5 publication."""

    return load_phase4_runs(path)


def phase5_report_data(
    records: tuple[BenchmarkRunRecord, ...], candidate_id: str
) -> dict[str, Any]:
    """Recompute the freeze decision and quality-gated aggregates from raw records."""

    decision = freeze_candidate_on_validation(records, candidate_id)
    comparisons = compare_paired_runs(records)
    neural = {record.run_id: record for record in records if record.solver_mode is SolverMode.NEURAL}
    classical = {
        record.run_id: record for record in records if record.solver_mode is SolverMode.CLASSICAL
    }
    by_size: dict[str, list[Any]] = defaultdict(list)
    for comparison in comparisons:
        if comparison.quality_preserved and comparison.speedup_vs_classical is not None:
            size = neural[comparison.neural_run_id].size_class
            if size in SIZE_CLASSES:
                by_size[size].append(comparison)
    size_data = {
        size: {
            "pair_count": len(by_size[size]),
            "classical_median_seconds": _median(
                [classical[item.classical_run_id].total_runtime_seconds for item in by_size[size]]
            ),
            "neural_median_seconds": _median(
                [neural[item.neural_run_id].total_runtime_seconds for item in by_size[size]]
            ),
            "speedup_median": _median([item.speedup_vs_classical for item in by_size[size]]),
        }
        for size in SIZE_CLASSES
    }
    return {
        "records": records,
        "comparisons": comparisons,
        "decision": decision,
        "size_data": size_data,
        "quality_pair_count": sum(
            item.quality_preserved and item.speedup_vs_classical is not None for item in comparisons
        ),
        "comparable_pair_count": sum(item.comparable for item in comparisons),
    }


def write_phase5_figures(data: dict[str, Any], output_dir: str | Path) -> None:
    """Write decision figures using only quality-preserved paired measurements."""

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
    axis.set_title("Phase 5 decision: measured paired runtime")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "phase5_runtime_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.plot(sizes, [data["size_data"][size]["speedup_median"] for size in sizes], "o-")
    axis.set_xlabel("size-class-v1")
    axis.set_ylabel("Median speedup (Classical / Neural)")
    axis.set_title("Phase 5 decision: measured quality-gated speedup")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "phase5_speedup_by_size.png", dpi=160)
    plt.close(figure)


def write_phase5_summary(data: dict[str, Any], path: str | Path, runs_path: str) -> None:
    """Write the factual Phase 5 decision summary."""

    decision = data["decision"]
    source_path = Path(runs_path)
    if source_path.is_absolute():
        with suppress(ValueError):
            runs_path = str(source_path.relative_to(Path.cwd()))
    lines = [
        "# Bilan de publication de la Phase 5",
        "",
        f"Source brute validée : `{runs_path}` (schéma `benchmark-run-v1`).",
        "",
        "## Décision",
        "",
        f"Le candidat `{decision.candidate_id}` n'est pas gelé : **{decision.reason}**.",
        "La politique supervisée bornée de Phase 4 reste donc le candidat retenu, avec le pricing exact et le fallback exact.",
        "La comparaison porte sur le temps mur-à-mur total agrégé, et non sur le seul temps d'inférence.",
        "",
        "## Couverture et mesure",
        "",
        f"- Exécutions : **{len(data['records'])}** ; paires : **{len(data['comparisons'])}**.",
        f"- Paires comparables : **{data['comparable_pair_count']}** ; paires à qualité préservée : **{data['quality_pair_count']}**.",
        f"- Runtime Classical agrégé : **{_seconds(decision.classical_total_runtime_seconds)} s**.",
        f"- Runtime Neural agrégé : **{_seconds(decision.candidate_total_runtime_seconds)} s**.",
        "- Les différences d'objectif et les speedups ont été recalculés depuis les enregistrements bruts.",
        "",
        "| Taille | Paires admissibles | Médiane Classical (s) | Médiane Neural (s) | Médiane speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in SIZE_CLASSES:
        item = data["size_data"][size]
        lines.append(
            f"| {size} | {item['pair_count']} | {_seconds(item['classical_median_seconds'])} | {_seconds(item['neural_median_seconds'])} | {_seconds(item['speedup_median'])} |"
        )
    lines += [
        "",
        "## Garanties et limites",
        "",
        "Toutes les lignes de la source sont conservées dans l'analyse. Les figures n'utilisent que les paires dont les deux plans sont faisables, convergés et de même objectif. Le pricing exact, le fallback exact et la vérification indépendante restent obligatoires.",
        "La couverture ne contient aucune paire LARGE ou XL ; ces mesures ne démontrent donc pas un gain généralisable sur les grandes instances.",
        "",
        "## Figures",
        "",
        "- [`phase5_runtime_comparison.png`](phase5_runtime_comparison.png) : runtime médian des paires à qualité préservée.",
        "- [`phase5_speedup_by_size.png`](phase5_speedup_by_size.png) : speedup médian avec référence `1x`.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median(values: list[float | None]) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    return float(numbers[middle]) if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
