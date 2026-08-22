"""Phase 6 paired publication tables and figures from validated final results."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks import SIZE_CLASSES, BenchmarkRunRecord, build_paired_tables


def write_paired_tables_markdown(
    report: dict[str, Any],
    output_path: str | Path,
    classical_source: str,
    neural_source: str,
) -> None:
    """Render the paired quality, runtime, memory, iteration and column tables."""

    lines = [
        "# Tableaux appariés de la Phase 6",
        "",
        f"Sources brutes validées : `{_relative(classical_source)}` et `{_relative(neural_source)}` (schéma `benchmark-run-v1`).",
        "",
        "## Couverture",
        "",
        f"- Exécutions : **{report['run_count']}** ; paires : **{report['pair_count']}** ; paires admissibles : **{report['admissible_pair_count']}**.",
        f"- Tolérance de différence d'objectif : **{report['quality_tolerance_bars']} barre(s)**. Les différences d'objectif et les speedups sont recalculés depuis les enregistrements bruts.",
        "- Les médianes par instance n'agrègent que les répétitions admissibles ; chaque paire, y compris échec ou violation, reste conservée dans [`phase-6-paired-tables.json`](phase-6-paired-tables.json).",
        "",
        "## Qualité (barres)",
        "",
        "| Instance | Types | Répétitions | Admissibles | Objectif Classical médian | Objectif Neural médian | Différence médiane | Violations qualité |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['number_of_piece_types']} | {item['repetition_count']}"
            f" | {item['admissible_repetition_count']} | {_number(item['objective_classical_bars_median'])}"
            f" | {_number(item['objective_neural_bars_median'])} | {_number(item['objective_difference_vs_classical_median'])}"
            f" | {item['quality_violation_pair_count']} |"
        )
    lines += [
        "",
        "## Runtime mur-à-mur (s)",
        "",
        "| Instance | Types | Admissibles | Classical médian (s) | Neural médian (s) | Speedup médian |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['number_of_piece_types']} | {item['admissible_repetition_count']}"
            f" | {_seconds(item['classical_runtime_seconds_median'])} | {_seconds(item['neural_runtime_seconds_median'])}"
            f" | {_seconds(item['speedup_vs_classical_median'])} |"
        )
    lines += [
        "",
        "## Mémoire (pic tracemalloc, octets)",
        "",
        "| Instance | Types | Admissibles | Pic Classical médian | Pic Neural médian |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['number_of_piece_types']} | {item['admissible_repetition_count']}"
            f" | {_count(item['classical_peak_memory_bytes_median'])} | {_count(item['neural_peak_memory_bytes_median'])} |"
        )
    lines += [
        "",
        "## Itérations CG",
        "",
        "| Instance | Types | Admissibles | Itérations Classical médian | Itérations Neural médian |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['number_of_piece_types']} | {item['admissible_repetition_count']}"
            f" | {_count(item['classical_cg_iterations_median'])} | {_count(item['neural_cg_iterations_median'])} |"
        )
    lines += [
        "",
        "## Colonnes (C = Classical, N = Neural)",
        "",
        "| Instance | Types | Admissibles | Générées C/N | Ajoutées C/N | Finales C/N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['number_of_piece_types']} | {item['admissible_repetition_count']}"
            f" | {_count(item['classical_generated_columns_median'])} / {_count(item['neural_generated_columns_median'])}"
            f" | {_count(item['classical_added_columns_median'])} / {_count(item['neural_added_columns_median'])}"
            f" | {_count(item['classical_final_columns_median'])} / {_count(item['neural_final_columns_median'])} |"
        )
    lines += [
        "",
        "## Garanties et limites",
        "",
        "Les deux maîtres entiers sont résolus sur les colonnes générées uniquement : les objectifs rapportés sont des optimaux sur colonnes générées et ne préjugent pas d'un optimum entier global. Aucune médiane n'agrège une paire hors tolérance de qualité ou incomplètement mesurée ; ces paires restent listées dans le JSON source. La mémoire est le pic d'allocations tracé par `tracemalloc` pendant le solveur, pas la consommation RSS du processus.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def phase6_runtime_comparison_data(
    classical_records: Sequence[BenchmarkRunRecord],
    neural_records: Sequence[BenchmarkRunRecord],
    size_class_by_instance: Mapping[str, str],
    quality_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Aggregate validated final paired runs into per-stratum runtime medians.

    The paired tables are rebuilt from the raw campaign records with the same
    machinery as the published tables; only quality-preserved admissible
    repetitions feed the medians. Grouping uses the frozen pre-evaluation
    ``target_size_class`` of the final instance manifest, not the per-run
    measured size class, so both modes are grouped on the same strata.
    """

    report = build_paired_tables(classical_records, neural_records, quality_tolerance)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in report["pairs"]:
        if not pair["admissible"]:
            continue
        instance_id = pair["instance_id"]
        if instance_id not in size_class_by_instance:
            raise ValueError(
                f"paired instance is missing from the final instance manifest: {instance_id}"
            )
        grouped[size_class_by_instance[instance_id]].append(pair)
    size_data = {}
    for size in SIZE_CLASSES:
        rows = grouped.get(size, [])
        size_data[size] = {
            "pair_count": len(rows),
            "instance_count": len({row["instance_id"] for row in rows}),
            "classical_median_seconds": _median(
                [row["classical_total_runtime_seconds"] for row in rows]
            ),
            "neural_median_seconds": _median(
                [row["neural_total_runtime_seconds"] for row in rows]
            ),
        }
    return {"report": report, "size_data": size_data}


def write_phase6_runtime_comparison(data: dict[str, Any], output_dir: str | Path) -> None:
    """Write ``runtime_comparison.png`` from admissible final pairs only."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = [size for size in SIZE_CLASSES if data["size_data"][size]["pair_count"]]
    if not sizes:
        raise ValueError("no admissible pair is available for the runtime comparison figure")
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
    axis.set_xlabel("Target size class (size-class-v1)")
    axis.set_ylabel("Median total runtime (s)")
    axis.set_title("Final evaluation: measured paired runtime at preserved quality")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "runtime_comparison.png", dpi=160)
    plt.close(figure)


def _relative(source: str) -> str:
    with suppress(ValueError):
        return str(Path(source).resolve().relative_to(Path.cwd().resolve()))
    return str(source)


def _median(values: list[float | None]) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    return (
        float(numbers[middle]) if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2
    )


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def _count(value: float | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
