"""Phase 6 paired publication tables rendered from validated report data."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from contextlib import suppress
from pathlib import Path
from typing import Any


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


def _relative(source: str) -> str:
    with suppress(ValueError):
        return str(Path(source).resolve().relative_to(Path.cwd().resolve()))
    return str(source)


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def _count(value: float | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
