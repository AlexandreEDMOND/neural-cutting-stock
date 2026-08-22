"""Phase 7 exact-gap publication from the persisted and validated report."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXACT_GAP_REPORT = "exact-gap.json"


def write_exact_gap_breakdown_markdown(
    breakdown: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_path: str | Path,
) -> None:
    """Render the family and size-class margin tables of the exact-gap report."""

    totals = breakdown["totals"]
    lines = [
        "# Bilan chiffré des écarts à la référence exacte (Phase 7)",
        "",
        f"Source validée : [`{Path(source_path).name}`]({Path(source_path).name}) (schéma `{breakdown['source_schema_version']}`).",
        "Chaque écart compare la baseline classique (`optimal_over_generated_columns_only`) à l'optimum entier certifié de sa référence exacte vérifiée ; aucune durée n'entre dans ce bilan.",
        "",
        "## Couverture",
        "",
        f"- Instances avec référence : **{totals['instance_count']}** ; instances exclues sans baseline persistée : **{totals['excluded_instance_count']}** (diagnostics conservés dans le rapport source).",
        f"- Références : **{totals['optimal_reference_count']}** optimales, **{totals['lower_bound_only_reference_count']}** borne seule, **{totals['failed_reference_count']}** en échec ; vérifications indépendantes en échec : **{totals['verification_failure_count']}**.",
        f"- Écarts disponibles : **{totals['gap_available_count']}**, dont **{totals['zero_gap_count']}** nuls et **{totals['positive_gap_count']}** positifs.",
        "",
        "## Marge par classe de taille",
        "",
        "| Classe | Instances | Types | Écarts disponibles | Marge nulle | Marge positive | Écart médian maximal (barres) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for group in breakdown["by_size_class"]:
        lines.append(_group_row(group))
    lines += [
        "",
        "## Marge par famille",
        "",
        "| Famille | Types | Instances | Écarts disponibles | Marge nulle | Marge positive | Écart médian maximal (barres) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in breakdown["by_family"]:
        lines.append(_family_row(group))
    lines += [
        "",
        "## Instances à marge positive",
        "",
    ]
    positives = _unique_positive_instances(breakdown)
    if not positives:
        lines.append("Aucune instance mesurée ne perd de barre face à son optimum entier certifié.")
    else:
        lines += [
            "| Instance | Classe | Famille | Types | Optimum entier | Baseline médiane | Écart médian (barres) | Écarts par répétition |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for entry in positives:
            lines.append(
                f"| `{entry['instance_id'][:12]}…` | {_label(entry['size_class'])}"
                f" | `{str(entry['family_id'])[:12]}…` | {entry['number_of_piece_types']}"
                f" | {entry['integer_optimum_bars']}"
                f" | {_number(entry['baseline_objective_bars_median'])}"
                f" | {_number(entry['gap_bars_median'])}"
                f" | {';'.join(str(gap) for gap in entry['gap_bars_per_repetition'])} |"
            )
    unavailable_lines = _unavailable_lines(breakdown)
    if unavailable_lines:
        lines += [
            "",
            "## Écarts indisponibles",
            "",
            "Ces groupes conservent leurs instances sans qu'elles contribuent aux marges :",
            "",
            *unavailable_lines,
        ]
    lines += [
        "",
        "## Garanties et limites",
        "",
        "Les références proviennent d'un MILP sur motifs énumérés, vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum, contrôle croisé d'énumération sur sous-échantillon). Les objectifs classiques restent des optimaux sur colonnes générées uniquement et ne préjugent pas d'un optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les exclusions complètes et leurs raisons figurent dans [`exact-gap.json`](exact-gap.json). Le bilan se régénère depuis les données persistées avec `uv run python scripts/report_phase7_exact_gap_breakdown.py`.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _unique_positive_instances(breakdown: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    positives = {
        entry["instance_id"]: entry
        for group in breakdown["by_size_class"]
        for entry in group["positive_instances"]
    }
    return sorted(
        positives.values(), key=lambda entry: (-entry["gap_bars_median"], entry["instance_id"])
    )


def _unavailable_lines(breakdown: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    rendered: set[str] = set()
    for section in ("by_size_class", "by_family"):
        for group in breakdown[section]:
            for reason, count in group["gap_unavailable_reasons"].items():
                line = f"- {_label(group['key'])} ({section}) : `{reason}` — {count} instance(s)."
                if line not in rendered:
                    rendered.add(line)
                    lines.append(line)
    return lines


def _group_row(group: Mapping[str, Any]) -> str:
    return (
        f"| {_label(group['key'])} | {group['instance_count']} | {_types(group)}"
        f" | {group['gap_available_count']} | {group['zero_gap_count']} | {group['positive_gap_count']}"
        f" | {_number(group['max_gap_bars_median'])} |"
    )


def _family_row(group: Mapping[str, Any]) -> str:
    key = "n/a" if group["key"] is None else f"`{str(group['key'])[:12]}…`"
    return (
        f"| {key} | {_types(group)} | {group['instance_count']}"
        f" | {group['gap_available_count']} | {group['zero_gap_count']} | {group['positive_gap_count']}"
        f" | {_number(group['max_gap_bars_median'])} |"
    )


def _types(group: Mapping[str, Any]) -> str:
    return ", ".join(f"{types}×{count}" for types, count in group["piece_type_counts"].items())


def _label(key: object) -> str:
    return "n/a" if key is None else str(key)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


__all__ = ["write_exact_gap_breakdown_markdown"]
