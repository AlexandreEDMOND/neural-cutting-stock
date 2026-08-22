"""Phase 7 exact-gap publication from the persisted and validated report."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_cutting_stock.benchmarks.exact_gap_breakdown import build_exact_gap_breakdown
from neural_cutting_stock.visualization._shared import number as _number

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
            lines.append(_positive_row(entry))
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


def write_phase7_summary(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_path: str | Path,
) -> None:
    """Write the factual Phase 7 publication summary from the validated report.

    Every number is derived from the persisted ``exact-gap-v1`` report and its
    rebuilt family/size breakdown; no duration enters the summary because
    quality is the phase metric. The publication is refused when no gap is
    available against a verified exact reference.
    """

    breakdown = build_exact_gap_breakdown(report)
    totals = breakdown["totals"]
    if not totals["gap_available_count"]:
        raise ValueError("no gap is available against a verified exact reference")
    methods = sorted(
        {
            (entry["reference_method"], entry["reference_method_limits"])
            for entry in report["instances"]
        }
    )
    exclusions_by_reason: dict[str, int] = {}
    for item in report.get("excluded", []):
        reason = item["reason"]
        exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
    environment = report["environment"]
    cross_check = "activé" if report["cross_check_with_enumeration"] else "désactivé"
    positives = _unique_positive_instances(breakdown)
    max_gap = max(
        (
            group["max_gap_bars_median"]
            for group in breakdown["by_size_class"]
            if group["max_gap_bars_median"] is not None
        ),
        default=None,
    )
    lines = [
        "# Bilan de la Phase 7",
        "",
        f"Source validée : [`{Path(source_path).name}`]({Path(source_path).name}) (schéma `{report['schema_version']}`), régénérable depuis les corpus et campagnes persistés.",
        "Aucune durée n'entre dans ce bilan : la qualité est la métrique reine de la phase et les durées restent journalisées à titre informatif.",
        "",
        "## Méthode de référence et tolérances",
        "",
    ]
    for method, limits in methods:
        lines.append(f"- Référence : méthode `{method}`, limites `{limits}`.")
    lines += [
        f"- Tolérances : intégralité **{_number(report['integrality_tolerance'])}**, faisabilité **{_number(report['feasibility_tolerance'])}**.",
        f"- Contrôle croisé d'énumération : **{cross_check}**.",
        f"- Environnement tracé : commit `{environment['code_commit'][:12]}…`, Python {environment['python_version']}, {environment['dependency_versions']}.",
        "",
        "## Couverture",
        "",
        f"- Instances avec référence exacte : **{totals['instance_count']}** ; instances exclues sans référence rattachable : **{totals['excluded_instance_count']}** (diagnostics conservés dans le rapport source).",
        f"- Références : **{totals['optimal_reference_count']}** optimales, **{totals['lower_bound_only_reference_count']}** borne seule, **{totals['failed_reference_count']}** en échec ; vérifications indépendantes en échec : **{totals['verification_failure_count']}**.",
        f"- Écarts disponibles : **{totals['gap_available_count']}**, dont **{totals['zero_gap_count']}** nuls et **{totals['positive_gap_count']}** positifs.",
    ]
    for reason, count in sorted(exclusions_by_reason.items()):
        lines.append(f"- Exclusion : « {reason} » — {count} instance(s).")
    lines += [
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
        "## Instances à marge positive",
        "",
    ]
    if not positives:
        lines.append("Aucune instance mesurée ne perd de barre face à son optimum entier certifié.")
    else:
        lines += [
            "| Instance | Classe | Famille | Types | Optimum entier | Baseline médiane | Écart médian (barres) | Écarts par répétition |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for entry in positives:
            lines.append(_positive_row(entry))
    lines += [
        "",
        "## Lecture et suite",
        "",
        f"Sur les {totals['gap_available_count']} écarts disponibles, {totals['zero_gap_count']} sont nuls et {totals['positive_gap_count']} sont positifs ; l'écart médian maximal observé vaut {_number(max_gap)} barre(s), sur un objectif de découpe. Les leviers du générateur susceptibles de créer des trous entiers non triviaux sont documentés, sans activation ni mesure nouvelle, dans [`docs/phase-7-gap-levers.md`](../docs/phase-7-gap-levers.md) ; leur vérification expérimentale relève des cases P8.01, P8.03 et P8.04.",
        "",
        "## Garanties et limites",
        "",
        "Les références proviennent d'un MILP sur motifs énumérés, vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum, contrôle croisé d'énumération sur sous-échantillon lorsque activé). Les objectifs classiques restent des optimaux sur colonnes générées uniquement (`optimal_over_generated_columns_only`) et ne préjugent pas d'un optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les exclusions complètes et leurs raisons figurent dans [`exact-gap.json`](exact-gap.json).",
        "",
        "## Commandes de reproduction",
        "",
        "Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :",
        "",
        "```bash",
        "uv sync --extra dev",
        "uv run python scripts/report_phase7_exact_gap.py",
        "uv run python scripts/report_phase7_exact_gap_breakdown.py",
        "uv run python scripts/report_phase7_summary.py",
        "```",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _positive_row(entry: Mapping[str, Any]) -> str:
    return (
        f"| `{entry['instance_id'][:12]}…` | {_label(entry['size_class'])}"
        f" | `{str(entry['family_id'])[:12]}…` | {entry['number_of_piece_types']}"
        f" | {entry['integer_optimum_bars']}"
        f" | {_number(entry['baseline_objective_bars_median'])}"
        f" | {_number(entry['gap_bars_median'])}"
        f" | {';'.join(str(gap) for gap in entry['gap_bars_per_repetition'])} |"
    )


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


__all__ = ["write_exact_gap_breakdown_markdown", "write_phase7_summary"]
