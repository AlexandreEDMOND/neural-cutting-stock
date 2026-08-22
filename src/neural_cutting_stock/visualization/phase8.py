"""Phase 8 family-margin publication from the persisted and validated report."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_cutting_stock.visualization._shared import number as _number


def write_family_margins_markdown(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_path: str | Path,
) -> None:
    """Render the measured margins and retention decision of each Phase 8 family.

    Every number comes from the persisted ``family-margins-v1`` report; no
    duration enters the publication because quality is the phase metric.
    """

    counts = report["counts"]
    environment = report["environment"]
    cross_check = "activé" if report["cross_check_with_enumeration"] else "désactivé"
    lines = [
        "# Marges de qualité des nouvelles familles (Phase 8)",
        "",
        f"Source validée : [`{Path(source_path).name}`]({Path(source_path).name}) "
        f"(schéma `{report['schema_version']}`), produite par exécutions réelles : une baseline "
        "classique et une référence exacte MILP vérifiée indépendamment par instance. Aucune durée "
        "n'entre dans ce bilan.",
        "",
        "## Méthode, seuil et tolérances",
        "",
        f"- Référence : méthode `milp_on_enumerated_patterns`, limites `{report['reference_method_limits']}`.",
        f"- Tolérances : coût réduit **{_number(report['reduced_cost_tolerance'])}**, intégralité "
        f"**{_number(report['integrality_tolerance'])}**, faisabilité **{_number(report['feasibility_tolerance'])}**.",
        f"- Contrôle croisé d'énumération : **{cross_check}**.",
        "- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un écart "
        f"disponible et qu'au moins **{report['significant_positive_share'] * 100:.0f} %** des "
        "instances perdent au moins une barre face à leur optimum entier certifié.",
        f"- Environnement tracé : commit `{environment['code_commit'][:12]}…`, Python {environment['python_version']}, {environment['dependency_versions']}.",
        "",
        "## Marge par famille",
        "",
        "| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive | Retenue |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for family in report["families"]:
        lines.append(_family_row(family))
    lines += [
        "",
        f"{counts['retained_family_count']} famille(s) retenue(s) sur {counts['family_count']} ; "
        f"part positive globale : {counts['positive_gap_count']} instances positives sur "
        f"{counts['instance_count']} mesurées.",
        "",
    ]
    unmeasured = report["unmeasured_families"]
    if unmeasured:
        lines += [
            "## Familles non mesurées",
            "",
            "Ces familles déclarées n'ont produit aucune mesure faute de support ; elles ne peuvent pas être retenues en l'état :",
            "",
        ]
        for item in unmeasured:
            lines.append(f"- `{item['family_label']}` — {item['reason']}.")
        lines.append("")
    unavailable = [entry for entry in report["instances"] if not entry["gap_available"]]
    positives = sorted(
        (entry for entry in report["instances"] if entry["gap_available"] and entry["gap_bars"] > 0),
        key=lambda entry: (-entry["gap_bars"], entry["family_label"], entry["instance_id"]),
    )
    lines += [
        "## Instances à marge positive",
        "",
    ]
    if not positives:
        lines.append("Aucune instance mesurée ne perd de barre face à son optimum entier certifié.")
    else:
        lines += [
            "| Instance | Famille | Types | Optimum entier | Baseline classique | Écart (barres) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for entry in positives:
            lines.append(
                f"| `{entry['instance_id'][:12]}…` | `{entry['family_label']}`"
                f" | {entry['number_of_piece_types']}"
                f" | {entry['integer_optimum_bars']}"
                f" | {_number(entry['classical_bars'])}"
                f" | {_number(entry['gap_bars'])} |"
            )
    unavailability_line = (
        f" {len(unavailable)} instance(s) restent sans écart disponible et conservent leur diagnostic."
        if unavailable
        else ""
    )
    lines += [
        "",
        "## Garanties et limites",
        "",
        "Les références proviennent d'un MILP sur motifs énumérés et sont vérifiées indépendamment "
        "(faisabilité du plan, borne LP ≤ optimum). Les objectifs classiques restent des optimaux sur "
        "colonnes générées uniquement (`optimal_over_generated_columns_only`) et ne préjugent pas d'un "
        "optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les raisons "
        "complètes figurent dans le rapport source."
        + unavailability_line,
        "",
        "Le bilan se régénère depuis les données persistées avec "
        "`uv run python scripts/report_phase8_family_margins.py`.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _family_row(family: Mapping[str, Any]) -> str:
    share = f"{family['positive_share_of_instances'] * 100:.0f} %"
    return (
        f"| `{family['family_label']}` | {family['instance_count']}"
        f" | {family['gap_available_count']} | {family['zero_gap_count']}"
        f" | {family['positive_gap_count']} | {share}"
        f" | {'oui' if family['retained'] else 'non'} |"
    )


__all__ = ["write_family_margins_markdown"]
