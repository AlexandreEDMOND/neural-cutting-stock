"""Phase 8 family-margin publication from the persisted and validated report."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_cutting_stock.benchmarks import (
    FAMILY_MARGINS_SCHEMA_VERSION,
    QUALITY_PARTITIONS_SCHEMA_VERSION,
)
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


def write_quality_benchmark_choice(
    margins_report: Mapping[str, Any],
    partitions_manifest: Mapping[str, Any],
    output_path: str | Path,
    *,
    margins_link: str,
    partitions_link: str,
) -> None:
    """Render the interim Phase 8 bilan: per-family margins and the final choice.

    Every number comes from the persisted ``family-margins-v1`` measurement and
    the frozen ``phase-8-quality-partitions-v1`` manifest; the publication is
    refused when the frozen partitions do not cover exactly the families the
    measurement retains. No duration enters the document.
    """

    _validate_choice_sources(margins_report, partitions_manifest)
    counts = margins_report["counts"]
    environment = margins_report["environment"]
    cross_check = "activé" if margins_report["cross_check_with_enumeration"] else "désactivé"
    instances = margins_report["instances"]
    retained = [family for family in margins_report["families"] if family["retained"]]
    rejected = [family for family in margins_report["families"] if not family["retained"]]
    retained_gaps = [
        entry["gap_bars"]
        for entry in instances
        if entry["gap_available"] and any(f["family_label"] == entry["family_label"] for f in retained)
    ]
    lines = [
        "# Choix du benchmark qualité final (Phase 8)",
        "",
        "Bilan intermédiaire de la Phase 8 : tableau des marges mesurées par famille et choix "
        "documenté du benchmark qualité final utilisé par les phases suivantes. Sources validées : "
        f"[`{margins_link}`]({margins_link}) (schéma `{margins_report['schema_version']}`) et "
        f"[`{partitions_link}`]({partitions_link}) (schéma "
        f"`{partitions_manifest['schema_version']}`). Toutes les valeurs proviennent d'exécutions "
        "réelles ; aucune durée n'entre dans ce bilan.",
        "",
        "## Méthode de mesure et règle de rétention",
        "",
        "- Référence : méthode `milp_on_enumerated_patterns`, limites "
        f"`{margins_report['reference_method_limits']}`.",
        f"- Tolérances : coût réduit **{_number(margins_report['reduced_cost_tolerance'])}**, "
        f"intégralité **{_number(margins_report['integrality_tolerance'])}**, faisabilité "
        f"**{_number(margins_report['feasibility_tolerance'])}**.",
        f"- Contrôle croisé d'énumération : **{cross_check}**.",
        "- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un "
        "écart disponible et qu'au moins "
        f"**{margins_report['significant_positive_share'] * 100:.0f} %** des instances perdent au "
        "moins une barre face à leur optimum entier certifié.",
        f"- Environnement tracé : commit `{environment['code_commit'][:12]}…`, Python "
        f"{environment['python_version']}, {environment['dependency_versions']}.",
        "",
        "## Marges par famille",
        "",
        "| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive "
        "| Écart maximal (barres) | Somme des écarts (barres) | Retenue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family in margins_report["families"]:
        lines.append(_choice_row(family, instances))
    lines += [
        "",
        f"{counts['retained_family_count']} famille(s) retenue(s) sur {counts['family_count']} ; "
        f"part positive globale : {counts['positive_gap_count']} instances positives sur "
        f"{counts['instance_count']} mesurées.",
        "",
        "## Choix du benchmark qualité final",
        "",
        "Le benchmark qualité final retenu pour les phases suivantes est le plan de partitions gelé "
        f"[`{partitions_link}`]({partitions_link}) (`plan_id` "
        f"`{partitions_manifest['plan_id'][:12]}…`) : exactement les familles retenues ci-dessus, "
        "sans aucune autre. Toute amélioration de qualité revendiquée sera entraînée sur `train`, "
        "ajustée sur `validation` et mesurée sur `test` de ces partitions ; aucune graine n'est "
        "partagée entre deux partitions et chaque `instance_id` n'y apparaît qu'une fois.",
        "",
        "| Partition | Graines | Instances | Familles |",
        "|---|---|---:|---:|",
    ]
    statistics = partitions_manifest["statistics"]
    for partition in ("train", "validation", "test"):
        lines.append(
            f"| {partition} | {_seed_span(partitions_manifest['seed_partitions'][partition])} "
            f"| {statistics['partition_instance_counts'][partition]} "
            f"| {statistics['partition_family_counts'][partition]} |"
        )
    lines += [
        "",
        "## Familles écartées du benchmark qualité final",
        "",
        "Les familles non retenues restent modélisées et exécutables (générateurs, solveur, schémas) "
        "mais n'entrent pas dans le benchmark qualité final : leur marge mesurée ne satisfait pas la "
        "règle de rétention, si bien qu'une amélioration de qualité y serait invérifiable face à "
        "l'optimum entier certifié.",
        "",
        "| Famille | Part positive | Instances à marge positive | Écart maximal (barres) |",
        "|---|---:|---:|---:|",
    ]
    if rejected:
        for family in rejected:
            share = f"{family['positive_share_of_instances'] * 100:.0f} %"
            lines.append(
                f"| `{family['family_label']}` | {share} "
                f"| {family['positive_gap_count']}/{family['instance_count']} "
                f"| {family['max_gap_bars']} |"
            )
    else:
        lines.append("Aucune : toutes les familles mesurées satisfont la règle de rétention.")
    unmeasured = margins_report["unmeasured_families"]
    if unmeasured:
        lines += [
            "",
            "Familles déclarées sans aucune mesure faute de support, donc non retenues en l'état :",
            "",
        ]
        for item in unmeasured:
            lines.append(f"- `{item['family_label']}` — {item['reason']}.")
    lines += [
        "",
        "Conséquence documentée : le benchmark qualité final ne couvre que des barres à format "
        "unique, sans kerf exercé (`kerf = 0`) ; un kerf exercé ou un multi-formats ne pourra y "
        "entrer qu'après la démonstration d'une marge satisfaisant la règle de rétention sur de "
        "nouvelles familles mesurées.",
        "",
        "## Garanties et limites",
        "",
        "- Les objectifs classiques restent des optimaux sur colonnes générées uniquement "
        "(`optimal_over_generated_columns_only`) ; la marge est définie face à la référence exacte "
        "MILP vérifiée indépendamment, pas face au maître entier restreint.",
        f"- Les écarts gagnables mesurés restent modestes : de {_number(min(retained_gaps))} à "
        f"{_number(max(retained_gaps))} barres par instance retenue, soit "
        f"{_number(sum(retained_gaps))} barres au total sur les "
        f"{len(partitions_manifest['assignments'])} instances du plan.",
        "- Aucune durée n'entre dans ce bilan : la qualité est la métrique reine.",
        "- Aucun écart indisponible ni famille non mesurée n'est filtré silencieusement ; les "
        "diagnostics complets figurent dans le rapport source.",
        "",
        "Le document se régénère depuis les données persistées avec "
        "`uv run python scripts/report_phase8_quality_benchmark.py`.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_phase8_summary(
    margins_report: Mapping[str, Any],
    partitions_manifest: Mapping[str, Any],
    output_path: str | Path,
    *,
    margins_link: str,
    partitions_link: str,
) -> None:
    """Write the factual Phase 8 closure summary from persisted, validated sources.

    Every number is derived from the persisted ``family-margins-v1`` measurement
    and the frozen ``phase-8-quality-partitions-v1`` manifest; the publication
    is refused on any drift between them or when no gap is available against a
    verified exact reference. No duration enters the summary because quality is
    the phase metric.
    """

    _validate_choice_sources(margins_report, partitions_manifest)
    counts = margins_report["counts"]
    if not counts["gap_available_count"]:
        raise ValueError("no gap is available against a verified exact reference")
    environment = margins_report["environment"]
    cross_check = "activé" if margins_report["cross_check_with_enumeration"] else "désactivé"
    instances = margins_report["instances"]
    retained = [family for family in margins_report["families"] if family["retained"]]
    retained_labels = {family["family_label"] for family in retained}
    plan_gaps = [
        entry["gap_bars"]
        for entry in instances
        if entry["gap_available"] and entry["family_label"] in retained_labels
    ]
    unavailable_count = sum(not entry["gap_available"] for entry in instances)
    lines = [
        "# Bilan de la Phase 8",
        "",
        "Sources validées : "
        f"[`{Path(margins_link).name}`]({margins_link}) (schéma `{margins_report['schema_version']}`) "
        f"et [`{partitions_link}`]({partitions_link}) (schéma "
        f"`{partitions_manifest['schema_version']}`), produites par exécutions réelles : une "
        "baseline classique et une référence exacte MILP vérifiée indépendamment par instance. "
        "Aucune durée n'entre dans ce bilan.",
        "",
        "## Méthode, seuil et tolérances",
        "",
        "- Référence : méthode `milp_on_enumerated_patterns`, limites "
        f"`{margins_report['reference_method_limits']}`.",
        f"- Tolérances : coût réduit **{_number(margins_report['reduced_cost_tolerance'])}**, "
        f"intégralité **{_number(margins_report['integrality_tolerance'])}**, faisabilité "
        f"**{_number(margins_report['feasibility_tolerance'])}**.",
        f"- Contrôle croisé d'énumération : **{cross_check}**.",
        "- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un "
        "écart disponible et qu'au moins "
        f"**{margins_report['significant_positive_share'] * 100:.0f} %** des instances perdent au "
        "moins une barre face à leur optimum entier certifié.",
        f"- Environnement tracé : commit `{environment['code_commit'][:12]}…`, Python "
        f"{environment['python_version']}, {environment['dependency_versions']}.",
        "",
        "## Couverture mesurée",
        "",
        f"- Familles déclarées et mesurées : **{counts['family_count']}**, soit "
        f"**{counts['instance_count']}** instances avec baseline classique et référence exacte.",
        f"- {_coverage_line(margins_report['families'])}",
        f"- Écarts disponibles : **{counts['gap_available_count']}**, dont "
        f"**{counts['positive_gap_count']}** positifs ; les instances perdant des barres face à "
        "l'optimum entier certifié existent donc sur ce corpus.",
    ]
    if unavailable_count:
        lines.append(
            f"- {unavailable_count} instance(s) restent sans écart disponible et conservent leur "
            "diagnostic dans le rapport source."
        )
    unmeasured = margins_report["unmeasured_families"]
    if unmeasured:
        lines.append(
            f"- {len(unmeasured)} famille(s) déclarée(s) sans aucune mesure faute de support, non "
            "retenues en l'état : "
            + ", ".join(f"`{item['family_label']}`" for item in unmeasured)
            + "."
        )
    lines += [
        "",
        "## Marge par famille",
        "",
        "| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive "
        "| Écart maximal (barres) | Somme des écarts (barres) | Retenue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family in margins_report["families"]:
        lines.append(_choice_row(family, instances))
    lines += [
        "",
        f"{counts['retained_family_count']} famille(s) retenue(s) sur {counts['family_count']} ; "
        f"part positive globale : {counts['positive_gap_count']} instances positives sur "
        f"{counts['instance_count']} mesurées.",
        "",
        "## Benchmark qualité final retenu",
        "",
        "Le benchmark qualité final de la phase est le plan de partitions gelé "
        f"[`{partitions_link}`]({partitions_link}) (`plan_id` "
        f"`{partitions_manifest['plan_id'][:12]}…`) : exactement les familles retenues ci-dessus, "
        "sans aucune autre. Toute amélioration de qualité sera entraînée sur `train`, ajustée sur "
        "`validation` et mesurée sur `test` ; aucune graine n'est partagée entre deux partitions "
        "et chaque `instance_id` n'y apparaît qu'une fois.",
        "",
        "| Partition | Graines | Instances | Familles |",
        "|---|---|---:|---:|",
    ]
    statistics = partitions_manifest["statistics"]
    for partition in ("train", "validation", "test"):
        lines.append(
            f"| {partition} | {_seed_span(partitions_manifest['seed_partitions'][partition])} "
            f"| {statistics['partition_instance_counts'][partition]} "
            f"| {statistics['partition_family_counts'][partition]} |"
        )
    lines += [
        "",
        f"Sur ces {len(partitions_manifest['assignments'])} instances du plan, la marge mesurée "
        f"s'étend de {_number(min(plan_gaps))} à {_number(max(plan_gaps))} barres par instance, "
        f"soit {_number(sum(plan_gaps))} barres gagnables au total face aux optimaux entiers "
        "certifiés."
        f" {_plan_scope_line(retained)}",
        "",
        "## Garanties et limites",
        "",
        "- Les références proviennent d'un MILP sur motifs énumérés et sont vérifiées "
        "indépendamment (faisabilité du plan, borne LP ≤ optimum).",
        "- Les objectifs classiques restent des optimaux sur colonnes générées uniquement "
        "(`optimal_over_generated_columns_only`) ; la marge est définie face à la référence exacte "
        "vérifiée, pas face au maître entier restreint.",
        "- Aucune durée n'entre dans ce bilan : la qualité est la métrique reine de la phase.",
        "- Aucun écart indisponible ni famille non mesurée n'est filtré silencieusement ; les "
        "diagnostics complets figurent dans le rapport source.",
        "",
        "## Commandes de reproduction",
        "",
        "Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins "
        "publiés :",
        "",
        "```bash",
        "uv sync --extra dev",
        "uv run python scripts/report_phase8_family_margins.py",
        "uv run python scripts/freeze_phase8_partitions.py",
        "uv run python scripts/report_phase8_quality_benchmark.py",
        "uv run python scripts/report_phase8_summary.py",
        "```",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _coverage_line(families: list[Any]) -> str:
    kerfs = sorted({family["configuration"].get("kerf", 0.0) or 0.0 for family in families})
    formats = sorted(
        {
            len(family["configuration"]["stock_lengths"])
            if "stock_lengths" in family["configuration"]
            else 1
            for family in families
        }
    )
    types = sorted(
        {
            family["configuration"]["number_of_types"]
            for family in families
            if "number_of_types" in family["configuration"]
        }
    )
    distributions = sorted(
        {
            family["configuration"]["demand_distribution"]
            for family in families
            if "demand_distribution" in family["configuration"]
        }
    )
    parts = []
    if kerfs[-1] > 0:
        parts.append(f"kerf strictement positif exercé (`kerf = {_number(kerfs[-1])}`)")
    if formats[-1] > 1:
        parts.append(f"multi-formats ({formats[-1]} longueurs de stock)")
    if distributions:
        parts.append(
            f"profils de demande structurés ({', '.join(f'`{d}`' for d in distributions)})"
        )
    if types:
        parts.append(f"montée en taille jusqu'à {_number(types[-1])} types")
    return "Variantes couvertes : " + "; ".join(parts) + "."


def _plan_scope_line(retained: list[Any]) -> str:
    single_format_zero_kerf = all(
        (family["configuration"].get("kerf", 0.0) or 0.0) == 0
        and "stock_lengths" not in family["configuration"]
        for family in retained
    )
    if single_format_zero_kerf:
        return (
            "Le plan ne couvre que des barres à format unique sans kerf exercé (`kerf = 0`) : un "
            "kerf exercé ou un multi-formats ne pourra y entrer qu'après la démonstration d'une "
            "marge satisfaisant la règle de rétention sur de nouvelles familles mesurées."
        )
    return (
        "Le plan couvre les variantes déclarées des familles retenues telles que mesurées dans le "
        "rapport source."
    )


def _validate_choice_sources(
    margins_report: Mapping[str, Any], partitions_manifest: Mapping[str, Any]
) -> None:
    if not isinstance(margins_report, dict):
        raise ValueError("the margin report must be a mapping")
    if margins_report.get("schema_version") != FAMILY_MARGINS_SCHEMA_VERSION:
        raise ValueError(f"unsupported margin report: {margins_report.get('schema_version')!r}")
    if not isinstance(partitions_manifest, dict):
        raise ValueError("the quality partition manifest must be a mapping")
    if partitions_manifest.get("schema_version") != QUALITY_PARTITIONS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported quality partition manifest: {partitions_manifest.get('schema_version')!r}"
        )
    source = partitions_manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != FAMILY_MARGINS_SCHEMA_VERSION
        or source.get("significant_positive_share") != margins_report.get("significant_positive_share")
    ):
        raise ValueError("the frozen partitions must cite the persisted margin measurement")
    measured = sorted(
        family["family_label"] for family in margins_report["families"] if family["retained"]
    )
    if not measured:
        raise ValueError("the margin report retains no family for the quality benchmark")
    frozen = sorted(family["family_label"] for family in partitions_manifest["families"])
    if measured != frozen:
        raise ValueError(
            "the frozen partitions do not cover exactly the retained families: "
            f"measurement retains {measured}, partitions freeze {frozen}"
        )


def _choice_row(family: Mapping[str, Any], instances: list[Any]) -> str:
    share = f"{family['positive_share_of_instances'] * 100:.0f} %"
    gap_sum = sum(
        entry["gap_bars"]
        for entry in instances
        if entry["gap_available"] and entry["family_label"] == family["family_label"]
    )
    return (
        f"| `{family['family_label']}` | {family['instance_count']}"
        f" | {family['gap_available_count']} | {family['zero_gap_count']}"
        f" | {family['positive_gap_count']} | {share}"
        f" | {family['max_gap_bars']} | {gap_sum}"
        f" | {'oui' if family['retained'] else 'non'} |"
    )


def _seed_span(seeds: list[int]) -> str:
    if len(seeds) > 2 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}–{seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


__all__ = [
    "write_family_margins_markdown",
    "write_phase8_summary",
    "write_quality_benchmark_choice",
]
