"""Phase 6 paired publication tables, figures and summary from validated final results."""

# Publication prose intentionally follows the report format.
# ruff: noqa: E501

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks import SIZE_CLASSES, BenchmarkRunRecord, build_paired_tables
from neural_cutting_stock.benchmarks.stats import median
from neural_cutting_stock.visualization._shared import number as _number
from neural_cutting_stock.visualization._shared import seconds


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
            f" | {seconds(item['classical_runtime_seconds_median'])} | {seconds(item['neural_runtime_seconds_median'])}"
            f" | {seconds(item['speedup_vs_classical_median'])} |"
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
    repetitions feed the runtime and speedup medians. Grouping uses the frozen
    pre-evaluation ``target_size_class`` of the final instance manifest, not
    the per-run measured size class, so both modes are grouped on the same
    strata.
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
            "classical_median_seconds": median(
                [row["classical_total_runtime_seconds"] for row in rows]
            ),
            "neural_median_seconds": median(
                [row["neural_total_runtime_seconds"] for row in rows]
            ),
            "speedup_median": median([row["speedup_vs_classical"] for row in rows]),
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


def write_phase6_speedup_by_size(data: dict[str, Any], output_dir: str | Path) -> None:
    """Write ``speedup_by_size.png`` from admissible final pairs only."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = [size for size in SIZE_CLASSES if data["size_data"][size]["pair_count"]]
    if not sizes:
        raise ValueError("no admissible pair is available for the speedup by size figure")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.plot(sizes, [data["size_data"][size]["speedup_median"] for size in sizes], "o-")
    axis.set_xlabel("Target size class (size-class-v1)")
    axis.set_ylabel("Median speedup (Classical / Neural)")
    axis.set_title("Final evaluation: measured paired speedup at preserved quality")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "speedup_by_size.png", dpi=160)
    plt.close(figure)


def write_phase6_summary(
    data: dict[str, Any],
    output_path: str | Path,
    *,
    classical_source: str,
    neural_source: str,
    config_path: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    final_manifest_path: str | Path,
    model_artifact_path: str | Path,
) -> None:
    """Write the factual final Phase 6 publication summary.

    Every number is derived from the validated paired report and the frozen
    configuration; the manifest and model identities are hashed from the
    published files themselves. No result is invented or imported.
    """

    report = data["report"]
    protocol = config["protocol"]
    status_counts: dict[str, int] = {}
    for row in report["pairs"]:
        for field in ("classical_run_status", "neural_run_status"):
            status_counts[row[field]] = status_counts.get(row[field], 0) + 1
    statuses = ", ".join(f"`{name}` : {count}" for name, count in sorted(status_counts.items()))
    quality_violations = sum(item["quality_violation_pair_count"] for item in report["instances"])
    measured_instances = [
        item for item in report["instances"] if item["admissible_repetition_count"]
    ]
    if not measured_instances:
        raise ValueError("no admissible pair is available for the final summary")
    speedups = [item["speedup_vs_classical_median"] for item in measured_instances]
    faster = sum(speedup > 1.0 for speedup in speedups)
    slower = sum(speedup < 1.0 for speedup in speedups)
    statistics = manifest["statistics"]
    strata = ", ".join(
        f"{size} {statistics['target_size_class_counts'].get(size, 0)}" for size in SIZE_CLASSES
    )
    runtime_limit = (
        "aucune limite de temps"
        if protocol["max_runtime_seconds"] is None
        else f"max_runtime_seconds {protocol['max_runtime_seconds']} s"
    )
    iteration_limit = (
        "aucune limite d'itérations"
        if protocol["max_cg_iterations"] is None
        else f"max_cg_iterations {protocol['max_cg_iterations']}"
    )
    lines = [
        "# Bilan final de la Phase 6",
        "",
        f"Sources brutes validées : `{_relative(classical_source)}` et `{_relative(neural_source)}` (schéma `benchmark-run-v1`).",
        "La conclusion scientifique complète est publiée dans [`docs/conclusion.md`](../docs/conclusion.md).",
        "",
        "## Protocole gelé",
        "",
        f"- Configuration : `{_relative(config_path)}` (`{config['schema_version']}`), comparaison appariée par `{protocol['comparison']}`, {protocol['repetitions']} répétitions par instance et par mode, ordre d'exécution `{protocol['execution_order']}`, modèle `{protocol['model_loading']}`.",
        f"- Tolérances : différence d'objectif **{protocol['quality_tolerance_bars']:g} barre(s)**, coût réduit **{protocol['reduced_cost_tolerance']:g}**.",
        f"- Budgets : {runtime_limit} ; {iteration_limit}.",
        f"- Instances non vues : **{statistics['instance_count']}** hors corpus Phase 3 ({strata}).",
        f"- Modèle évalué : [`{config['model']['artifact']}`](../{config['model']['artifact']}) (`{config['model']['model_id']}`, politique `{config['model']['policy']}`).",
        "",
        "## Couverture et qualité",
        "",
        f"- Exécutions : **{report['run_count']}** ; paires : **{report['pair_count']}** ; paires admissibles : **{report['admissible_pair_count']}**.",
        f"- Violations de qualité : **{quality_violations}** paire(s) à la tolérance déclarée.",
        f"- Statuts terminaux : {statuses}.",
        "- Les différences d'objectif et les speedups sont recalculés depuis les enregistrements bruts ; chaque paire, y compris échec ou violation, reste conservée dans [`phase-6-paired-tables.json`](phase-6-paired-tables.json).",
        "",
        "## Runtime mur-à-mur par strate cible",
        "",
        "| Strate cible | Instances | Paires admissibles | Classical médian (s) | Neural médian (s) | Speedup médian |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for size in SIZE_CLASSES:
        item = data["size_data"][size]
        lines.append(
            f"| {size} | {_count(item['instance_count'])} | {_count(item['pair_count'])}"
            f" | {seconds(item['classical_median_seconds'])} | {seconds(item['neural_median_seconds'])}"
            f" | {seconds(item['speedup_median'])} |"
        )
    lines += [
        "",
        f"- Speedup médian par instance : de **{min(speedups):.6f}** à **{max(speedups):.6f}** ; Neural CG est plus lent sur {slower} des {len(measured_instances)} instances et plus rapide sur {faster}.",
        "- Les strates reprennent le `target_size_class` figé du manifeste final, jamais le `size_class` mesuré par enregistrement.",
        "",
        "## Rapports détaillés",
        "",
        "- Tableaux appariés : [`phase-6-paired-tables.md`](phase-6-paired-tables.md) et [`phase-6-paired-tables.json`](phase-6-paired-tables.json).",
        "- Échecs, timeouts et violations : [`phase-6-failures.json`](phase-6-failures.json).",
        "- Généralisation au-delà des tailles d'entraînement : [`phase-6-generalization.json`](phase-6-generalization.json).",
        "- Incertitude des répétitions : [`phase-6-uncertainty.json`](phase-6-uncertainty.json).",
        "- Figures issues des paires admissibles : [`runtime_comparison.png`](runtime_comparison.png) et [`speedup_by_size.png`](speedup_by_size.png).",
        "",
        "## Garanties et limites",
        "",
        "Le pricing exact certifie l'optimalité de la relaxation linéaire du maître complet à la tolérance déclarée lorsqu'aucune colonne améliorante n'est trouvée. Le maître entier final est résolu sur les colonnes générées uniquement : ses objectifs restent qualifiés `optimal_over_generated_columns_only`, sans preuve d'optimalité entière globale. Chaque plan est vérifié indépendamment pour la demande, la capacité, le kerf et l'objectif. Aucune exécution n'est filtrée des sources ; les campagnes proviennent d'un environnement matériel unique tracé dans leurs métadonnées, donc seules les comparaisons appariées intra-campagne sont interprétables.",
        "",
        "## Manifeste final et artefacts",
        "",
        f"- Manifeste : `{_relative(final_manifest_path)}` (`{manifest['schema_version']}`), `manifest_id` `{manifest['manifest_id']}`, SHA-256 `{_sha256(final_manifest_path)}`.",
        f"- Artefact modèle : `{_relative(model_artifact_path)}`, SHA-256 `{_sha256(model_artifact_path)}`.",
        "",
        "## Commandes de reproduction",
        "",
        "Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :",
        "",
        "```bash",
        "uv sync --extra dev",
        "uv run python scripts/generate_phase6_manifest.py",
        "uv run python scripts/run_phase6_classical.py",
        "uv run python scripts/run_phase6_neural.py",
        "uv run python scripts/report_phase6_uncertainty.py",
        "uv run python scripts/report_phase6_failures.py",
        "uv run python scripts/report_phase6_generalization.py",
        "uv run python scripts/report_phase6_paired_tables.py",
        "uv run python scripts/plot_phase6_results.py",
        "uv run python scripts/report_phase6_summary.py",
        "```",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(source: str) -> str:
    with suppress(ValueError):
        return str(Path(source).resolve().relative_to(Path.cwd().resolve()))
    return str(source)


def _count(value: float | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
