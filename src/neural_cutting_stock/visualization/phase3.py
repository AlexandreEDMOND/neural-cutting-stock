"""Publication figures derived from the persisted Phase 3 trajectory corpus."""

# Markdown report lines intentionally follow the publication format.
# ruff: noqa: E501

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neural_cutting_stock.benchmarks import (
    corpus_statistics,
    read_corpus_manifest,
    read_trajectory,
    replay_trajectory,
    trajectory_sha256,
)


def load_phase3_corpus(path: str | Path) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Load, hash-check, and replay every trajectory listed in a Phase 3 manifest."""

    manifest_path = Path(path)
    manifest = read_corpus_manifest(manifest_path)
    trajectories = tuple(
        read_trajectory(manifest_path.parent / entry["path"])
        for entry in manifest["trajectories"]
    )
    entries = manifest["trajectories"]
    if [trajectory.metadata.trajectory_id for trajectory in trajectories] != [
        entry["trajectory_id"] for entry in entries
    ]:
        raise ValueError("manifest trajectory order or identity differs")
    for trajectory, entry in zip(trajectories, entries, strict=True):
        if trajectory_sha256(trajectory) != entry["sha256"]:
            raise ValueError(f"trajectory hash differs: {entry['trajectory_id']}")
        validation = replay_trajectory(trajectory)
        if not validation.valid:
            raise ValueError(
                f"trajectory {entry['trajectory_id']} is invalid: " + "; ".join(validation.errors)
            )
    partitions = {entry["trajectory_id"]: entry["partition"] for entry in entries}
    if corpus_statistics(trajectories, partitions) != manifest["statistics"]:
        raise ValueError("manifest statistics differ from the persisted trajectories")
    return manifest, trajectories


def phase3_report_data(manifest: dict[str, Any], trajectories: tuple[Any, ...]) -> dict[str, Any]:
    """Compute descriptive corpus data without adding measurements to the source corpus."""

    partition_by_id = {
        entry["trajectory_id"]: entry["partition"] for entry in manifest["trajectories"]
    }
    by_partition: dict[str, dict[str, int]] = {}
    for partition in ("train", "validation", "test"):
        selected = [
            item for item in trajectories if partition_by_id[item.metadata.trajectory_id] == partition
        ]
        by_partition[partition] = {
            "trajectory_count": len(selected),
            "iteration_count": sum(len(item.iterations) for item in selected),
            "piece_type_count": sum(len(item.metadata.piece_lengths) for item in selected),
        }
    return {
        "by_partition": by_partition,
        "number_of_piece_types": [len(item.metadata.piece_lengths) for item in trajectories],
        "total_demands": [sum(item.metadata.demands) for item in trajectories],
        "status_counts": dict(Counter(item.status.value for item in trajectories)),
    }


def write_phase3_figures(
    manifest: dict[str, Any], trajectories: tuple[Any, ...], output_dir: str | Path
) -> None:
    """Write descriptive Phase 3 figures using only validated corpus values."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = phase3_report_data(manifest, trajectories)
    partitions = ("train", "validation", "test")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x = range(len(partitions))
    axis.bar(
        [index - 0.18 for index in x],
        [data["by_partition"][partition]["iteration_count"] for partition in partitions],
        width=0.36,
        label="Iterations",
    )
    axis.bar(
        [index + 0.18 for index in x],
        [data["by_partition"][partition]["piece_type_count"] for partition in partitions],
        width=0.36,
        label="Piece types",
    )
    axis.set_xticks(tuple(x), partitions)
    axis.set_ylabel("Count")
    axis.set_title("Validated Phase 3 corpus structure")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "phase3_corpus_structure.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for partition, piece_types, total_demand in zip(
        (manifest_entry["partition"] for manifest_entry in manifest["trajectories"]),
        data["number_of_piece_types"],
        data["total_demands"],
        strict=True,
    ):
        axis.scatter(piece_types, total_demand, label=partition, s=70)
    axis.set_xlabel("Number of piece types")
    axis.set_ylabel("Total demand")
    axis.set_title("Validated trajectory instance dimensions")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "phase3_instance_dimensions.png", dpi=160)
    plt.close(figure)


def write_phase3_summary(manifest: dict[str, Any], trajectories: tuple[Any, ...], path: str | Path) -> None:
    """Write the factual Phase 3 publication summary."""

    data = phase3_report_data(manifest, trajectories)
    rows = "\n".join(
        f"| {partition} | {values['trajectory_count']} | {values['iteration_count']} | "
        f"{values['piece_type_count']} |"
        for partition, values in data["by_partition"].items()
    )
    text = f"""# Bilan de publication de la Phase 3

Source unique : [`manifest.json`](../data/phase-3-corpus/manifest.json), corpus `{manifest['corpus_id']}`, schéma `{manifest['schema_version']}`.

## Corpus validé

- Trajectoires persistées : **{manifest['statistics']['trajectory_count']}** ; instances : **{manifest['statistics']['instance_count']}**.
- Statuts : {', '.join(f'`{status}`: {count}' for status, count in sorted(data['status_counts'].items()))}.
- Répartition : train **{data['by_partition']['train']['trajectory_count']}**, validation **{data['by_partition']['validation']['trajectory_count']}**, test **{data['by_partition']['test']['trajectory_count']}**.
- Itérations enregistrées : **{manifest['statistics']['iteration_count']}** ; colonnes ajoutées : **{manifest['statistics']['columns_added']}** ; motifs sélectionnés : **{manifest['statistics']['selected_pattern_count']}**.
- Validation : chaque hash SHA-256 du manifeste correspond au JSON persistant et chaque trajectoire a été rejouée par le solveur classique exact sans erreur.

| Partition | Trajectoires | Itérations | Types de pièces |
|---|---:|---:|---:|
{rows}

## Portée et limites

Le corpus est un petit corpus de collecte, pas un benchmark de temps : les durées de collecte ne sont pas agrégées ni interprétées comme une performance. Il ne contient aucune colonne candidate enregistrée, aucune colonne ajoutée et aucun exemple de dataset (`selected_pattern_count = 0`). Il ne permet donc pas d'évaluer un modèle appris, un classement de colonnes ou un speedup.

L'environnement déclaré est `{manifest['environment']['python_version']}`, `{manifest['environment']['dependency_versions']}`, `{manifest['environment']['hardware_id']}` ; le commit déclaré par les trajectoires est `{manifest['environment']['code_commit']}`. La régénération et le rejeu sont documentés dans [`data/phase-3-corpus/README.md`](../data/phase-3-corpus/README.md).

## Figures

- [`phase3_corpus_structure.png`](phase3_corpus_structure.png) représente les comptes d'itérations et de types de pièces par partition.
- [`phase3_instance_dimensions.png`](phase3_instance_dimensions.png) représente les dimensions des trois instances persistées.

Les deux figures sont descriptives et proviennent exclusivement du manifeste et des trajectoires validées ; elles ne montrent ni runtime, ni comparaison Classical CG/Neural CG, ni résultat scientifique extrapolé.
"""
    Path(path).write_text(text, encoding="utf-8")
