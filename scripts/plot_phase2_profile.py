"""Generate the persisted-data figures and Markdown report for Phase 2."""

# Markdown table and prose lines intentionally follow the report format.
# ruff: noqa: E501

import argparse
from pathlib import Path

from neural_cutting_stock.visualization._shared import seconds
from neural_cutting_stock.visualization.phase2 import (
    load_phase2_profile,
    phase2_report_data,
    write_phase2_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = load_phase2_profile(args.profile)
    write_phase2_figures(profile, args.output_dir)
    _write_summary(profile, args.output_dir / "phase-2-summary.md")


def _write_summary(profile: dict, path: Path) -> None:
    data = phase2_report_data(profile)
    successful = data["successful_runs"]
    status_lines = "\n".join(
        f"- `{status}`: {count}"
        for status, count in sorted(profile["status_counts"].items())
    )
    size_lines = "\n".join(
        f"| {size} | {values['count']} | {seconds(values['runtime_median_seconds'])} | "
        f"{seconds(values['runtime_min_seconds'])}--{seconds(values['runtime_max_seconds'])} | "
        f"{values['iterations_median']} |"
        for size, values in data["size_data"].items()
    )
    component_lines = "\n".join(
        f"| `{component}` | {profile['component_totals_seconds'][component]:.6f} | "
        f"{profile['component_shares'][component] * 100:.2f}% |"
        for component in profile["component_totals_seconds"]
    )
    text = f"""# Bilan de publication de la Phase 2

Source unique : [`phase-2-baseline-profile.json`](phase-2-baseline-profile.json), profil persistant `{profile['profile_schema_version']}`, catégories `{profile['size_class_schema_version']}`.

## Corpus et traçabilité

- Exécutions enregistrées : **{profile['run_count']}** ; succès : **{profile['successful_run_count']}**.
- Commit mesuré dans les enregistrements : {_distinct(successful, 'code_commit')}.
- Environnement : {_distinct(successful, 'python_version')}, {_distinct(successful, 'dependency_versions')}, {_distinct(successful, 'hardware_id')}.
- Configuration : {_distinct(successful, 'config_id')} ; graines observées : {_distinct(successful, 'seed')}.
- Tous les enregistrements sont `classical` et les plans réussis sont marqués faisables.

Statuts conservés :
{status_lines}

## Profil mesuré

| Catégorie | Exécutions réussies | Médiane runtime (s) | Min--max runtime (s) | Médiane itérations CG |
|---|---:|---:|---:|---:|
{size_lines}

| Composant | Total (s) | Part du temps instrumenté |
|---|---:|---:|
{component_lines}

Le composant dominant mesuré est **`{profile['dominant_component']}`**. Les parts ci-dessus sont calculées sur les composants instrumentés, incluant `unattributed_runtime`; elles ne constituent pas un speedup.

## Figures

- [`classical_runtime_by_size.png`](classical_runtime_by_size.png) montre la médiane du temps mur-à-mur par catégorie `size-class-v1`, avec l'intervalle min--max du corpus disponible.
- [`classical_runtime_components.png`](classical_runtime_components.png) montre les temps cumulés enregistrés par composant.

Les deux figures ont été générées depuis le JSON source par `scripts/plot_phase2_profile.py`. Aucune comparaison Neural CG, aucun speedup et aucune extrapolation de performance ne sont publiés : les données de cette phase ne contiennent que la baseline classique.
"""
    path.write_text(text, encoding="utf-8")


def _distinct(records: list[dict], field: str) -> str:
    values = sorted({str(record[field]) for record in records})
    return ", ".join(f"`{value}`" for value in values)


if __name__ == "__main__":
    main()
