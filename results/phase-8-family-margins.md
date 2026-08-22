# Marges de qualité des nouvelles familles (Phase 8)

Source validée : [`phase-8-family-margins.json`](phase-8-family-margins.json) (schéma `family-margins-v1`), produite par exécutions réelles : une baseline classique et une référence exacte MILP vérifiée indépendamment par instance. Aucune durée n'entre dans ce bilan.

## Méthode, seuil et tolérances

- Référence : méthode `milp_on_enumerated_patterns`, limites `maximal_patterns:max_search_space_size=10000000,max_patterns=100000`.
- Tolérances : coût réduit **1e-09**, intégralité **1e-09**, faisabilité **1e-09**.
- Contrôle croisé d'énumération : **désactivé**.
- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un écart disponible et qu'au moins **50 %** des instances perdent au moins une barre face à leur optimum entier certifié.
- Environnement tracé : commit `c1f2a8bac4e9…`, Python 3.11.15, numpy==2.4.6,scipy==1.17.1.

## Marge par famille

| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive | Retenue |
|---|---:|---:|---:|---:|---:|---|
| `kerf-exercised-uniform-t4-v1` | 6 | 6 | 5 | 1 | 17 % | non |
| `kerf-exercised-uniform-t6-v1` | 6 | 6 | 4 | 2 | 33 % | non |
| `multi-stock-formats-t4-v1` | 6 | 6 | 5 | 1 | 17 % | non |
| `scaled-tight-divisibility-t12-v1` | 6 | 6 | 0 | 6 | 100 % | oui |
| `structured-tight-divisibility-t3-v1` | 6 | 6 | 0 | 6 | 100 % | oui |
| `structured-tight-divisibility-t4-v1` | 6 | 6 | 0 | 6 | 100 % | oui |

3 famille(s) retenue(s) sur 6 ; part positive globale : 22 instances positives sur 36 mesurées.

## Instances à marge positive

| Instance | Famille | Types | Optimum entier | Baseline classique | Écart (barres) |
|---|---|---:|---:|---:|---:|
| `02be7ee4ce40…` | `scaled-tight-divisibility-t12-v1` | 12 | 341 | 347 | 6 |
| `22a9d5d9c1b3…` | `scaled-tight-divisibility-t12-v1` | 12 | 419 | 425 | 6 |
| `2e4b00a9b8a5…` | `scaled-tight-divisibility-t12-v1` | 12 | 398 | 404 | 6 |
| `4f9c63312c5b…` | `scaled-tight-divisibility-t12-v1` | 12 | 394 | 400 | 6 |
| `c8f9eb999d72…` | `scaled-tight-divisibility-t12-v1` | 12 | 391 | 397 | 6 |
| `f890cf01ef15…` | `scaled-tight-divisibility-t12-v1` | 12 | 299 | 305 | 6 |
| `01b5a876d7a4…` | `structured-tight-divisibility-t4-v1` | 4 | 37 | 39 | 2 |
| `23e0fe633b02…` | `structured-tight-divisibility-t4-v1` | 4 | 34 | 36 | 2 |
| `701c0ae70701…` | `structured-tight-divisibility-t4-v1` | 4 | 26 | 28 | 2 |
| `ddd6241be3d7…` | `structured-tight-divisibility-t4-v1` | 4 | 43 | 45 | 2 |
| `dea0eba30d5f…` | `structured-tight-divisibility-t4-v1` | 4 | 44 | 46 | 2 |
| `f9a8b35dbc2b…` | `structured-tight-divisibility-t4-v1` | 4 | 42 | 44 | 2 |
| `f7ddc2e4a2c5…` | `kerf-exercised-uniform-t4-v1` | 4 | 38 | 39 | 1 |
| `06d43443e538…` | `kerf-exercised-uniform-t6-v1` | 6 | 46 | 47 | 1 |
| `9fd418fd0c65…` | `kerf-exercised-uniform-t6-v1` | 6 | 49 | 50 | 1 |
| `5ef6f13e68b5…` | `multi-stock-formats-t4-v1` | 4 | 20 | 21 | 1 |
| `5250b4ad75b9…` | `structured-tight-divisibility-t3-v1` | 3 | 22 | 23 | 1 |
| `577a1c8a265a…` | `structured-tight-divisibility-t3-v1` | 3 | 34 | 35 | 1 |
| `8a70203d87d4…` | `structured-tight-divisibility-t3-v1` | 3 | 16 | 17 | 1 |
| `8df29b8c097c…` | `structured-tight-divisibility-t3-v1` | 3 | 23 | 24 | 1 |
| `c45e8dd7bdce…` | `structured-tight-divisibility-t3-v1` | 3 | 34 | 35 | 1 |
| `cd3f91ce0ad7…` | `structured-tight-divisibility-t3-v1` | 3 | 27 | 28 | 1 |

## Garanties et limites

Les références proviennent d'un MILP sur motifs énumérés et sont vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum). Les objectifs classiques restent des optimaux sur colonnes générées uniquement (`optimal_over_generated_columns_only`) et ne préjugent pas d'un optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les raisons complètes figurent dans le rapport source.

Le bilan se régénère depuis les données persistées avec `uv run python scripts/report_phase8_family_margins.py`.
