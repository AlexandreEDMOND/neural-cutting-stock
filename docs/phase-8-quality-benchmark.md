# Choix du benchmark qualité final (Phase 8)

Bilan intermédiaire de la Phase 8 : tableau des marges mesurées par famille et choix documenté du benchmark qualité final utilisé par les phases suivantes. Sources validées : [`../results/phase-8-family-margins.json`](../results/phase-8-family-margins.json) (schéma `family-margins-v1`) et [`../data/phase-8-partitions/manifest.json`](../data/phase-8-partitions/manifest.json) (schéma `phase-8-quality-partitions-v1`). Toutes les valeurs proviennent d'exécutions réelles ; aucune durée n'entre dans ce bilan.

## Méthode de mesure et règle de rétention

- Référence : méthode `milp_on_enumerated_patterns`, limites `maximal_patterns:max_search_space_size=10000000,max_patterns=100000`.
- Tolérances : coût réduit **1e-09**, intégralité **1e-09**, faisabilité **1e-09**.
- Contrôle croisé d'énumération : **désactivé**.
- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un écart disponible et qu'au moins **50 %** des instances perdent au moins une barre face à leur optimum entier certifié.
- Environnement tracé : commit `c1f2a8bac4e9…`, Python 3.11.15, numpy==2.4.6,scipy==1.17.1.

## Marges par famille

| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive | Écart maximal (barres) | Somme des écarts (barres) | Retenue |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `kerf-exercised-uniform-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 1 | 1 | non |
| `kerf-exercised-uniform-t6-v1` | 6 | 6 | 4 | 2 | 33 % | 1 | 2 | non |
| `multi-stock-formats-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 1 | 1 | non |
| `scaled-tight-divisibility-t12-v1` | 6 | 6 | 0 | 6 | 100 % | 6 | 36 | oui |
| `structured-tight-divisibility-t3-v1` | 6 | 6 | 0 | 6 | 100 % | 1 | 6 | oui |
| `structured-tight-divisibility-t4-v1` | 6 | 6 | 0 | 6 | 100 % | 2 | 12 | oui |

3 famille(s) retenue(s) sur 6 ; part positive globale : 22 instances positives sur 36 mesurées.

## Choix du benchmark qualité final

Le benchmark qualité final retenu pour les phases suivantes est le plan de partitions gelé [`../data/phase-8-partitions/manifest.json`](../data/phase-8-partitions/manifest.json) (`plan_id` `b8ba8c2065d6…`) : exactement les familles retenues ci-dessus, sans aucune autre. Toute amélioration de qualité revendiquée sera entraînée sur `train`, ajustée sur `validation` et mesurée sur `test` de ces partitions ; aucune graine n'est partagée entre deux partitions et chaque `instance_id` n'y apparaît qu'une fois.

| Partition | Graines | Instances | Familles |
|---|---|---:|---:|
| train | 1–3 | 9 | 3 |
| validation | 4 | 3 | 3 |
| test | 5, 6 | 6 | 3 |

## Familles écartées du benchmark qualité final

Les familles non retenues restent modélisées et exécutables (générateurs, solveur, schémas) mais n'entrent pas dans le benchmark qualité final : leur marge mesurée ne satisfait pas la règle de rétention, si bien qu'une amélioration de qualité y serait invérifiable face à l'optimum entier certifié.

| Famille | Part positive | Instances à marge positive | Écart maximal (barres) |
|---|---:|---:|---:|
| `kerf-exercised-uniform-t4-v1` | 17 % | 1/6 | 1 |
| `kerf-exercised-uniform-t6-v1` | 33 % | 2/6 | 1 |
| `multi-stock-formats-t4-v1` | 17 % | 1/6 | 1 |

Conséquence documentée : le benchmark qualité final ne couvre que des barres à format unique, sans kerf exercé (`kerf = 0`) ; un kerf exercé ou un multi-formats ne pourra y entrer qu'après la démonstration d'une marge satisfaisant la règle de rétention sur de nouvelles familles mesurées.

## Garanties et limites

- Les objectifs classiques restent des optimaux sur colonnes générées uniquement (`optimal_over_generated_columns_only`) ; la marge est définie face à la référence exacte MILP vérifiée indépendamment, pas face au maître entier restreint.
- Les écarts gagnables mesurés restent modestes : de 1 à 6 barres par instance retenue, soit 54 barres au total sur les 18 instances du plan.
- Aucune durée n'entre dans ce bilan : la qualité est la métrique reine.
- Aucun écart indisponible ni famille non mesurée n'est filtré silencieusement ; les diagnostics complets figurent dans le rapport source.

Le document se régénère depuis les données persistées avec `uv run python scripts/report_phase8_quality_benchmark.py`.
