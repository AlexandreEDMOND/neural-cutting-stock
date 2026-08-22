# Bilan de la Phase 8

Sources validées : [`phase-8-family-margins.json`](phase-8-family-margins.json) (schéma `family-margins-v1`) et [`../data/phase-8-partitions/manifest.json`](../data/phase-8-partitions/manifest.json) (schéma `phase-8-quality-partitions-v1`), produites par exécutions réelles : une baseline classique et une référence exacte MILP vérifiée indépendamment par instance. Aucune durée n'entre dans ce bilan.

## Méthode, seuil et tolérances

- Référence : méthode `milp_on_enumerated_patterns`, limites `maximal_patterns:max_search_space_size=10000000,max_patterns=100000`.
- Tolérances : coût réduit **1e-09**, intégralité **1e-09**, faisabilité **1e-09**.
- Contrôle croisé d'énumération : **désactivé**.
- Règle de rétention : une famille est retenue lorsque chaque instance mesurée dispose d'un écart disponible et qu'au moins **50 %** des instances perdent au moins une barre face à leur optimum entier certifié.
- Environnement tracé : commit `c1f2a8bac4e9…`, Python 3.11.15, numpy==2.4.6,scipy==1.17.1.

## Couverture mesurée

- Familles déclarées et mesurées : **6**, soit **36** instances avec baseline classique et référence exacte.
- Variantes couvertes : kerf strictement positif exercé (`kerf = 2`); multi-formats (2 longueurs de stock); profils de demande structurés (`awkward_divisibility_v1`, `uniform_integer_v1`); montée en taille jusqu'à 12 types.
- Écarts disponibles : **36**, dont **22** positifs ; les instances perdant des barres face à l'optimum entier certifié existent donc sur ce corpus.

## Marge par famille

| Famille | Instances | Écarts disponibles | Marge nulle | Marge positive | Part positive | Écart maximal (barres) | Somme des écarts (barres) | Retenue |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `kerf-exercised-uniform-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 1 | 1 | non |
| `kerf-exercised-uniform-t6-v1` | 6 | 6 | 4 | 2 | 33 % | 1 | 2 | non |
| `multi-stock-formats-t4-v1` | 6 | 6 | 5 | 1 | 17 % | 1 | 1 | non |
| `scaled-tight-divisibility-t12-v1` | 6 | 6 | 0 | 6 | 100 % | 6 | 36 | oui |
| `structured-tight-divisibility-t3-v1` | 6 | 6 | 0 | 6 | 100 % | 1 | 6 | oui |
| `structured-tight-divisibility-t4-v1` | 6 | 6 | 0 | 6 | 100 % | 2 | 12 | oui |

3 famille(s) retenue(s) sur 6 ; part positive globale : 22 instances positives sur 36 mesurées.

## Benchmark qualité final retenu

Le benchmark qualité final de la phase est le plan de partitions gelé [`../data/phase-8-partitions/manifest.json`](../data/phase-8-partitions/manifest.json) (`plan_id` `b8ba8c2065d6…`) : exactement les familles retenues ci-dessus, sans aucune autre. Toute amélioration de qualité sera entraînée sur `train`, ajustée sur `validation` et mesurée sur `test` ; aucune graine n'est partagée entre deux partitions et chaque `instance_id` n'y apparaît qu'une fois.

| Partition | Graines | Instances | Familles |
|---|---|---:|---:|
| train | 1–3 | 9 | 3 |
| validation | 4 | 3 | 3 |
| test | 5, 6 | 6 | 3 |

Sur ces 18 instances du plan, la marge mesurée s'étend de 1 à 6 barres par instance, soit 54 barres gagnables au total face aux optimaux entiers certifiés. Le plan ne couvre que des barres à format unique sans kerf exercé (`kerf = 0`) : un kerf exercé ou un multi-formats ne pourra y entrer qu'après la démonstration d'une marge satisfaisant la règle de rétention sur de nouvelles familles mesurées.

## Garanties et limites

- Les références proviennent d'un MILP sur motifs énumérés et sont vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum).
- Les objectifs classiques restent des optimaux sur colonnes générées uniquement (`optimal_over_generated_columns_only`) ; la marge est définie face à la référence exacte vérifiée, pas face au maître entier restreint.
- Aucune durée n'entre dans ce bilan : la qualité est la métrique reine de la phase.
- Aucun écart indisponible ni famille non mesurée n'est filtré silencieusement ; les diagnostics complets figurent dans le rapport source.

## Commandes de reproduction

Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :

```bash
uv sync --extra dev
uv run python scripts/report_phase8_family_margins.py
uv run python scripts/freeze_phase8_partitions.py
uv run python scripts/report_phase8_quality_benchmark.py
uv run python scripts/report_phase8_summary.py
```
