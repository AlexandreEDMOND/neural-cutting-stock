# Bilan de publication de la Phase 5

Source brute validée : `results/phase-4-benchmark-runs.csv` (schéma `benchmark-run-v1`).

## Décision

Le candidat `linear-scorer-v1-zero-weight` n'est pas gelé : **no_total_runtime_improvement**.
La politique supervisée bornée de Phase 4 reste donc le candidat retenu, avec le pricing exact et le fallback exact.
La comparaison porte sur le temps mur-à-mur total agrégé, et non sur le seul temps d'inférence.

## Couverture et mesure

- Exécutions : **8** ; paires : **4**.
- Paires comparables : **4** ; paires à qualité préservée : **4**.
- Runtime Classical agrégé : **0.024098 s**.
- Runtime Neural agrégé : **0.038581 s**.
- Les différences d'objectif et les speedups ont été recalculés depuis les enregistrements bruts.

| Taille | Paires admissibles | Médiane Classical (s) | Médiane Neural (s) | Médiane speedup |
|---|---:|---:|---:|---:|
| SMALL | 3 | 0.006379 | 0.007853 | 1.029572 |
| MEDIUM | 1 | 0.006240 | 0.018792 | 0.332036 |
| LARGE | 0 | n/a | n/a | n/a |
| XL | 0 | n/a | n/a | n/a |

## Garanties et limites

Toutes les lignes de la source sont conservées dans l'analyse. Les figures n'utilisent que les paires dont les deux plans sont faisables, convergés et de même objectif. Le pricing exact, le fallback exact et la vérification indépendante restent obligatoires.
La couverture ne contient aucune paire LARGE ou XL ; ces mesures ne démontrent donc pas un gain généralisable sur les grandes instances.

## Figures

- [`phase5_runtime_comparison.png`](phase5_runtime_comparison.png) : runtime médian des paires à qualité préservée.
- [`phase5_speedup_by_size.png`](phase5_speedup_by_size.png) : speedup médian avec référence `1x`.
