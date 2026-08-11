# Bilan de publication de la Phase 4

Source brute validée : `results/phase-4-benchmark-runs.csv` (schéma `benchmark-run-v1`).

## Couverture

- Exécutions : **8** ; paires : **4**.
- Paires comparables : **4** ; paires à qualité préservée : **4**.
- Les différences d'objectif et les speedups ont été recalculés depuis les enregistrements bruts ; aucune valeur dérivée persistée n'est utilisée.

| Taille | Paires admissibles | Médiane Classical (s) | Médiane Neural (s) | Médiane speedup |
|---|---:|---:|---:|---:|
| SMALL | 3 | 0.006379 | 0.007853 | 1.029572 |
| MEDIUM | 1 | 0.006240 | 0.018792 | 0.332036 |
| LARGE | 0 | n/a | n/a | n/a |
| XL | 0 | n/a | n/a | n/a |

## Garanties et limites

Toutes les exécutions restent représentées, y compris les statuts non réussis. Une paire n'alimente les figures que si les deux plans sont faisables, convergés et ont une différence d'objectif nulle. Le contrôle exact du pricing et la vérification indépendante du plan restent ceux du solveur classique.

Le corpus Phase 3 ne contient aucun candidat sélectionné : cette publication ne prétend donc pas mesurer la qualité d'un modèle entraîné sur ce corpus. Les mesures publiées décrivent uniquement la politique et le modèle identifiés dans les enregistrements bruts.

## Figures

- [`runtime_comparison.png`](runtime_comparison.png) : temps mur-à-mur médian des paires admissibles.
- [`speedup_by_size.png`](speedup_by_size.png) : speedup médian des mêmes paires, avec référence `1x`.
