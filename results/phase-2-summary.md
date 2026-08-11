# Bilan de publication de la Phase 2

Source unique : [`phase-2-baseline-profile.json`](phase-2-baseline-profile.json), profil persistant `baseline-profile-v1`, catégories `size-class-v1`.

## Corpus et traçabilité

- Exécutions enregistrées : **8** ; succès : **8**.
- Commit mesuré dans les enregistrements : `6375ebe4b5de95d0ed696d6a4b3a080858e28f28`.
- Environnement : `3.11.15`, `numpy/scipy locked`, `darwin/CPU/default threads`.
- Configuration : `a358b986869facccce6e9b4b0d5006c5cf086b333d61f822d7b7998080e0559d` ; graines observées : `11`, `12`.
- Tous les enregistrements sont `classical` et les plans réussis sont marqués faisables.

Statuts conservés :
- `optimal_lp_restricted_ip`: 8

## Profil mesuré

| Catégorie | Exécutions réussies | Médiane runtime (s) | Min--max runtime (s) | Médiane itérations CG |
|---|---:|---:|---:|---:|
| SMALL | 3 | 0.006309 | 0.006302--0.006690 | 1.0 |
| MEDIUM | 1 | 0.025305 | 0.025305--0.025305 | 1.0 |
| LARGE | 3 | 0.114617 | 0.102383--0.115580 | 2.0 |
| XL | 1 | 0.171033 | 0.171033--0.171033 | 5.0 |

| Composant | Total (s) | Part du temps instrumenté |
|---|---:|---:|
| `column_management_runtime` | 0.000728 | 0.13% |
| `integer_master_runtime` | 0.128506 | 23.44% |
| `master_problem_runtime` | 0.059296 | 10.82% |
| `pricing_runtime` | 0.358096 | 65.32% |
| `unattributed_runtime` | 0.000170 | 0.03% |
| `verification_runtime` | 0.001423 | 0.26% |

Le composant dominant mesuré est **`pricing_runtime`**. Les parts ci-dessus sont calculées sur les composants instrumentés, incluant `unattributed_runtime`; elles ne constituent pas un speedup.

## Figures

- [`classical_runtime_by_size.png`](classical_runtime_by_size.png) montre la médiane du temps mur-à-mur par catégorie `size-class-v1`, avec l'intervalle min--max du corpus disponible.
- [`classical_runtime_components.png`](classical_runtime_components.png) montre les temps cumulés enregistrés par composant.

Les deux figures ont été générées depuis le JSON source par `scripts/plot_phase2_profile.py`. Aucune comparaison Neural CG, aucun speedup et aucune extrapolation de performance ne sont publiés : les données de cette phase ne contiennent que la baseline classique.
