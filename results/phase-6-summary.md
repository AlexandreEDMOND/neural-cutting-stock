# Bilan final de la Phase 6

Sources brutes validées : `results/phase-6-classical-runs.csv` et `results/phase-6-neural-runs.csv` (schéma `benchmark-run-v1`).
La conclusion scientifique complète est publiée dans [`docs/conclusion.md`](../docs/conclusion.md).

## Protocole gelé

- Configuration : `configs/phase-6-final.json` (`phase-6-final-freeze-v1`), comparaison appariée par `paired_instance_id`, 3 répétitions par instance et par mode, ordre d'exécution `classical_then_neural`, modèle `preloaded_per_process`.
- Tolérances : différence d'objectif **0 barre(s)**, coût réduit **1e-09**.
- Budgets : aucune limite de temps ; aucune limite d'itérations.
- Instances non vues : **12** hors corpus Phase 3 (SMALL 3, MEDIUM 3, LARGE 3, XL 3).
- Modèle évalué : [`models/linear-scorer-v1-zero-weight.json`](../models/linear-scorer-v1-zero-weight.json) (`linear-scorer-v1-zero-weight`, politique `bounded-column-selection-v1`).

## Couverture et qualité

- Exécutions : **72** ; paires : **36** ; paires admissibles : **36**.
- Violations de qualité : **0** paire(s) à la tolérance déclarée.
- Statuts terminaux : `optimal_lp_restricted_ip` : 72.
- Les différences d'objectif et les speedups sont recalculés depuis les enregistrements bruts ; chaque paire, y compris échec ou violation, reste conservée dans [`phase-6-paired-tables.json`](phase-6-paired-tables.json).

## Runtime mur-à-mur par strate cible

| Strate cible | Instances | Paires admissibles | Classical médian (s) | Neural médian (s) | Speedup médian |
|---|---:|---:|---:|---:|---:|
| SMALL | 3 | 9 | 0.017027 | 0.019093 | 0.915631 |
| MEDIUM | 3 | 9 | 0.127440 | 0.102831 | 1.203069 |
| LARGE | 3 | 9 | 0.108184 | 0.125215 | 0.797016 |
| XL | 3 | 9 | 0.214556 | 0.660288 | 0.201712 |

- Speedup médian par instance : de **0.039832** à **1.988822** ; Neural CG est plus lent sur 9 des 12 instances et plus rapide sur 3.
- Les strates reprennent le `target_size_class` figé du manifeste final, jamais le `size_class` mesuré par enregistrement.

## Rapports détaillés

- Tableaux appariés : [`phase-6-paired-tables.md`](phase-6-paired-tables.md) et [`phase-6-paired-tables.json`](phase-6-paired-tables.json).
- Échecs, timeouts et violations : [`phase-6-failures.json`](phase-6-failures.json).
- Généralisation au-delà des tailles d'entraînement : [`phase-6-generalization.json`](phase-6-generalization.json).
- Incertitude des répétitions : [`phase-6-uncertainty.json`](phase-6-uncertainty.json).
- Figures issues des paires admissibles : [`runtime_comparison.png`](runtime_comparison.png) et [`speedup_by_size.png`](speedup_by_size.png).

## Garanties et limites

Le pricing exact certifie l'optimalité de la relaxation linéaire du maître complet à la tolérance déclarée lorsqu'aucune colonne améliorante n'est trouvée. Le maître entier final est résolu sur les colonnes générées uniquement : ses objectifs restent qualifiés `optimal_over_generated_columns_only`, sans preuve d'optimalité entière globale. Chaque plan est vérifié indépendamment pour la demande, la capacité, le kerf et l'objectif. Aucune exécution n'est filtrée des sources ; les campagnes proviennent d'un environnement matériel unique tracé dans leurs métadonnées, donc seules les comparaisons appariées intra-campagne sont interprétables.

## Manifeste final et artefacts

- Manifeste : `data/phase-6-final/manifest.json` (`phase-6-instance-manifest-v1`), `manifest_id` `cc139868f3500f38b74dc0c3db41dc93582593a08bc4ff90dafa102f30e4d1f7`, SHA-256 `e99a8ddaac12c20794d4d63ef28227beb5211bbc301736dae6d88a4a5768ab6c`.
- Artefact modèle : `models/linear-scorer-v1-zero-weight.json`, SHA-256 `96c79e92ad6d488e3371304c3f94f4e1e28222e181859dc343ca8b3aead0eaff`.

## Commandes de reproduction

Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :

```bash
uv sync --extra dev
uv run python scripts/generate_phase6_manifest.py
uv run python scripts/run_phase6_classical.py
uv run python scripts/run_phase6_neural.py
uv run python scripts/report_phase6_uncertainty.py
uv run python scripts/report_phase6_failures.py
uv run python scripts/report_phase6_generalization.py
uv run python scripts/report_phase6_paired_tables.py
uv run python scripts/plot_phase6_results.py
uv run python scripts/report_phase6_summary.py
```
