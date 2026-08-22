# Résultats

Les publications des Phases 1 à 5 sont produites à partir de données persistées et validées :
[`phase-1-summary.md`](phase-1-summary.md), [`phase-2-summary.md`](phase-2-summary.md),
[`phase-3-summary.md`](phase-3-summary.md), [`phase-4-summary.md`](phase-4-summary.md) et
[`phase-5-summary.md`](phase-5-summary.md).

## Évaluation finale (Phase 6)

Le bilan final est [`phase-6-summary.md`](phase-6-summary.md). Il publie le manifeste des instances
non vues [`../data/phase-6-final/manifest.json`](../data/phase-6-final/manifest.json), les tableaux
appariés ([`phase-6-paired-tables.md`](phase-6-paired-tables.md)), les rapports d'échecs, de
généralisation et d'incertitude, les commandes de reproduction ainsi que les figures finales :

- [`runtime_comparison.png`](runtime_comparison.png) — temps mur-à-mur médian des paires admissibles ;
- [`speedup_by_size.png`](speedup_by_size.png) — speedup médian par strate cible gelée.

Ces deux figures sont générées uniquement depuis les exécutions brutes validées selon
[le protocole de benchmark](../docs/benchmark_protocol.md). Aucun graphique ou nombre factice ne
doit être ajouté. La conclusion scientifique de la Phase 6 est publiée dans
[`../docs/conclusion.md`](../docs/conclusion.md).

## Référence exacte et écarts (Phase 7)

Le bilan de clôture de la phase est [`phase-7-summary.md`](phase-7-summary.md). L'écart de la
baseline classique (`optimal_over_generated_columns_only`) aux références exactes vérifiées est
persisté au schéma `exact-gap-v1` dans [`exact-gap.json`](exact-gap.json) et
[`exact-gap.csv`](exact-gap.csv), avec les exclusions et diagnostics conservés. Le rapport ne
contient aucune durée ; il se régénère depuis les corpus et campagnes persistés avec :

```bash
uv run python scripts/report_phase7_exact_gap.py
```

Le bilan chiffré des marges par famille et par classe de taille est publié dans
[`exact-gap-breakdown.md`](exact-gap-breakdown.md) et [`exact-gap-breakdown.json`](exact-gap-breakdown.json),
agrégé uniquement depuis le rapport `exact-gap-v1` persisté :

```bash
uv run python scripts/report_phase7_exact_gap_breakdown.py
```

Le bilan de phase se régénère depuis le même rapport persisté :

```bash
uv run python scripts/report_phase7_summary.py
```
