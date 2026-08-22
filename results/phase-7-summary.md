# Bilan de la Phase 7

Source validée : [`exact-gap.json`](exact-gap.json) (schéma `exact-gap-v1`), régénérable depuis les corpus et campagnes persistés.
Aucune durée n'entre dans ce bilan : la qualité est la métrique reine de la phase et les durées restent journalisées à titre informatif.

## Méthode de référence et tolérances

- Référence : méthode `milp_on_enumerated_patterns`, limites `maximal_patterns:max_search_space_size=10000000,max_patterns=100000`.
- Tolérances : intégralité **1e-09**, faisabilité **1e-09**.
- Contrôle croisé d'énumération : **désactivé**.
- Environnement tracé : commit `5d05240fbbba…`, Python 3.11.15, numpy==2.4.6,scipy==1.17.1.

## Couverture

- Instances avec référence exacte : **16** ; instances exclues sans référence rattachable : **9** (diagnostics conservés dans le rapport source).
- Références : **16** optimales, **0** borne seule, **0** en échec ; vérifications indépendantes en échec : **0**.
- Écarts disponibles : **16**, dont **15** nuls et **1** positifs.
- Exclusion : « instance data was never persisted; the exact reference cannot be bound to this baseline » — 8 instance(s).
- Exclusion : « no persisted classical baseline run » — 1 instance(s).

## Marge par classe de taille

| Classe | Instances | Types | Écarts disponibles | Marge nulle | Marge positive | Écart médian maximal (barres) |
|---|---:|---|---:|---:|---:|---:|
| SMALL | 7 | 2×5, 3×2 | 7 | 7 | 0 | 0 |
| MEDIUM | 3 | 4×3 | 3 | 3 | 0 | 0 |
| LARGE | 3 | 6×3 | 3 | 2 | 1 | 1 |
| XL | 3 | 8×3 | 3 | 3 | 0 | 0 |

## Instances à marge positive

| Instance | Classe | Famille | Types | Optimum entier | Baseline médiane | Écart médian (barres) | Écarts par répétition |
|---|---|---|---:|---:|---:|---:|---|
| `d71a500910a2…` | LARGE | `ecdaf63fdab4…` | 6 | 19 | 20 | 1 | 1.0;1.0;1.0 |

## Lecture et suite

Sur les 16 écarts disponibles, 15 sont nuls et 1 sont positifs ; l'écart médian maximal observé vaut 1 barre(s), sur un objectif de découpe. Les leviers du générateur susceptibles de créer des trous entiers non triviaux sont documentés, sans activation ni mesure nouvelle, dans [`docs/phase-7-gap-levers.md`](../docs/phase-7-gap-levers.md) ; leur vérification expérimentale relève des cases P8.01, P8.03 et P8.04.

## Garanties et limites

Les références proviennent d'un MILP sur motifs énumérés, vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum, contrôle croisé d'énumération sur sous-échantillon lorsque activé). Les objectifs classiques restent des optimaux sur colonnes générées uniquement (`optimal_over_generated_columns_only`) et ne préjugent pas d'un optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les exclusions complètes et leurs raisons figurent dans [`exact-gap.json`](exact-gap.json).

## Commandes de reproduction

Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :

```bash
uv sync --extra dev
uv run python scripts/report_phase7_exact_gap.py
uv run python scripts/report_phase7_exact_gap_breakdown.py
uv run python scripts/report_phase7_summary.py
```
