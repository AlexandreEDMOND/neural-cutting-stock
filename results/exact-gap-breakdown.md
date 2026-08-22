# Bilan chiffré des écarts à la référence exacte (Phase 7)

Source validée : [`exact-gap.json`](exact-gap.json) (schéma `exact-gap-v1`).
Chaque écart compare la baseline classique (`optimal_over_generated_columns_only`) à l'optimum entier certifié de sa référence exacte vérifiée ; aucune durée n'entre dans ce bilan.

## Couverture

- Instances avec référence : **16** ; instances exclues sans baseline persistée : **9** (diagnostics conservés dans le rapport source).
- Références : **16** optimales, **0** borne seule, **0** en échec ; vérifications indépendantes en échec : **0**.
- Écarts disponibles : **16**, dont **15** nuls et **1** positifs.

## Marge par classe de taille

| Classe | Instances | Types | Écarts disponibles | Marge nulle | Marge positive | Écart médian maximal (barres) |
|---|---:|---|---:|---:|---:|---:|
| SMALL | 7 | 2×5, 3×2 | 7 | 7 | 0 | 0 |
| MEDIUM | 3 | 4×3 | 3 | 3 | 0 | 0 |
| LARGE | 3 | 6×3 | 3 | 2 | 1 | 1 |
| XL | 3 | 8×3 | 3 | 3 | 0 | 0 |

## Marge par famille

| Famille | Types | Instances | Écarts disponibles | Marge nulle | Marge positive | Écart médian maximal (barres) |
|---|---|---:|---:|---:|---:|---:|
| `0f4bb00c8bfe…` | 4×3 | 3 | 3 | 3 | 0 | 0 |
| `268daf05c1f2…` | 8×3 | 3 | 3 | 3 | 0 | 0 |
| `36719ab4e0c9…` | 3×2 | 2 | 2 | 2 | 0 | 0 |
| `96bc46475a90…` | 2×5 | 5 | 5 | 5 | 0 | 0 |
| `ecdaf63fdab4…` | 6×3 | 3 | 3 | 2 | 1 | 1 |

## Instances à marge positive

| Instance | Classe | Famille | Types | Optimum entier | Baseline médiane | Écart médian (barres) | Écarts par répétition |
|---|---|---|---:|---:|---:|---:|---|
| `d71a500910a2…` | LARGE | `ecdaf63fdab4…` | 6 | 19 | 20 | 1 | 1.0;1.0;1.0 |

## Garanties et limites

Les références proviennent d'un MILP sur motifs énumérés, vérifiées indépendamment (faisabilité du plan, borne LP ≤ optimum, contrôle croisé d'énumération sur sous-échantillon). Les objectifs classiques restent des optimaux sur colonnes générées uniquement et ne préjugent pas d'un optimum entier global. Aucun écart indisponible n'est filtré silencieusement ; les exclusions complètes et leurs raisons figurent dans [`exact-gap.json`](exact-gap.json). Le bilan se régénère depuis les données persistées avec `uv run python scripts/report_phase7_exact_gap_breakdown.py`.
