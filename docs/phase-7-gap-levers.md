# Leviers de trous entiers dans le générateur déterministe (Phase 7)

Ce document répond à la condition de la case P7.07 : la marge de qualité mesurée en Phase 7 étant
quasi nulle partout, il identifie — sans les activer — les paramètres du générateur déterministe
[`SyntheticInstanceGenerator`](../src/neural_cutting_stock/benchmarks/generator.py) qui contrôlent
l'apparition de trous entiers non triviaux, c'est-à-dire un écart positif entre la baseline
classique (`optimal_over_generated_columns_only`) et l'optimum entier certifié du maître complet.

## Condition vérifiée : marge quasi nulle partout

D'après [`results/exact-gap-breakdown.json`](../results/exact-gap-breakdown.json)
(schéma `exact-gap-breakdown-v1`, produit par `scripts/report_phase7_exact_gap_breakdown.py`) :

- 16 écarts disponibles sur 25 instances considérées (9 exclusions conservées) ;
- **15 écarts nuls, 1 écart positif** (1 barre) ;
- par classe de taille : SMALL 7/7 nuls, MEDIUM 3/3 nuls, LARGE 2/3 nuls et 1 positif, XL 3/3 nuls.

La condition « marge quasi nulle partout » est donc satisfaite : l'identification des leviers
s'impose avant la Phase 8.

## Mécanisme du trou entier

La boucle classique ajoute des colonnes tant que le pricing exact trouve un coût réduit
strictement négatif. À convergence, la relaxation du maître restreint atteint celle du maître
complet, mais tout motif à coût réduit positif ou nul absent du pool n'est jamais généré. Le maître
entier final optimise alors sur ce sous-ensemble : son optimum entier peut dépasser celui du
maître complet résolu sur tous les motifs maximaux, qui sert de référence exacte. Un trou apparaît
lorsque la combinaison entière optimale requiert au moins un motif jamais généré parce qu'il était
« marginal » au sens LP. Les paramètres du générateur augmentent cette probabilité lorsqu'ils
rendent ces motifs optimaux entiers rares et fins : remplissages exacts, paires exclusives, restes
de division.

## Ce que montre l'unique instance à marge positive

L'instance `d71a500910a2…` ([`results/exact-gap.csv`](../results/exact-gap.csv),
manifeste [`data/phase-6-final/manifest.json`](../data/phase-6-final/manifest.json), graine 303,
famille LARGE `ecdaf63fdab4…`) porte sur ses données réelles toutes les structures visées :

- longueurs `[14, 21, 34, 40, 43, 82]`, demandes `[10, 3, 7, 9, 8, 7]`, barre 100, kerf 0 ;
- multiplicités seules : 7, 4, 2, 2, 2, 1 pièces par barre ; les motifs homogènes couvrent la
  demande avec 23 barres contre un optimum entier certifié de 19 : la complémentarité entre types
  vaut 4 barres, et la baseline restreinte n'en capture que 3 ;
- remplissage exact `43 + 43 + 14 = 100` et paire exclusive `82 + 14 = 96` (aucune autre longueur
  ne se combine avec 82, car `82 + 21 = 103 > 100`) ;
- demande 10 non divisible par la multiplicité 7 de sa longueur : deux barres homogènes offrent
  14 emplacements pour 10 pièces, et absorber autrement ce surplus exige des motifs mixtes.

Fait notable pour la suite : la borne LP de cette instance est entière (`lp_bound_bars = 19.0`,
égale à l'optimum entier). Le trou vient donc du pool restreint au moment du maître entier, pas
d'une borne LP fractionnaire ; a contrario, plusieurs instances à borne LP fractionnaire
(`29.5`, `14.5` dans [`results/exact-gap.csv`](../results/exact-gap.csv)) présentent un écart nul.
Une borne LP fractionnaire ne suffit ni n'est nécessaire : c'est la présence des motifs
nécessaires dans le pool généré qui décide.

## Leviers identifiés dans le générateur déterministe

Aucun levier ci-dessous n'est activé ni mesuré dans ce cycle ; chaque ligne indique la case de
Phase 8 qui pourra l'activer et le vérifier expérimentalement.

| Paramètre | Valeurs actuelles | Effet observé ou attendu | Levier vers des trous non triviaux |
|---|---|---|---|
| `piece_length_range` | `(10, 90)` sur une barre de 100 | Ratios détendus : chaque type admet des multiplicités allant jusqu'à 10 (longueur 10) et de nombreuses complémentarités ; les motifs mixtes utiles à l'arrondi entrent dans le pool, avec 15 écarts nuls sur 16 | Fenêtre étroite calée juste au-dessus de `stock_length/(k+1)` (ratios tendus) : multiplicités bornées, remplissages exacts et paires exclusives plus fréquents |
| `demand_range` | `(1, 10)` | Totaux faibles ; les restes de division sont petits et absorbés par les motifs déjà générés | Demandes peu divisibles par les multiplicités naturelles et bornes hautes plus grandes : surplus structurels exigeant des motifs mixtes marginaux |
| `length_distribution` / `demand_distribution` | `"uniform_integer_v1"` | Champs persistés et hachés dans `family_id`, mais inertes : `generate()` échantillonne toujours uniformément (`rng.sample`, `rng.randint`) | Point d'activation naturel des profils structurés de P8.03 (longueurs groupées, demandes corrélées aux multiplicités) sans toucher aux schémas |
| `number_of_types` | 2 à 8, avec plage de longueurs large | Densité de types faible devant la plage : duales bien séparées, peu d'égalités de coût réduit | Densifier les types dans une plage étroite : égalités de coût réduit fréquentes, pool restreint plus sensible au choix |
| `stock_length` | `100.0` partout | Divisibilités régulières avec des longueurs entières uniformes | Valeurs non rondes (par exemple 97) cassant les alignements de division |
| `kerf` | `0.0` dans toutes les campagnes | Aucun déplacement des frontières de capacité | Kerf strictement positif exercé (P8.01) : la capacité effective diminue selon la cardinalité du motif ; deux pièces de longueur 50 exigent `100 + 2×kerf` et ne tiennent plus ensemble dès `kerf > 0`, ce qui fait chuter la multiplicité de 2 à 1 |

La graine n'est pas un levier structurel : sur les 16 écarts disponibles répartis en cinq familles,
elle ne produit qu'une seule marge positive ; c'est bien la configuration des familles qui
commande la marge.

## Portée et limites

- Ce document n'introduit aucun code, aucune donnée ni aucune mesure nouvelle : tous les chiffres
  cités proviennent des artefacts persistés listés ci-dessus.
- Les effets attendus des leviers restent des hypothèses qualitatives jusqu'à leur activation ;
  leur vérification expérimentale (« vérifier qu'ils produisent des trous entiers non triviaux »)
  relève des cases P8.01, P8.03 et P8.04.
- Le bilan chiffré source se régénère depuis les données validées avec
  `uv run python scripts/report_phase7_exact_gap_breakdown.py`.
