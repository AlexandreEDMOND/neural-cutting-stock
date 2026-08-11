# Décision P5.03 : optimisation séquentielle

## Décision

Ne pas entreprendre d'optimisation séquentielle à ce stade. La politique supervisée
bornée de la Phase 4 reste le candidat retenu, avec le pricing exact et le fallback
exact comme garde-fous. Cette décision ne constitue pas une conclusion de performance
généralisable : les mesures disponibles sont trop peu couvertes pour établir un gain
sur les grandes instances.

## Mesures utilisées

La source brute est [`results/phase-4-benchmark-runs.csv`](../results/phase-4-benchmark-runs.csv),
au schéma `benchmark-run-v1`. Elle contient quatre paires exécutées sur les mêmes
instances et ressources, toutes convergées, faisables et avec
`objective_difference_vs_classical = 0`.

| Catégorie | Paires admissibles | Médiane Classical (s) | Médiane Neural (s) | Médiane speedup |
|---|---:|---:|---:|---:|
| SMALL | 3 | 0.006379 | 0.007853 | 1.029572 |
| MEDIUM | 1 | 0.006240 | 0.018792 | 0.332036 |
| LARGE | 0 | n/a | n/a | n/a |
| XL | 0 | n/a | n/a | n/a |

Le speedup est le runtime classique apparié divisé par le runtime neural. Les
valeurs dérivées ont été recalculées depuis les enregistrements bruts, avant toute
agrégation. Le neural est donc plus lent sur l'unique paire `MEDIUM`, et aucune
mesure `LARGE` ou `XL` ne permet de vérifier l'hypothèse sur les régimes difficiles.

La CSV publiée conserve `feature_preparation_runtime` pour les exécutions neurales,
mais `neural_inference_runtime` y est nul. Ce champ manquant interdit d'attribuer
quantitativement le surcoût à l'inférence seule ; il ne change pas la comparaison
end-to-end, qui inclut le temps total mesuré. Aucun speedup ne repose donc sur le
seul temps d'inférence.

## Justification et portée

Une politique séquentielle plus complexe ajouterait de l'état, des décisions et du
coût d'exécution sans bénéfice end-to-end établi par ces mesures. Elle risquerait
également de compliquer le contrôle exact sans résoudre la couverture insuffisante
des grandes instances. La décision est donc de ne pas la concevoir ni l'implémenter
dans ce cycle. Toute révision devra s'appuyer sur de nouvelles paires appariées,
inclure tous les coûts récurrents, conserver les échecs et vérifier la qualité avant
d'agréger les runtimes.
