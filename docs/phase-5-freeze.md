# Gel du candidat de Phase 5

Le gel est une décision calculée par
`freeze_candidate_on_validation` depuis les enregistrements bruts appariés de
validation. Il est autorisé uniquement si toutes les paires ont un plan
faisable, un statut convergé et une différence d'objectif dans la tolérance
déclarée, puis si :

```text
sum(candidate.total_runtime_seconds)
    < sum(classical.total_runtime_seconds)
```

La comparaison porte donc sur le temps mur-à-mur total de la validation, et
non sur le seul temps d'inférence ni sur une moyenne de speedups individuels.
Un échec, une mesure manquante, une violation de qualité ou une égalité de
runtime empêche le gel et fournit une raison diagnostique dans la décision.
Les enregistrements bruts restent la source de vérité ; cette décision ne
modifie ni le pricing exact ni les garanties du solveur.
