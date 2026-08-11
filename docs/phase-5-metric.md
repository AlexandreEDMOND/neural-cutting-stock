# Métrique P5.05 : temps total et qualité

## Contrat

La métrique versionnée est `quality-gated-speedup-v1`. Elle s'applique à une
paire `Classical CG`/`Neural CG` identifiée par le même `instance_id` et la même
répétition.

```text
speedup_vs_classical = classical_total_runtime / neural_total_runtime
score = speedup_vs_classical, si quality_preserved et comparable
score = 0, sinon
```

`total_runtime_seconds` est le temps mur-à-mur de l'entrée dans `solve` jusqu'au
statut final et à la vérification indépendante du plan. Le score n'utilise donc
ni le seul temps d'inférence ni une décomposition de runtime.

## Garde de qualité

`quality_preserved` est vrai uniquement lorsque les deux exécutions ont le statut
`optimal_lp_restricted_ip`, des plans faisables et une différence d'objectif dans
la tolérance déclarée. La différence est recalculée avant le score :

```text
objective_difference_vs_classical = neural_objective - classical_objective
```

Une paire hors tolérance, invalide, en échec ou sans temps positif conserve ses
diagnostics dans les résultats bruts, mais ne peut pas obtenir un score positif.
Cette porte dure empêche de récompenser une réduction de temps obtenue par une
dégradation de qualité.

## Usage

`quality_gated_speedup` produit le score et les diagnostics de la paire. Les
agrégations de campagne doivent d'abord être faites au niveau des paires
admissibles, puis résumées par instance et catégorie de difficulté. Les scores
ne sont pas une nouvelle fonction de perte du modèle : aucune optimisation
supplémentaire n'est introduite par P5.05.
