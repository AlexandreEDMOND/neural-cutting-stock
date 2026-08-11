# Schéma de trajectoire de génération de colonnes

Le contrat `cg-trajectory-v1` décrit une exécution classique complète du Cutting Stock 1D,
depuis les conventions numériques jusqu'à son statut terminal. Il est défini dans
`neural_cutting_stock.benchmarks.trajectory` et sérialisé par `ColumnGenerationTrajectory.to_dict()`.

## Structure

Un document contient :

- `schema_version` : valeur fixe `cg-trajectory-v1` ;
- `metadata` : `trajectory_id`, `instance_id`, version du solveur, graine, configuration,
  environnement, données de l'instance et tolérances ;
- `iterations` : liste ordonnée et contiguë d'observations indexées à partir de 1 ; chaque
  observation peut porter l'identifiant d'instance, les valeurs des colonnes du RMP et le nombre
  de motifs présents dans le RMP ;
- `status` : `converged`, `resource_limit` ou `failed` ;
- `termination_reason` et `error_message` pour le diagnostic terminal.

Les champs de `TrajectoryIteration` couvrent l'état observable du RMP, les duales, le pricing,
les motifs candidats et retenus, le fallback exact et les durées. Une mesure indisponible est
`null`, jamais une valeur par défaut. Les listes de motifs sont des tuples d'entiers dans l'ordre
des types de l'instance.

## Invariants

- `piece_lengths` et `demands` ont le même ordre et la même longueur non nulle ;
- les tolérances de coût réduit, intégralité et faisabilité sont explicites et non négatives ;
- les itérations sont numérotées `1..n` sans trou ;
- une trajectoire non convergée porte un diagnostic non vide ;
- `to_dict()` conserve la version, les statuts textuels et un ordre déterministe des champs.
- chaque résolution du RMP produit un `RMPState` dans le résultat de `ColumnGeneration`, avec son
  index, ses motifs, son résultat et sa durée ; le runner transmet l'`instance_id` stable à chaque
  état.

La collecte des états du RMP et des identifiants d'instance est réalisée par P3.02. Les champs
relatifs aux duales, candidats, colonnes retenues et fallbacks seront remplis dans les étapes
suivantes.
