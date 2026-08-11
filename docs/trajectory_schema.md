# Schéma de trajectoire de génération de colonnes

Le contrat `cg-trajectory-v2` décrit une exécution classique complète du Cutting Stock 1D,
depuis les conventions numériques jusqu'à son statut terminal. Il est défini dans
`neural_cutting_stock.benchmarks.trajectory` et sérialisé par `ColumnGenerationTrajectory.to_dict()`.

## Structure

Un document contient :

- `schema_version` : valeur fixe `cg-trajectory-v2` ;
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
- `selected_patterns` contient les motifs retenus à cette itération, `columns_added` et les autres
  compteurs décrivent le progrès observé ; lorsqu'ils sont tous présents,
  `final_column_count = initial_column_count + columns_added`.
- `rmp_runtime_seconds`, `pricing_runtime_seconds` et `column_management_runtime_seconds` sont des
  durées mur-à-mur non négatives, et `exact_fallback` indique explicitement si le garde-fou exact a
  été utilisé. Une décision indisponible reste `null`.

La collecte des états du RMP et des identifiants d'instance est réalisée par P3.02. Les champs de
duales et de pricing sont définis par P3.03 et P3.04 ; P3.05 complète le contrat des colonnes
retenues, du progrès, des durées et du fallback. La construction d'un lecteur et le rejeu sont
réservés à P3.06.

## Surcoût de collecte

`collect_trajectory(result, metadata)` matérialise une trajectoire uniquement à partir des
observations immuables déjà produites par `ColumnGeneration`, puis mesure la construction et la
sérialisation JSON. Il n'est donc pas appelé dans la boucle de pricing et ne peut pas modifier une
décision classique. La mesure expose `collection_runtime_seconds` et `serialized_size_bytes`.
Chaque trajectoire collectée doit ensuite être passée à `replay_trajectory` ; la comparaison des
statuts et des motifs constitue le contrôle de non-altération. Le temps de collecte est rapporté
séparément et ne remplace jamais le temps mur-à-mur du solveur.

## Cible de colonne utile

P3.08 définit une cible contrefactuelle, sans entraîner ni sélectionner de modèle. Pour une même
instance et configuration, deux trajectoires classiques sont comparées : une référence sans la
colonne et une exécution où la colonne est retenue. Le travail mesuré est la somme des durées RMP,
pricing et gestion des colonnes de toutes les itérations. La réduction est :

```text
work_reduction_seconds = work_without_column_seconds - work_with_column_seconds
```

La colonne est `useful = true` seulement si cette réduction est strictement supérieure à la
tolérance déclarée. Les deux trajectoires doivent fournir toutes les durées ; une mesure absente
n'est jamais remplacée par zéro. La cible ne porte donc ni sur le seul temps d'inférence ni sur la
qualité finale, qui reste vérifiée séparément par le solveur classique.

Les partitions train, validation et test sont fixées avant cette collecte selon
[le protocole dédié](partition_protocol.md). Elles séparent simultanément les graines et les
familles de générateurs ; une combinaison qui croise deux partitions est rejetée.
