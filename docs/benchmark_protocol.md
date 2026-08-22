# Protocole de benchmark

## 1. Principe

Le benchmark doit répondre à une question end-to-end : à qualité comparable, Neural CG termine-t-il plus vite que Classical CG sur les mêmes instances ?

Chaque comparaison est appariée par `instance_id`. Les deux modes partagent la représentation du problème, le RMP, le pricing exact, le maître entier, le vérificateur et les limites de ressources. Seule la politique de proposition/sélection des colonnes varie.

Les catégories de difficulté sont versionnées par `size-class-v1`. Elles utilisent le temps
mur-à-mur classique mesuré, et non la demande seule : `SMALL` est strictement inférieur à
`0.015997 s`, `MEDIUM` à partir de cette valeur et strictement inférieur à `0.06385 s`, `LARGE`
à partir de cette valeur et strictement inférieur à `0.1433 s`, puis `XL`. Ces seuils sont les
milieux des trois plus grands écarts observés dans `results/phase-2-baseline-profile.json`.
Une exécution sans temps total (échec ou timeout avant mesure) conserve `size_class = null`.

## 2. Reproductibilité de l’instance

Chaque instance synthétique doit pouvoir être régénérée à partir de :

- `generator_name` et `generator_version` ;
- `seed` explicite ;
- longueur de stock et kerf ;
- nombre de types ;
- paramètres de distribution des longueurs ;
- paramètres de distribution des demandes ;
- règles de normalisation et d’arrondi.

`instance_id` est stable et dérivé d’une sérialisation canonique des données normalisées, ou vérifié par un hash associé. Les listes exactes de longueurs et demandes sont conservées dans le manifeste d’instances, pas réinventées depuis un libellé de taille.

## 3. Schéma minimal d’un enregistrement d’exécution

Les noms suivants sont le contrat commun des futurs fichiers tabulaires. Les champs additionnels sont autorisés, mais un changement de sens impose une nouvelle `schema_version`.

### Identité et environnement

| Champ | Type | Définition |
|---|---:|---|
| `schema_version` | chaîne | Version du schéma de résultat. |
| `run_id` | chaîne | Identifiant unique de l’exécution. |
| `instance_id` | chaîne | Identifiant stable de l’instance. |
| `solver_mode` | enum | `classical` ou `neural`. |
| `solver_version` | chaîne | Version/configuration de l’algorithme. |
| `seed` | entier | Graine de génération ; graines d’exécution séparées si nécessaire. |
| `size_class` | chaîne nullable | Catégorie `size-class-v1` issue du temps total classique mesuré. |
| `config_id` | chaîne | Hash ou identifiant de la configuration complète. |
| `code_commit` | chaîne | Commit Git mesuré. |
| `python_version` | chaîne | Version Python. |
| `dependency_versions` | chaîne/JSON | Versions NumPy, SciPy/HiGHS et, le cas échéant, PyTorch. |
| `hardware_id` | chaîne | CPU, mémoire, OS et politique de threads documentés. |
| `repetition` | entier | Index de répétition pour l’instance et le mode. |

### Caractéristiques de l’instance

| Champ | Type | Définition |
|---|---:|---|
| `stock_length` | réel | Longueur d’une barre. |
| `kerf` | réel | Trait de scie par pièce selon la convention v1. |
| `number_of_piece_types` | entier | Nombre de longueurs distinctes normalisées. |
| `total_demand` | entier | Somme des demandes. |
| `requested_length` | réel | `sum(length_i * demand_i)`. |
| `length_distribution` | chaîne | Famille et paramètres versionnés. |
| `demand_distribution` | chaîne | Famille et paramètres versionnés. |

### Qualité et matière

| Champ | Type | Définition |
|---|---:|---|
| `objective_value` | réel | Objectif du maître entier final ; égal au nombre de barres si succès. |
| `number_of_stock_bars` | entier | Nombre de barres du plan vérifié. |
| `lp_objective_value` | réel | Borne LP à convergence. |
| `restricted_integer_gap` | réel | Écart relatif ou absolu, avec convention déclarée, entre entier restreint et LP. |
| `total_waste` | réel | `bars * stock_length - requested_length`. |
| `trim_loss` | réel | Capacité résiduelle des motifs utilisés. |
| `kerf_loss` | réel | Matière consommée par les traits de scie. |
| `overproduction_length` | réel | Longueur produite au-delà de la demande. |
| `plan_feasible` | booléen | Résultat du vérificateur indépendant. |

### Travail de Column Generation

| Champ | Type | Définition |
|---|---:|---|
| `number_of_cg_iterations` | entier | Nombre de résolutions LP du RMP dans la boucle, convention incluse dans la version solveur. |
| `number_of_generated_columns` | entier | Colonnes améliorantes produites avant déduplication/filtrage. |
| `number_of_columns_added` | entier | Colonnes distinctes effectivement ajoutées au RMP. |
| `initial_column_count` | entier | Colonnes présentes avant la première résolution. |
| `final_column_count` | entier | Colonnes du maître entier final. |
| `duplicate_column_count` | entier | Colonnes proposées mais déjà présentes. |
| `final_reduced_cost` | réel | Meilleur coût réduit du dernier pricing exact. |

### Temps

Tous les temps sont exprimés en secondes et mesurés par une horloge monotone (`perf_counter`).

| Champ | Type | Définition |
|---|---:|---|
| `total_runtime_seconds` | réel | De l’entrée dans `solve` jusqu’au plan/statut final, vérification incluse. |
| `master_problem_runtime` | réel | Somme des résolutions LP du RMP. |
| `pricing_runtime` | réel | Somme de tous les appels de pricing, y compris fallback exact. |
| `integer_master_runtime` | réel | Résolution du maître entier restreint. |
| `column_management_runtime` | réel | Construction, filtrage et insertion des colonnes hors solveur. |
| `verification_runtime` | réel | Vérification finale indépendante. |
| `unattributed_runtime` | réel | Total moins composants instrumentés ; conservé pour détecter un oubli. |
| `peak_memory_bytes` | entier | Pic d’allocations Python tracé pendant l’exécution du solveur avec `tracemalloc`. |
| `exact_pricing_calls` | entier | Nombre d’appels au pricing exact, contrôle final compris. |

Le chargement et la validation de l’instance sont exclus si les deux modes reçoivent le même objet déjà construit. Toute préparation spécifique au mode neural, y compris features et transfert de tenseurs, est incluse. La politique de chargement du modèle (à froid ou préchargé) doit être déclarée ; le graphique principal utilise une politique identique pour toutes les tailles et n’omet aucun coût récurrent.

### Statuts

| Champ | Type | Définition |
|---|---:|---|
| `run_status` | enum | `optimal_lp_restricted_ip`, `timeout`, `infeasible`, `solver_error`, `invalid_plan` ou statut versionné. |
| `master_status` | chaîne | Dernier statut du solveur LP. |
| `pricing_status` | chaîne | Dernier statut du pricing exact. |
| `integer_master_status` | chaîne | Statut du maître entier. |
| `termination_reason` | chaîne | Cause structurée de fin de boucle. |
| `error_message` | chaîne nullable | Diagnostic non vide en cas d’échec. |

### Champs Neural CG

Ils sont nuls pour le mode classique, et obligatoires pour le mode neural.

| Champ | Type | Définition |
|---|---:|---|
| `model_id` | chaîne | Checkpoint et configuration du modèle. |
| `neural_inference_runtime` | réel | Appel de la politique, construction des features comprise ; reste incluse dans le total. |
| `feature_preparation_runtime` | réel | Construction du pool de candidats et de l'état présenté à la politique. |
| `number_of_candidates` | entier | Nombre total de motifs présentés à la politique. |
| `number_of_selected_columns` | entier | Nombre total retenu par la politique. |
| `exact_fallback_calls` | entier | Nombre d’appels exacts de sauvegarde/contrôle. |
| `speedup_vs_classical` | réel | Médiane classique appariée divisée par le runtime neural selon la règle d’agrégation. |
| `objective_difference_vs_classical` | réel | Objectif neural moins objectif classique pour la paire. |

## 4. Mesure expérimentale

1. Geler le manifeste d’instances, la configuration et le commit.
2. Fixer les threads des solveurs et bibliothèques ; enregistrer la machine et éviter les charges concurrentes connues.
3. Définir avant le run le nombre de répétitions, le warm-up éventuel, les limites de temps et la politique de modèle.
4. Alterner ou randomiser de façon déterministe l’ordre Classical/Neural afin de limiter les biais thermiques et temporels.
5. Exécuter les deux modes dans des processus isolés si des caches ou fuites mémoire peuvent affecter la comparaison.
6. Conserver chaque tentative, y compris timeout et erreur.
7. Valider le plan, le bilan matière, les statuts et la décomposition temporelle avant agrégation.

Les campagnes classiques acceptent deux limites coopératives, incluses dans `config_id` :
`max_runtime_seconds` borne le temps mur-à-mur de la génération de colonnes et
`max_cg_iterations` borne le nombre de résolutions du RMP. Une limite atteinte produit le statut
solver `limit_reached`, le statut de campagne `timeout` et le motif `resource_limit`; la cellule
reste persistée avec ses mesures partielles et n'est jamais traitée comme une convergence.

L’ordre des générateurs dans une matrice ne change ni son `config_id` ni les `run_id`. Les fichiers
bruts sont écrits dans l’ordre canonique des `run_id`, afin qu’une permutation de l’ordre d’exécution
ne modifie pas leur sérialisation.

Le nombre de répétitions est fixé dans la configuration finale après observation de la variance. Le
rapport `paired-uncertainty-v1` agrège d'abord chaque instance et publie la moyenne, la médiane,
l'écart-type d'échantillon et un intervalle de confiance normal à 95 % lorsque deux répétitions ou
plus sont admissibles. Les échecs restent comptés et exclus des statistiques admissibles.

## 5. Contrôle de qualité avant comparaison

Pour chaque `instance_id` :

```text
objective_difference_vs_classical = neural_objective - classical_objective
speedup_vs_classical = classical_runtime / neural_runtime
```

Le seuil principal est initialement `objective_difference_vs_classical == 0` barre. Toute tolérance différente devra être annoncée avant l’évaluation et les résultats à objectif dégradé seront montrés séparément.

Une exécution invalide, non appariée ou dont le plan est infaisable n’entre pas silencieusement dans une courbe de speedup. Elle reste dans les données et apparaît dans un tableau de couverture/échecs.

## 6. Génération des figures finales

La Phase 2 introduira un script versionné, prévu sous la forme :

```bash
python scripts/plot_runtime_comparison.py \
  --runs results/benchmark_runs.csv \
  --instances results/instances.csv \
  --output-dir results
```

Le script devra :

1. vérifier `schema_version`, unicité des `run_id`, appariement et statuts ;
2. recalculer les différences d’objectif et speedups au lieu de faire confiance à une colonne dérivée ;
3. refuser de présenter comme qualité préservée une paire hors tolérance ;
4. agréger d’abord les répétitions au niveau instance, puis les instances au niveau difficulté ;
5. produire `results/runtime_comparison.png` avec le temps mur-à-mur Classical/Neural ;
6. produire `results/speedup_by_size.png` avec une ligne de référence à `1x` ;
7. écrire à côté un résumé tabulaire des effectifs, échecs et différences d’objectif ;
8. intégrer dans les métadonnées ou un manifeste la source, le commit et les paramètres de génération.

L’axe des abscisses utilisera les catégories de difficulté versionnées, ordonnées de `SMALL` à `XL`, et pourra être complété par des graphiques structurels (par exemple nombre de types). Les seuils seront déterminés après profilage classique et gelés avant l’évaluation neuronale finale.

Aucun fichier PNG factice n’est créé pendant la fondation du dépôt.

## 7. Référence exacte de qualité `exact-reference-v1`

La vérité terrain de la qualité est persistée par instance sous le schéma versionné
`exact-reference-v1`. Une référence est produite par une méthode exacte déclarée et porte
l’optimum entier du maître complet lorsque celui-ci est prouvé, sinon la borne inférieure
certifiée associée.

### Champs

| Champ | Type | Définition |
|---|---:|---|
| `schema_version` | chaîne | `exact-reference-v1`. |
| `instance_id` | chaîne | Identifiant stable de l’instance référencée. |
| `reference_method` | enum | Méthode exacte productrice : `exhaustive_pattern_enumeration` ou `milp_on_enumerated_patterns`. |
| `status` | enum | `optimal` (optimum entier prouvé), `lower_bound_only` (borne seule certifiée) ou `failed`. |
| `integer_optimum_bars` | entier nullable | Nombre de barres de l’optimum entier prouvé ; requis et positif uniquement pour `optimal`. |
| `certified_lower_bound_bars` | réel nullable | Borne inférieure certifiée associée ; requise pour `optimal` et `lower_bound_only`, nulle sinon. |
| `integrality_tolerance` | réel | Tolérance d’intégralité déclarée de la méthode. |
| `feasibility_tolerance` | réel | Tolérance de faisabilité déclarée de la méthode. |
| `method_limits` | chaîne | Limites explicites de la méthode (bornes sur types et demandes, garde-fou mémoire…). |
| `error_message` | chaîne nullable | Diagnostic non vide en cas d’échec. |
| `code_commit`, `python_version`, `dependency_versions`, `hardware_id` | chaînes | Environnement de production de la référence. |

### Règles de cohérence

- Un statut `optimal` exige un optimum entier strictement positif et une borne inférieure telle
  que `certified_lower_bound_bars <= integer_optimum_bars + feasibility_tolerance`.
- Un statut `lower_bound_only` n’a pas de preuve d’optimalité entière : `integer_optimum_bars`
  doit rester nul.
- Un statut `failed` ne porte aucune valeur numérique et conserve son diagnostic ; les échecs
  restent persistés.
- Toute comparaison de la baseline à la référence passe par ce schéma persisté ; aucune valeur
  recomposée à la volée ne peut se substituer à une référence enregistrée. Un optimum entier issu
  d’un maître restreint reste qualifié `optimal_over_generated_columns_only` et n’est jamais
  confondu avec cette référence.
