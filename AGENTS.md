# Constitution du projet

Ce fichier s’applique à l’ensemble du dépôt. Toute session de développement doit préserver les invariants ci-dessous. En cas de conflit entre une idée d’implémentation et cette constitution, la constitution prévaut sauf demande explicite du responsable du projet.

## Mission unique

Répondre expérimentalement à la question suivante :

> Un agent de deep RL peut-il améliorer la qualité des solutions de découpe (moins de barres, moins de perte) produites pour le Cutting Stock 1D par rapport à la baseline classique de génération de colonnes, sans aucune contrainte de temps mur-à-mur ?

Note historique : les phases 1 à 6 ont répondu à la question initiale d'accélération du temps
mur-à-mur par une réponse négative rigoureusement documentée ([docs/conclusion.md](docs/conclusion.md)).
Cette réponse reste valable et close ; la mission évolue vers la qualité à partir de la phase 7.

## Invariants non négociables

1. Le problème étudié est exclusivement le **Cutting Stock 1D**, y compris ses variantes monodimensionnelles déclarées (multi-formats de barres, kerf exercé).
2. La **génération de colonnes** est le socle d’optimisation classique et la baseline de comparaison.
3. Le ML sert à **améliorer la qualité des solutions** : proposition de colonnes supplémentaires, amélioration primal au-delà du maître entier restreint, ou guidage de la recherche ; il ne remplace pas le solveur ni la vérification exacte.
4. Le temps mur-à-mur est **hors périmètre d’évaluation** : une méthode plus lente est acceptable dès lors qu’elle améliore l’objectif mesuré. Toute différence doit être mesurée et visible.
5. Un pricing exact ou un fallback exact reste le garde-fou avant toute déclaration de convergence.
6. La métrique de performance principale est l’**objectif de découpe** (nombre de barres, puis perte totale) comparé sur des paires d’instances identiques, et l’écart à une référence exacte lorsqu’elle existe. Les durées sont enregistrées à titre informatif et ne constituent jamais un critère de succès.
7. Le kerf est supporté dès le modèle du problème, avec la convention documentée ; son exercice effectif dans les campagnes devient un axe de recherche à partir de la phase 8.
8. Les autres familles de solveurs sont hors périmètre sauf demande explicite : Pointer Networks, solveur RL end-to-end remplaçant la génération de colonnes, algorithmes génétiques, recuit simulé, AlphaZero, bin-packing RL générique, découpe 2D/3D, routage, ordonnancement, multi-usines ou arrivées stochastiques. Le deep RL d’amélioration primal adossé à la boucle classique est au contraire le sujet central à partir de la phase 9.
9. Les benchmarks doivent être reproductibles : graines, configuration, version du code, environnement, statuts et données brutes doivent être traçables.
10. Toute figure de performance doit provenir de mesures réelles. Ne jamais inventer, interpoler comme mesure, ni publier de nombres synthétiques comme résultats.

## Garanties à nommer correctement

- Un pricing exact sans colonne améliorante certifie l’optimalité de la relaxation linéaire du maître complet, à la tolérance numérique déclarée.
- Le maître entier final est résolu sur les colonnes générées. Sans branch-and-price ou preuve additionnelle, son résultat ne doit pas être qualifié d’optimum entier global du problème complet.
- Les contraintes de demande et de capacité ne dépendent jamais d’une prédiction neuronale ; toute solution publiée doit être vérifiée indépendamment.
- Les tolérances de coût réduit, d’intégralité et de faisabilité sont explicites dans la configuration et les résultats.

## Discipline d’architecture

- `problem/` porte les données, invariants et validations du Cutting Stock 1D.
- `solver/` porte le RMP, le pricing exact, le maître entier et l’orchestration de Column Generation. Il ne dépend pas du ML.
- `benchmarks/` génère des instances reproductibles, exécute les solveurs et écrit le schéma commun.
- `visualization/` lit uniquement des résultats persistés et validés ; elle ne contient pas de résultats codés en dur.
- `learning/` ne sera introduit qu’en Phase 4 et dépendra des interfaces classiques, jamais l’inverse.
- `scripts/` reste une couche d’entrée fine ; la logique testable demeure dans `src/neural_cutting_stock/`.
- Éviter les abstractions spéculatives et les dossiers d’algorithmes alternatifs.

## Règles d’expérimentation

- Comparer Classical CG et les approches neurales sur les mêmes identifiants d’instances et les mêmes limites de ressources.
- Enregistrer l’objectif de découpe en priorité ; les temps total et par composant (RMP, pricing, maître entier, gestion des colonnes, inférence, entraînement) restent journalisés à titre informatif.
- Ne jamais conclure à une amélioration de qualité sans comparaison appariée et, lorsque disponible, référence exacte ou borne inférieure certifiée.
- Vérifier et reporter `objective_difference_vs_classical` avant toute agrégation.
- Conserver les échecs, timeouts et violations dans les rapports ; ne pas les filtrer silencieusement.
- Ne fixer les seuils `SMALL`/`MEDIUM`/`LARGE`/`XL` qu’après profilage, puis versionner leur définition.

## Processus attendu pour chaque changement

1. Lire [ROADMAP.md](ROADMAP.md) et traiter uniquement sa première case non cochée.
2. Ajouter ou adapter les tests de correction et de reproductibilité pertinents.
3. Mesurer avant d’optimiser ; ne pas ajouter une complexité dont le profilage ne montre pas le besoin.
4. Documenter toute modification de formulation, métrique, tolérance ou protocole.
5. Ne jamais incorporer de faux résultats, même à titre décoratif.
6. Une itération correspond à une case, un changement borné et un commit ; ne jamais commencer la case suivante dans le même cycle.
7. Cocher uniquement la case terminée après validation. Une étape nouvellement découverte peut être ajoutée non cochée avec un identifiant suffixé, sans être réalisée immédiatement.

## Progression autorisée

La première case non cochée de [ROADMAP.md](ROADMAP.md) est l’unique source de vérité sur la phase et le travail autorisés. Quand toutes les cases d’une phase sont cochées, la première case de la phase suivante devient automatiquement autorisée. Aucun texte de ce fichier ne doit figer manuellement une « phase courante » susceptible de devenir obsolète.
