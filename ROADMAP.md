# Feuille de route atomique

Cette feuille de route pilote le développement autonome du projet. Les phases 1 à 6 ont répondu à
la question initiale — un composant appris peut-il accélérer significativement la génération de
colonnes sans dégrader la qualité ? — par une réponse négative documentée dans
`docs/conclusion.md`. À compter de la phase 7, elle répond à une nouvelle question :

> Un agent de deep RL peut-il améliorer la qualité des solutions de découpe (moins de barres,
> moins de perte) par rapport à la baseline classique de génération de colonnes, sans aucune
> contrainte de temps mur-à-mur ?

La qualité est la métrique reine ; les durées sont journalisées mais ne constituent jamais un
critère de succès.

## Règle d'exécution

- Une case cochable correspond à une itération, un changement borné et un commit.
- La première case non cochée, lue de haut en bas, est l'unique travail autorisé du cycle.
- Une case n'est cochée qu'après implémentation, tests et validation de ses critères.
- Les cases suivantes ne sont jamais commencées dans le même cycle.
- Une étape indispensable découverte en cours de route peut être ajoutée sous forme de nouvelle
  case non cochée avec un suffixe stable (`P2.04a`, par exemple). Elle ne doit pas être réalisée
  dans le cycle qui l'ajoute.
- Chaque phase se termine par trois itérations obligatoires : nettoyage du code, publication d'un
  bilan fondé sur des exécutions réelles, puis mise à jour du README et validation de la phase.
- Une image n'est ajoutée que si des données réelles la rendent informative. Un bilan textuel est
  toujours acceptable ; aucun résultat, nombre ou graphique ne doit être inventé.

## Phase 1 — Baseline classique de génération de colonnes

Objectif : disposer d'un solveur classique correct, vérifié et honnête sur ses garanties.

- [x] **P1.01** — Modéliser et valider les instances (`stock_length`, kerf, longueurs et demandes).
- [x] **P1.02** — Tester la convention de kerf, y compris les limites décimales et motifs partagés.
- [x] **P1.03** — Construire des motifs homogènes initiaux garantissant un RMP faisable.
- [x] **P1.04** — Résoudre le RMP linéaire et tester la convention de signe des valeurs duales.
- [x] **P1.05** — Implémenter le pricing entier exact et le calcul explicite du coût réduit.
- [x] **P1.06** — Comparer le pricing exact à une énumération exhaustive sur de petites capacités.
- [x] **P1.07** — Dédupliquer les colonnes et appliquer une tolérance de coût réduit explicite.
- [x] **P1.08** — Itérer la génération de colonnes jusqu'au contrôle exact de convergence.
- [x] **P1.09** — Comparer la relaxation finale au maître exhaustif sur un corpus vérifiable.
- [x] **P1.10** — Résoudre le maître entier restreint et nommer correctement sa garantie.
- [x] **P1.11** — Vérifier indépendamment capacité, couverture, objectif, kerf et bilan matière.
- [x] **P1.12** — Exposer le solveur classique, ses diagnostics et ses statuts via la CLI structurée.
- [x] **P1.13** — Auditer la Phase 1 et retirer code mort, doublons et abstractions devenues inutiles.
- [x] **P1.14** — Exécuter la validation de Phase 1 et publier `results/phase-1-summary.md` avec les résultats réels.
- [x] **P1.15** — Mettre à jour le README avec l'état, les garanties, les résultats et la clôture de Phase 1.

Critère de sortie : tous les tests passent ; sur le corpus exhaustif, la relaxation de Column
Generation atteint la même valeur que le maître complet ; la solution entière restreinte est
vérifiée et sa garantie est correctement reportée.

## Phase 2 — Benchmark classique et profilage

Objectif : mesurer la baseline classique, identifier son goulot réel et figer un protocole
reproductible avant tout apprentissage.

- [x] **P2.01** — Créer un générateur synthétique déterministe piloté par une graine explicite.
- [x] **P2.02** — Valider les dimensions et paramètres de configuration du générateur.
- [x] **P2.03** — Attribuer des identifiants stables et reproductibles aux instances générées.
- [x] **P2.04** — Définir le schéma versionné des résultats bruts, statuts et métadonnées d'environnement.
- [x] **P2.05** — Implémenter un runner classique sur une matrice d'instances configurée.
- [x] **P2.06** — Persister toutes les exécutions brutes sans filtrer échecs, violations ni timeouts.
- [x] **P2.07** — Mesurer temps total, RMP, pricing, maître entier et gestion des colonnes.
- [x] **P2.08** — Construire une matrice séparant types, demande, longueur de barre, distributions et kerf.
- [x] **P2.09** — Tester la reproductibilité des instances, ordres d'exécution et résultats sérialisés.
- [x] **P2.10** — Ajouter limites de ressources et statuts explicites aux campagnes de benchmark.
- [x] **P2.11** — Profiler la baseline et identifier le goulot dominant avec des mesures persistées.
- [x] **P2.12** — Définir et versionner `SMALL`, `MEDIUM`, `LARGE`, `XL` à partir des profils réels.
- [x] **P2.13** — Auditer la Phase 2 et retirer code mort, doublons et instrumentation sans valeur mesurée.
- [x] **P2.14** — Publier `results/phase-2-summary.md` et les figures classiques justifiées par les données réelles.
- [x] **P2.15** — Mettre à jour le README avec le protocole, les profils et la clôture de Phase 2.

Critère de sortie : un corpus classique reproductible couvre plusieurs régimes de difficulté, le
goulot principal est mesuré et les catégories de taille sont justifiées par les données.

## Phase 3 — Trajectoires et données d'apprentissage

Objectif : collecter des trajectoires rejouables sans modifier les décisions ni les garanties du
solveur classique.

- [x] **P3.01** — Définir et versionner le schéma d'une trajectoire de génération de colonnes.
- [x] **P3.02** — Enregistrer les états du RMP et les identifiants d'instance à chaque itération.
- [x] **P3.03** — Enregistrer les duales avec ordre des types, tolérances et conventions explicites.
- [x] **P3.04** — Enregistrer motifs candidats, coûts réduits et résultat du pricing exact.
- [x] **P3.05** — Enregistrer colonnes retenues, progrès, durées et décisions de fallback.
- [x] **P3.06** — Implémenter un validateur et un lecteur capables de rejouer une trajectoire.
- [x] **P3.07** — Mesurer le surcoût de collecte et vérifier qu'il n'altère pas les décisions classiques.
- [x] **P3.08** — Définir une cible de colonne utile liée à une réduction mesurable du travail total.
- [x] **P3.09** — Définir les partitions train, validation et test par graine et famille avant collecte.
- [x] **P3.10** — Construire le dataset à partir des seules trajectoires validées.
- [x] **P3.11** — Tester l'absence de fuite d'instance et l'invariance à l'ordre des types.
- [x] **P3.12** — Produire un petit corpus versionné ou reproductible avec manifeste et statistiques réelles.
- [x] **P3.13** — Auditer la Phase 3 et retirer code mort, doublons et champs de trajectoire inutilisés.
- [x] **P3.14** — Publier `results/phase-3-summary.md` et les figures de données justifiées par le corpus réel.
- [x] **P3.15** — Mettre à jour le README avec le schéma, les partitions et la clôture de Phase 3.

Critère de sortie : les trajectoires sont validées et rejouables, aucune fuite n'existe entre les
partitions et la collecte respecte les décisions classiques dans les tolérances déclarées.

## Phase 4 — Sélection apprise de colonnes

Objectif : ajouter la plus petite politique apprise utile, sans remplacer le solveur ni son contrôle
exact de convergence.

- [x] **P4.01** — Figer l'interface état, motif, score et décision entre solveur classique et couche apprise.
- [x] **P4.02** — Construire des features compatibles avec un nombre variable et une permutation des types.
- [x] **P4.03** — Exposer un pool déterministe de candidats sans modifier le pricing exact final.
- [x] **P4.04** — Implémenter le chargement reproductible des exemples et partitions de Phase 3.
- [x] **P4.05** — Implémenter le plus petit modèle supervisé capable de scorer état et motif.
- [x] **P4.06** — Ajouter une commande d'entraînement avec graines, configuration et métadonnées persistées.
- [x] **P4.07** — Évaluer hors entraînement le classement des colonnes avec des métriques définies à l'avance.
- [x] **P4.08** — Versionner les artefacts du modèle et vérifier leur compatibilité au chargement.
- [x] **P4.09** — Intégrer la politique de sélection avec budget de candidats configurable.
- [x] **P4.10** — Garantir le pricing exact final, le fallback exact et la vérification indépendante du plan.
- [x] **P4.11** — Exposer `--solver neural` sans rendre PyTorch obligatoire pour le mode classique.
- [x] **P4.12** — Comparer Classical CG et Neural CG sur validation avec temps et qualité appariés.
- [x] **P4.13** — Auditer la Phase 4 et retirer code mort, doublons et expérimentations non retenues.
- [x] **P4.14** — Publier `results/phase-4-summary.md` et les figures neuronales issues de mesures réelles.
- [x] **P4.15** — Mettre à jour le README avec le modèle, les garanties, les résultats et la clôture de Phase 4.

Critère de sortie : le mode neural produit des solutions vérifiées, ne termine qu'après contrôle
exact et réduit sur validation un coût pertinent sans dégrader la qualité au-delà du seuil déclaré.

## Phase 5 — Optimisation mesurée et robustesse

Objectif : optimiser uniquement les goulots observés et décider expérimentalement si une politique
séquentielle plus complexe est justifiée.

- [x] **P5.01** — Profiler de bout en bout le mode neural, préparation et inférence comprises.
- [x] **P5.02** — Comparer le profil neural au profil classique sur les mêmes instances et ressources.
- [x] **P5.03** — Documenter la décision mesurée d'entreprendre ou non une optimisation séquentielle.
- [x] **P5.04** — Définir état, action et horizon si justifié, sinon formaliser l'optimisation alternative retenue.
- [x] **P5.05** — Définir une récompense ou métrique d'optimisation alignée sur temps total et qualité.
- [x] **P5.06** — Implémenter la plus petite optimisation conforme à la décision P5.03.
- [x] **P5.07** — Conserver des budgets explicites, un fallback exact et des limites de ressources.
- [x] **P5.08** — Mesurer puis optimiser, si utile, batching, cache ou réutilisation d'état solveur.
- [x] **P5.09** — Tester les cas limites, timeouts, erreurs modèle et indisponibilités d'artefact.
- [x] **P5.10** — Ajouter des tests de non-régression sur qualité, convergence et reproductibilité.
- [x] **P5.11** — Mesurer mémoire, appels exacts, candidats, colonnes retenues et coût d'inférence.
- [x] **P5.12** — Geler le candidat optimisé uniquement s'il améliore le temps total sur validation.
- [x] **P5.13** — Auditer la Phase 5 et retirer code mort, doublons et branches expérimentales rejetées.
- [x] **P5.14** — Publier `results/phase-5-summary.md` et les figures de décision issues des mesures réelles.
- [x] **P5.15** — Mettre à jour le README avec la décision, les optimisations et la clôture de Phase 5.

Critère de sortie : toute complexité conservée montre un gain end-to-end mesuré sans perte de
qualité ; sinon la politique supervisée de Phase 4 reste la solution retenue et la décision est tracée.

## Phase 6 — Évaluation finale et résultat reproductible

Objectif : répondre sans ambiguïté à la question de recherche sur des instances non vues.

- [x] **P6.01** — Geler code, dépendances, modèles, configurations, partitions et protocole final.
- [x] **P6.02** — Générer et valider le manifeste final des instances non vues et des catégories de taille.
- [x] **P6.03** — Exécuter la baseline classique finale avec données brutes et environnement tracés.
- [x] **P6.04** — Exécuter le mode neural final sur exactement les mêmes instances et ressources.
- [x] **P6.05** — Vérifier chaque paire et calculer les différences d'objectif avant toute agrégation.
- [x] **P6.06** — Répéter les mesures selon la variabilité observée et reporter l'incertitude.
- [x] **P6.07** — Tester la généralisation vers des tailles supérieures à celles d'entraînement.
- [x] **P6.08** — Conserver et analyser échecs, violations, fallbacks et timeouts des deux modes.
- [x] **P6.09** — Produire les tableaux appariés de qualité, runtime, mémoire, itérations et colonnes.
- [x] **P6.10** — Générer `results/runtime_comparison.png` uniquement depuis les résultats finaux validés.
- [x] **P6.11** — Générer `results/speedup_by_size.png` uniquement depuis les paires de qualité admissible.
- [x] **P6.12** — Rédiger la conclusion scientifique, les limites et les conditions de reproductibilité.
- [x] **P6.13** — Effectuer l'audit final et retirer code mort, doublons et artefacts temporaires inutiles.
- [x] **P6.14** — Publier `results/phase-6-summary.md`, le manifeste final et les commandes de reproduction.
- [x] **P6.15** — Mettre à jour le README final avec résultats, figures, limites et réponse à l'hypothèse.

Critère de succès : Neural CG conserve une qualité comparable à Classical CG tout en réduisant
significativement le temps mur-à-mur, avec un bénéfice particulièrement visible sur les instances
les plus difficiles. Un modèle rapide mais moins bon, ou un gain limité à l'inférence, n'est pas un
succès.

## Risques suivis

- **Coût dominant mal ciblé** : profiler avant de concevoir ou complexifier le modèle.
- **Peu de marge pour apprendre** : mesurer pools multi-colonnes et gestion du RMP sans abandonner
  le backbone exact.
- **Garantie entière limitée** : ne jamais présenter l'Integer RMP comme un optimum entier global.
- **Overhead neural** : inclure préparation des features, inférence et fallback dans le temps total.
- **Mauvaise généralisation** : tester explicitement les tailles et familles hors distribution.
- **Biais de benchmark** : figer générateurs, graines, catégories et protocole avant l'évaluation.

## Prochaine itération

La prochaine itération est toujours la première case `- [ ]` de ce fichier. Aucun identifiant
n'est recopié dans cette section afin que cette règle reste correcte après chaque commit.

## Phase 7 — Référence exacte et mesure du trou d'optimalité entier

Objectif : établir la vérité terrain de la qualité. La baseline classique ne certifie son maître
entier que sur colonnes générées ; il faut une référence exacte pour savoir combien de barres
restent gagnables avant de chercher à les gagner.

- [x] **P7.01** — Définir le schéma versionné `exact-reference-v1` : instance_id, méthode de référence, statut, optimum entier, borne inférieure associée, limites de la méthode, environnement.
- [ ] **P7.02** — Implémenter l'énumération exhaustive des motifs maximaux pour instances bornées (types et demandes petits) avec génération paresseuse et garde-fou mémoire.
- [ ] **P7.03** — Résoudre le maître entier complet par MILP sur motifs énumérés via `scipy.optimize.milp` (HiGHS déjà présent) ; statut et preuve persistés dans `exact-reference-v1`.
- [ ] **P7.04** — Vérifier indépendamment chaque référence exacte : faisabilité du plan, cohérence borne LP ≤ optimum entier, contrôle croisé énumération/MILP sur un sous-échantillon.
- [ ] **P7.05** — Calculer l'écart de la baseline classique (`optimal_over_generated_columns_only`) à la référence exacte sur tout le corpus existant et persister `results/exact-gap.*`.
- [ ] **P7.06** — Publier le bilan chiffré des écarts par famille et taille : où une marge de qualité existe réellement, où elle est nulle.
- [ ] **P7.07** — Si la marge est quasi nulle partout, identifier dans le générateur déterministe les paramètres créant des trous entiers non triviaux (demandes peu divisibles, ratios tendus) et documenter ces leviers sans encore les activer.
- [ ] **P7.08** — Tests de non-régression : la CG classique reste optimale LP et inchangée en objectif sur toutes les instances disposant d'une référence exacte.
- [ ] **P7.09** — Nettoyage du code de la phase : consolidation des helpers, suppression du code mort.
- [ ] **P7.10** — Publication du bilan de phase fondé sur exécutions réelles, mise à jour du README et validation de la phase 7.

## Phase 8 — Familles d'instances à marge de qualité

Objectif : construire le benchmark où gagner des barres est possible. Une marge nulle rendrait la
piste RL invérifiable ; il faut des familles où la baseline classique perd mesurablement des
barres, tout en gardant une vérification exacte accessible.

- [ ] **P8.01** — Étendre le générateur déterministe au kerf strictement positif exercé (limite connue de la campagne finale) avec tests de convention conservative.
- [ ] **P8.02** — Ajouter le multi-formats de barres (2 à 3 longueurs de stock) comme variante monodimensionnelle déclarée, schéma et validations inclus.
- [ ] **P8.03** — Créer des profils de demande structurée défavorables à l'arrondi du maître restreint (divisibilité difficile, ratios tendus) et vérifier qu'ils produisent des trous entiers non triviaux.
- [ ] **P8.04** — Pousser la taille des instances (plus de types, demandes élevées) jusqu'au maintien possible d'une référence exacte MILP ou d'une borne inférieure certifiée.
- [ ] **P8.05** — Mesurer l'écart classique-vs-référence sur chaque nouvelle famille ; ne retenir que celles présentant une marge positive sur une part significative des instances.
- [ ] **P8.06** — Geler partitions entraînement/validation/test des familles retenues avec manifestes versionnés et sans fuite.
- [ ] **P8.07** — Étendre le schéma de résultats aux nouveaux champs (formats multiples, kerf exercé) avec rétrocompatibilité des campagnes antérieures.
- [ ] **P8.08** — Bilan intermédiaire : tableau des marges par famille, choix documenté du benchmark qualité final.
- [ ] **P8.09** — Nettoyage du code de la phase.
- [ ] **P8.10** — Publication du bilan de phase, mise à jour du README et validation de la phase 8.

## Phase 9 — Agent deep RL d'amélioration primal

Objectif : entraîner un agent qui améliore l'objectif au-delà du maître entier restreint classique,
sans contrainte de temps. Le solveur et la vérification exacte restent le socle ; l'agent propose,
le vérificateur dispose.

- [ ] **P9.01** — Définir l'interface `quality-agent` : entrées (instance, pool de colonnes, solution courante), sortie (motifs ou colonnes supplémentaires proposés), contrat de vérification indépendante systématique.
- [ ] **P9.02** — Construire l'environnement RL : épisode = raffinement itératif de la solution entière, observation = état du pool et de la solution, récompense = réduction de barres ou de perte, pénalité stricte pour plan invalide.
- [ ] **P9.03** — Introduire PyTorch comme dépendance justifiée et versionnée, avec entraînement reproductible : graines, checkpoints versionnés, courbes persistées.
- [ ] **P9.04** — Baseline d'apprentissage par imitation du choix exact sur petites instances, afin de valider l'interface avant tout RL profond.
- [ ] **P9.05** — Entraîner une politique profonde (algorithme documenté et justifié) sur les familles à marge de la phase 8, avec journal complet des expériences.
- [ ] **P9.06** — Intégrer le pipeline Neural-QC : partir de la solution classique restreinte, appliquer l'agent jusqu'à convergence ou budget d'amélioration déclaré.
- [ ] **P9.07** — Garde-fous de publication : toute solution finale vérifiée indépendamment, statuts honnêtes (amélioré/égal/dégradé), échecs conservés.
- [ ] **P9.08** — Évaluation offline sur partition de validation : gain moyen de barres vs baseline classique, par famille et taille.
- [ ] **P9.09** — Ablations obligatoires : recherche aléatoire et gloutonne à budget égal, pour prouver l'apport propre de l'apprentissage.
- [ ] **P9.10** — Nettoyage du code de la phase.
- [ ] **P9.11** — Publication du bilan de phase, mise à jour du README et validation de la phase 9.

## Phase 10 — Évaluation finale qualité et réponse

Objectif : répondre sans ambiguïté à la nouvelle question de recherche sur des instances non vues,
avec la même rigueur que la phase 6 mais la qualité comme métrique reine.

- [ ] **P10.01** — Geler code, dépendances, checkpoints, configurations, partitions et protocole final qualité.
- [ ] **P10.02** — Générer et valider le manifeste des instances non vues issues des familles à marge.
- [ ] **P10.03** — Exécuter la campagne appariée Classical CG vs Neural-QC sur instances non vues, répétitions incluses, durées journalisées hors critère.
- [ ] **P10.04** — Comparer chaque solution aux références exactes quand elles existent, sinon aux bornes inférieures certifiées.
- [ ] **P10.05** — Analyser l'incertitude des gains (intervalles de confiance par instance et agrégés).
- [ ] **P10.06** — Produire les figures de qualité (barres gagnées par famille et taille) dérivées exclusivement des données validées.
- [ ] **P10.07** — Rédiger la conclusion scientifique : réponse à la question qualité, portée exacte, limites et conditions de reproduction.
- [ ] **P10.08** — Publier les résultats finaux, mettre à jour le README et valider la phase 10.
