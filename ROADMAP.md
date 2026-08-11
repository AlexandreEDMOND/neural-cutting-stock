# Feuille de route atomique

Cette feuille de route pilote le développement autonome du projet. Elle répond à une seule
question : un composant appris peut-il accélérer significativement la génération de colonnes du
Cutting Stock 1D sur les grandes instances, sans dégrader sciemment la qualité de solution ?

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
- [ ] **P3.14** — Publier `results/phase-3-summary.md` et les figures de données justifiées par le corpus réel.
- [ ] **P3.15** — Mettre à jour le README avec le schéma, les partitions et la clôture de Phase 3.

Critère de sortie : les trajectoires sont validées et rejouables, aucune fuite n'existe entre les
partitions et la collecte respecte les décisions classiques dans les tolérances déclarées.

## Phase 4 — Sélection apprise de colonnes

Objectif : ajouter la plus petite politique apprise utile, sans remplacer le solveur ni son contrôle
exact de convergence.

- [ ] **P4.01** — Figer l'interface état, motif, score et décision entre solveur classique et couche apprise.
- [ ] **P4.02** — Construire des features compatibles avec un nombre variable et une permutation des types.
- [ ] **P4.03** — Exposer un pool déterministe de candidats sans modifier le pricing exact final.
- [ ] **P4.04** — Implémenter le chargement reproductible des exemples et partitions de Phase 3.
- [ ] **P4.05** — Implémenter le plus petit modèle supervisé capable de scorer état et motif.
- [ ] **P4.06** — Ajouter une commande d'entraînement avec graines, configuration et métadonnées persistées.
- [ ] **P4.07** — Évaluer hors entraînement le classement des colonnes avec des métriques définies à l'avance.
- [ ] **P4.08** — Versionner les artefacts du modèle et vérifier leur compatibilité au chargement.
- [ ] **P4.09** — Intégrer la politique de sélection avec budget de candidats configurable.
- [ ] **P4.10** — Garantir le pricing exact final, le fallback exact et la vérification indépendante du plan.
- [ ] **P4.11** — Exposer `--solver neural` sans rendre PyTorch obligatoire pour le mode classique.
- [ ] **P4.12** — Comparer Classical CG et Neural CG sur validation avec temps et qualité appariés.
- [ ] **P4.13** — Auditer la Phase 4 et retirer code mort, doublons et expérimentations non retenues.
- [ ] **P4.14** — Publier `results/phase-4-summary.md` et les figures neuronales issues de mesures réelles.
- [ ] **P4.15** — Mettre à jour le README avec le modèle, les garanties, les résultats et la clôture de Phase 4.

Critère de sortie : le mode neural produit des solutions vérifiées, ne termine qu'après contrôle
exact et réduit sur validation un coût pertinent sans dégrader la qualité au-delà du seuil déclaré.

## Phase 5 — Optimisation mesurée et robustesse

Objectif : optimiser uniquement les goulots observés et décider expérimentalement si une politique
séquentielle plus complexe est justifiée.

- [ ] **P5.01** — Profiler de bout en bout le mode neural, préparation et inférence comprises.
- [ ] **P5.02** — Comparer le profil neural au profil classique sur les mêmes instances et ressources.
- [ ] **P5.03** — Documenter la décision mesurée d'entreprendre ou non une optimisation séquentielle.
- [ ] **P5.04** — Définir état, action et horizon si justifié, sinon formaliser l'optimisation alternative retenue.
- [ ] **P5.05** — Définir une récompense ou métrique d'optimisation alignée sur temps total et qualité.
- [ ] **P5.06** — Implémenter la plus petite optimisation conforme à la décision P5.03.
- [ ] **P5.07** — Conserver des budgets explicites, un fallback exact et des limites de ressources.
- [ ] **P5.08** — Mesurer puis optimiser, si utile, batching, cache ou réutilisation d'état solveur.
- [ ] **P5.09** — Tester les cas limites, timeouts, erreurs modèle et indisponibilités d'artefact.
- [ ] **P5.10** — Ajouter des tests de non-régression sur qualité, convergence et reproductibilité.
- [ ] **P5.11** — Mesurer mémoire, appels exacts, candidats, colonnes retenues et coût d'inférence.
- [ ] **P5.12** — Geler le candidat optimisé uniquement s'il améliore le temps total sur validation.
- [ ] **P5.13** — Auditer la Phase 5 et retirer code mort, doublons et branches expérimentales rejetées.
- [ ] **P5.14** — Publier `results/phase-5-summary.md` et les figures de décision issues des mesures réelles.
- [ ] **P5.15** — Mettre à jour le README avec la décision, les optimisations et la clôture de Phase 5.

Critère de sortie : toute complexité conservée montre un gain end-to-end mesuré sans perte de
qualité ; sinon la politique supervisée de Phase 4 reste la solution retenue et la décision est tracée.

## Phase 6 — Évaluation finale et résultat reproductible

Objectif : répondre sans ambiguïté à la question de recherche sur des instances non vues.

- [ ] **P6.01** — Geler code, dépendances, modèles, configurations, partitions et protocole final.
- [ ] **P6.02** — Générer et valider le manifeste final des instances non vues et des catégories de taille.
- [ ] **P6.03** — Exécuter la baseline classique finale avec données brutes et environnement tracés.
- [ ] **P6.04** — Exécuter le mode neural final sur exactement les mêmes instances et ressources.
- [ ] **P6.05** — Vérifier chaque paire et calculer les différences d'objectif avant toute agrégation.
- [ ] **P6.06** — Répéter les mesures selon la variabilité observée et reporter l'incertitude.
- [ ] **P6.07** — Tester la généralisation vers des tailles supérieures à celles d'entraînement.
- [ ] **P6.08** — Conserver et analyser échecs, violations, fallbacks et timeouts des deux modes.
- [ ] **P6.09** — Produire les tableaux appariés de qualité, runtime, mémoire, itérations et colonnes.
- [ ] **P6.10** — Générer `results/runtime_comparison.png` uniquement depuis les résultats finaux validés.
- [ ] **P6.11** — Générer `results/speedup_by_size.png` uniquement depuis les paires de qualité admissible.
- [ ] **P6.12** — Rédiger la conclusion scientifique, les limites et les conditions de reproductibilité.
- [ ] **P6.13** — Effectuer l'audit final et retirer code mort, doublons et artefacts temporaires inutiles.
- [ ] **P6.14** — Publier `results/phase-6-summary.md`, le manifeste final et les commandes de reproduction.
- [ ] **P6.15** — Mettre à jour le README final avec résultats, figures, limites et réponse à l'hypothèse.

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
