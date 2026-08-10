# Feuille de route

Cette feuille de route organise un seul programme de recherche : accélérer la génération de colonnes du Cutting Stock 1D par une politique apprise, sans dégrader sciemment la qualité de solution.

Chaque phase a un critère de sortie. Une phase ne doit pas être déclarée terminée parce que du code existe, mais parce que ses garanties sont testées et ses artefacts reproductibles.

## Phase 0 — Fondations du dépôt

### Objectifs

- fixer la formulation mathématique et la convention de kerf ;
- fixer les frontières entre problème, solveur, benchmarks et future couche apprise ;
- définir les métriques, statuts, chronométrages et règles de reproductibilité ;
- créer le squelette Python et la constitution anti-dérive ;
- interdire les affirmations de performance sans mesure.

### Livrables

- [README.md](README.md) ;
- [AGENTS.md](AGENTS.md) ;
- [docs/formulation.md](docs/formulation.md) ;
- [docs/benchmark_protocol.md](docs/benchmark_protocol.md) ;
- configuration de packaging, qualité et tests ;
- paquets vides mais importables nécessaires à la Phase 1.

### Critère de sortie

Le dépôt s’installe, les tests de structure passent et les décisions qui affectent la correction ou les futurs résultats sont documentées sans ambiguïté.

## Phase 1 — Baseline classique de Column Generation

### Objectifs

1. Modéliser et valider une instance (`stock_length`, `kerf`, longueurs, demandes).
2. Construire des motifs homogènes initiaux garantissant un RMP faisable.
3. Résoudre le RMP linéaire et extraire des duales avec une convention de signe testée.
4. Résoudre exactement le pricing entier et calculer le coût réduit.
5. Dédupliquer les colonnes, appliquer une tolérance explicite et itérer jusqu’au contrôle exact de convergence.
6. Résoudre le maître entier sur l’ensemble final de colonnes.
7. Vérifier indépendamment capacité, couverture, objectif, kerf et bilan matière du plan produit.
8. Exposer un point d’entrée `--solver classical` avec une sortie structurée.

### Tests requis

- cas manuels à un ou quelques types, avec et sans kerf ;
- cas où plusieurs types partagent un motif utile ;
- rejet des longueurs ou demandes invalides et des pièces qui ne tiennent pas ;
- comparaison du pricing avec une énumération exhaustive sur de petites capacités ;
- absence de coût réduit négatif après convergence ;
- reproductibilité et vérification du plan entier ;
- statut explicite en cas d’infaisabilité, limite de temps ou erreur solveur.

### Critère de sortie

Tous les tests passent et, sur un petit corpus vérifiable par énumération, la relaxation de Column Generation atteint la même valeur que le maître contenant tous les motifs. La solution entière restreinte est faisable et son niveau de garantie est correctement reporté.

## Phase 2 — Benchmark classique et profilage

### Objectifs

- créer un générateur synthétique déterministe avec graine et version ;
- faire varier séparément nombre de types, demande totale, longueur de barre, distributions et kerf ;
- enregistrer le schéma commun défini dans le protocole ;
- profiler RMP, pricing, maître entier, itérations et gestion des colonnes ;
- établir une baseline classique immuable pour la première comparaison ;
- justifier puis versionner les catégories `SMALL`, `MEDIUM`, `LARGE`, `XL`.

### Décisions pilotées par les mesures

- SciPy/HiGHS suffit-il ou faut-il employer directement `highspy` pour réutiliser le modèle/basis ?
- le pricing domine-t-il, ou les résolutions répétées du RMP ?
- faut-il générer une ou plusieurs colonnes exactes par itération ?
- quelle dimension structurelle prédit réellement la difficulté ?

### Critère de sortie

Un jeu de résultats classique reproductible couvre plusieurs régimes de difficulté. Les profils identifient le goulot principal et les catégories de taille sont justifiées par des mesures, non par intuition.

## Phase 3 — Trajectoires et données d’apprentissage

### Objectifs

- instrumenter sans modifier la logique de correction du solveur classique ;
- enregistrer états du RMP, duales, motifs candidats, coûts réduits, progrès et décisions ;
- définir une cible de « colonne utile » reliée à une réduction mesurable du travail total ;
- séparer les instances par graine/famille avant toute construction des jeux train/validation/test ;
- versionner le schéma de trajectoire et vérifier son coût d’instrumentation.

### Critère de sortie

Les trajectoires peuvent être rejouées et validées. Il n’existe aucune fuite d’instance entre les partitions, et la collecte n’altère pas les décisions du solveur classique au-delà des tolérances.

## Phase 4 — Sélection apprise de colonnes

### Objectifs

- créer le plus petit modèle capable de scorer un couple état/motif ;
- utiliser une représentation compatible avec un nombre variable de types et sans dépendance à leur ordre ;
- commencer par une supervision simple à partir des trajectoires ;
- intégrer le modèle comme politique de sélection, avec budget et fallback configurables ;
- garder le solveur classique totalement exécutable sans PyTorch ;
- comptabiliser candidats, colonnes retenues, temps d’inférence et appels exacts.

### Critère de sortie

Le mode neural produit des solutions vérifiées, termine uniquement après le contrôle exact, et montre sur validation qu’il réduit un coût pertinent de la boucle sans dégrader la qualité au-delà du seuil déclaré.

## Phase 5 — Optimisation séquentielle, uniquement si justifiée

L’apprentissage par renforcement n’est pas automatique. Il ne sera entrepris que si la Phase 4 montre qu’une décision myope de classement limite le temps mur-à-mur.

### Objectifs conditionnels

- formuler l’état, l’action et l’horizon à l’intérieur de la boucle CG ;
- aligner la récompense sur le temps total, le nombre de colonnes utiles et la qualité finale ;
- conserver le pricing exact et les vérifications ;
- battre la politique supervisée sur des instances non vues avec un coût d’entraînement justifié.

### Critère de sortie

Une amélioration end-to-end statistiquement robuste par rapport à la Phase 4, sans perte de qualité. À défaut, conserver le modèle supervisé et arrêter cet axe.

## Phase 6 — Évaluation Neural CG

### Protocole

- geler code, modèles, configurations et partitions avant l’évaluation finale ;
- exécuter Classical CG et Neural CG sur les mêmes instances non vues ;
- répéter les mesures selon la variabilité observée ;
- rapporter qualité, temps total, itérations, colonnes, fallback, mémoire et échecs ;
- tester explicitement la généralisation vers des tailles supérieures à l’entraînement.

### Critère de sortie

Les tableaux appariés permettent de répondre sans ambiguïté à deux questions : la qualité est-elle préservée, et le speedup augmente-t-il sur `LARGE`/`XL` ?

## Phase 7 — Résultat final

### Livrables

- dataset de benchmark et métadonnées reproductibles ;
- rapport des statuts et différences d’objectif ;
- `results/runtime_comparison.png` ;
- `results/speedup_by_size.png` ;
- README mis à jour avec les figures réelles, la méthode et les limites ;
- commandes permettant de reproduire la baseline et les graphiques.

### Critère de succès du projet

Le projet réussit si Neural CG conserve une qualité comparable à Classical CG tout en réduisant significativement le temps mur-à-mur, avec un bénéfice particulièrement visible sur les instances les plus difficiles. Un modèle rapide mais moins bon, ou un gain limité à l’inférence, n’est pas un succès.

## Risques suivis dès le départ

- **Coût dominant mal ciblé** : le scoring de colonnes n’aidera pas si le pricing exact final ou le maître entier domine. Réponse : profiler avant de concevoir le modèle.
- **Peu de marge pour apprendre** : un pricing exact à une colonne peut déjà être peu coûteux. Réponse : mesurer des pools multi-colonnes et la gestion du RMP, sans abandonner le backbone.
- **Garantie entière limitée** : l’Integer RMP peut manquer des colonnes utiles à l’optimum entier. Réponse : reporter honnêtement le statut, comparer les deux modes avec le même protocole et envisager une procédure de réparation ciblée si les mesures le demandent.
- **Overhead neural** : préparation des features et inférence peuvent annuler le gain. Réponse : inclure tout cet overhead dans `total_runtime_seconds`.
- **Mauvaise généralisation en taille** : une politique entraînée sur de petites instances peut échouer sur `XL`. Réponse : représentations invariantes à la permutation, splits par familles et tests hors distribution.
- **Biais de benchmark** : catégories ou instances choisies après observation peuvent favoriser une méthode. Réponse : versionner générateurs, graines, catégories et protocole avant l’évaluation finale.

## Prochain jalon exact

> Implémenter et valider la baseline classique de génération de colonnes, en commençant par le modèle d’instance et ses tests d’invariants.
