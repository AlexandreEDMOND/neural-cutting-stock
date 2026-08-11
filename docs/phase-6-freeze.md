# Gel de l'évaluation finale

Le gel P6.01 est décrit par `configs/phase-6-final.json`. Son `code_commit` est
le seul identifiant de version du code ; les dépendances sont celles de `uv.lock`.
La configuration fixe le contrat `benchmark-run-v1`, les tolérances, la politique
de chargement, l'ordre d'exécution et une répétition par instance. Aucun résultat
de Phase 6 n'est inclus dans ce gel.

Le candidat opérationnel est l'artefact versionné
`models/linear-scorer-v1-zero-weight.json`, déjà identifié dans le bilan de Phase 5.
Il conserve l'interface `linear-scorer-v1` et la politique
`bounded-column-selection-v1`; ses poids n'impliquent aucun résultat d'apprentissage
ou de performance supplémentaire.

Les partitions train, validation et test sont celles du manifeste `phase-3-corpus-v1`.
La partition `test` est réservée à l'évaluation finale ; les futures instances non vues
et leurs catégories de taille seront ajoutées et validées par P6.02, pas par ce gel.

Les chemins des fichiers gelés sont maintenus dans la configuration. Toute modification
du code, des dépendances, de l'artefact, de la configuration ou du manifeste invalide le
gel et doit être signalée avant l'exécution.
