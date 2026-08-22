# Conclusion scientifique de la Phase 6

Ce document répond à la question de recherche du projet à partir des seules mesures persistées de
l'évaluation finale. Chaque nombre cité provient d'un des fichiers listés dans « Sources » ; aucun
résultat n'est inventé, interpolé ou importé d'une autre phase.

Sources :

- [`configs/phase-6-final.json`](../configs/phase-6-final.json) — protocole gelé ;
- [`data/phase-6-final/manifest.json`](../data/phase-6-final/manifest.json) — manifeste des instances non vues ;
- [`results/phase-6-classical-campaign.json`](../results/phase-6-classical-campaign.json) et
  [`results/phase-6-neural-campaign.json`](../results/phase-6-neural-campaign.json) — environnement d'exécution ;
- [`results/phase-6-paired-tables.md`](../results/phase-6-paired-tables.md) — tableaux appariés recalculés ;
- [`results/phase-6-failures.json`](../results/phase-6-failures.json) — échecs, timeouts et violations ;
- [`results/phase-6-generalization.json`](../results/phase-6-generalization.json) — généralisation en taille ;
- [`results/phase-6-uncertainty.json`](../results/phase-6-uncertainty.json) — incertitude des répétitions ;
- [`results/runtime_comparison.png`](../results/runtime_comparison.png) et
  [`results/speedup_by_size.png`](../results/speedup_by_size.png) — figures dérivées des paires validées.

## Question et protocole gelé

La question est celle de la constitution : un composant appris peut-il accélérer significativement
la génération de colonnes sur de grandes instances de Cutting Stock 1D, tout en préservant la
qualité de solution du solveur classique ?

Le protocole gelé (`phase-6-final-freeze-v1`) compare Classical CG et Neural CG sur les mêmes 12
instances non vues, appariées par `instance_id` : trois instances par catégorie cible `SMALL`,
`MEDIUM`, `LARGE`, `XL` (2, 4, 6 et 8 types de pièces), générées de façon déterministe à partir des
graines 101–103, 201–203, 301–303 et 401–403, avec `stock_length` 100.0, kerf 0.0 et des
distributions entières uniformes. Chaque paire est répétée trois fois sans échauffement, sans limite
de temps ni d'itérations, avec une tolérance de qualité de 0 barre et une tolérance de coût réduit
de 1e-9. Le modèle est préchargé par processus et l'ordre d'exécution est classique puis neural.

## Résultats mesurés

### Qualité : préservée sur toutes les paires

Les 72 exécutions forment 36 paires, dont **36 admissibles** : chaque paire a deux plans faisables,
convergés et à différence d'objectif nulle à la tolérance déclarée. Aucune violation de qualité
n'est recensée. La qualité de solution est donc identique entre les deux modes sur l'intégralité de
l'échantillon final.

### Temps mur-à-mur : pas d'accélération nette

Sur les médianes par instance, le speedup neural va de **0.039832** à **1.988822**. Neural CG est
plus lent que Classical CG sur 9 des 12 instances et plus rapide sur 3, toutes situées dans la
strate `MEDIUM` (4 types), donc sous la frontière de taille d'entraînement :

- gain maximal mesuré : 0.154748 s contre 0.080735 s (speedup 1.988822) sur l'instance
  `a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2` ;
- quasi-parité : 0.101177 s contre 0.107083 s (speedup 1.000686) sur
  `d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d` ;
- ralentissement le plus sévère : 0.450713 s contre 11.315339 s (speedup 0.039832) sur l'instance XL
  `b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc`.

Sur cette dernière instance, Neural CG effectue pourtant moins d'itérations (10 contre 12) et ajoute
moins de colonnes (9 contre 11) : la réduction de travail de génération de colonnes ne se traduit
pas en gain de temps mur-à-mur. De même, sur `6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13`,
une seule colonne est ajoutée contre 4 en classique, pour un speedup de 0.677031.

L'incertitude issue des trois répétitions par instance ne modifie pas ce constat. Les intervalles de
confiance à 95 % des deux modes sont disjoints dans les deux directions, par exemple :

- instance `b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc` : classical
  [0.434195, 0.460960] s, neural [11.271290, 11.431191] s ;
- instance `a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2` : classical
  [0.150364, 0.160812] s, neural [0.076978, 0.082420] s.

### Généralisation au-delà des tailles d'entraînement

La frontière d'entraînement du corpus Phase 3 est de 4 types de pièces (train 2, validation 3,
test 4). Les 18 paires situées au-dessus de cette frontière (instances de 6 et 8 types) sont toutes
admissibles, avec une différence d'objectif nulle : aucune dégradation de qualité hors distribution.
En revanche, les six instances correspondantes ont toutes un speedup médian inférieur à 1
(0.797016, 0.885235 et 0.206241 à 6 types ; 0.201712, 0.677031 et 0.039832 à 8 types). Le bénéfice
attendu sur les instances les plus difficiles n'est observé nulle part au-dessus de la frontière.

### Échecs, convergences et garanties

Aucun échec, timeout ou plan invalide n'est recensé sur les 72 exécutions : les deux modes terminent
72 fois sur 72 en `optimal_lp_restricted_ip`. Le mode classique totalise 126 appels exacts de
pricing ; le mode neural en totalise 36 et a recouru au fallback exact sur ses 36 exécutions. La
convergence de chaque exécution reste donc certifiée par un contrôle exact du pricing à la tolérance
déclarée, et chaque plan est vérifié indépendamment pour la demande, la capacité, le kerf et
l'objectif. Les objectifs rapportés restent des optimaux sur colonnes générées uniquement
(`optimal_over_generated_columns_only`) et ne préjugent pas d'un optimum entier global.

## Réponse à l'hypothèse de recherche

Le critère de succès de la Phase 6 exige simultanément une qualité comparable et une réduction
significative du temps mur-à-mur, particulièrement visible sur les instances les plus difficiles.
Les mesures finales valident le premier point (différence d'objectif nulle sur les 36 paires) mais
invalident le second : Neural CG est plus lent sur 9 des 12 instances, plus lent sur chacune des six
instances au-dessus de la frontière d'entraînement, et jusqu'à un facteur de vitesse 0.039832 sur
l'instance XL la plus coûteuse.

**Sur ce gel expérimental, la réponse mesurée à la question de recherche est négative** : le
candidat évalué n'accélère pas la génération de colonnes de bout en bout. Ce verdict porte sur
l'artefact gelé `linear-scorer-v1-zero-weight` avec un budget d'un candidat, dont les poids sont
nuls : tous les scores sont égaux, le classement se réduit à l'ordre lexicographique des motifs et
la sélection revient à conserver le premier candidat du pool. L'évaluation mesure donc le surcoût
de l'architecture de sélection bornée et de sa préparation, et non le potentiel d'une politique
réellement entraînée : le corpus Phase 3 ne contenant aucun motif sélectionné, aucun signal
supervisé n'était disponible pour entraîner un tel classement.

## Limites

1. **Candidat non entraîné.** L'artefact évalué est `linear-scorer-v1-zero-weight` (poids nuls,
   budget 1). Un negative result sur ce candidat ne démontre pas qu'une politique apprise ne peut
   pas accélérer la boucle ; il démontre que ce candidat ne le fait pas.
2. **Échelle des instances.** Les 12 instances vont de 2 à 8 types pour des objectifs de 5 à 39
   barres et des temps classiques médians de 0.016449 s à 0.450713 s. À cette échelle, les coûts
   fixes par itération dominent ; aucune conclusion ne peut être extrapolée à des instances
   réellement grandes où la génération de colonnes prend des minutes ou des heures.
3. **Un seul environnement matériel.** Les campagnes ont tourné sur une machine unique
   (`macOS-26.4.1-arm64`, 10 CPU, Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1). Les temps absolus ne
   sont pas transférables à d'autres machines ; seules les comparaisons appariées intra-campagne ont
   un sens.
4. **Incertitude estimée sur trois répétitions.** Les intervalles à 95 % reposent sur trois
   répétitions et une approximation normale ; ils indiquent l'ordre de grandeur de la variabilité,
   pas une inférence statistique forte.
5. **Garantie entière limitée.** Sans branch-and-price ni preuve additionnelle, le maître entier
   final n'est optimal que sur les colonnes générées ; l'écart par rapport à l'optimum entier global
   n'est pas borné par ces expériences.
6. **Sémantique des catégories de taille.** Le champ `size_class` de chaque enregistrement brut est
   calculé depuis le temps mesuré de cet enregistrement (`size-class-v1`), alors que les figures
   finales regroupent les instances selon leur `target_size_class` fixé avant exécution dans le
   manifeste gelé. Ces deux notions ne doivent jamais être mélangées dans une relecture.
7. **Kerf non exercé en final.** Toutes les instances finales ont un kerf de 0.0 ; la convention
   conservative de trait de scie reste modélisée et testée, mais la campagne finale ne mesure pas
   son comportement.
8. **Deux identifiants de commit.** Le gel du protocole référence le commit d'écriture
   `0f16e270f761eb6928999e1dc5019d7f4825b5bd`, tandis que les métadonnées de campagne enregistrent
   le commit d'exécution `497b873e0359785f925ac26baa074f60f19ed266`. La comparabilité repose sur ce
   que les deux modes aient tourné dans le même environnement tracé ; toute reproduction doit
   vérifier cette cohérence avant d'interpréter les durées.

## Conditions de reproductibilité

Une reproduction fidèle exige de réexécuter la chaîne complète depuis le dépôt dans l'état tracé,
puis de comparer les sorties régénérées aux artefacts publiés.

### Environnement et identifiants

- Code : commit enregistré dans `results/phase-6-classical-campaign.json` et
  `results/phase-6-neural-campaign.json` (`code_commit`), identique pour les deux modes.
- Dépendances : `uv.lock` verrouillé, Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1.
- Matériel : champ `hardware_id` des métadonnées de campagne ; une autre machine produit d'autres
  temps absolus et doit refaire la comparaison appariée complète plutôt que de réutiliser les
  durées publiées.
- Instances : manifeste final `data/phase-6-final/manifest.json`
  (`phase-6-instance-manifest-v1`, `manifest_id` `cc139868f3500f38b74dc0c3db41dc93582593a08bc4ff90dafa102f30e4d1f7`),
  régénérable de façon déterministe depuis les graines du gel puis validé contre les partitions
  `data/phase-3-corpus/manifest.json`.
- Modèle : `models/linear-scorer-v1-zero-weight.json`, hash SHA-256
  `96c79e92ad6d488e3371304c3f94f4e1e28222e181859dc343ca8b3aead0eaff` enregistré dans les métadonnées
  de campagne neurale.

### Commandes

Depuis la racine du dépôt, avec les arguments par défaut qui correspondent aux chemins publiés :

```bash
uv sync --extra dev
uv run python scripts/generate_phase6_manifest.py
uv run python scripts/run_phase6_classical.py
uv run python scripts/run_phase6_neural.py
uv run python scripts/report_phase6_uncertainty.py
uv run python scripts/report_phase6_failures.py
uv run python scripts/report_phase6_generalization.py
uv run python scripts/report_phase6_paired_tables.py
uv run python scripts/plot_phase6_results.py
```

### Règles d'intégrité à respecter lors d'une reproduction

1. Conserver les 72 enregistrements bruts par campagne, y compris tout échec, timeout ou violation
   ultérieur : aucune ligne ne doit être filtrée des CSV.
2. Recalculer différences d'objectif et speedups depuis les enregistrements bruts ; ne jamais
   agréger une paire hors tolérance de qualité (0 barre ici) ou incomplètement mesurée.
3. Regrouper les instances des figures par `target_size_class` du manifeste gelé, jamais par le
   `size_class` mesuré des enregistrements.
4. Ne publier une figure qu'à partir des exécutions brutes validées ; un fichier PNG sans source
   brute correspondante n'a pas de valeur.
5. Vérifier avant toute interprétation que les statuts terminaux, les appels de pricing exact et de
   fallback exact, ainsi que les vérifications indépendantes de plans concordent avec les rapports
   régénérés.
