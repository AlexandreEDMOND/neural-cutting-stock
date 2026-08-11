# Neural Cutting Stock — Learning to Accelerate Column Generation

> Projet de recherche focalisé sur une seule question : un modèle appris peut-il accélérer significativement la génération de colonnes pour le Cutting Stock 1D, tout en préservant la qualité de la méthode classique ?

## État du projet

La Phase 1 est clôturée. La baseline classique de génération de colonnes comprend la validation des instances et du kerf, le RMP linéaire, le pricing entier exact, la boucle de génération de colonnes, le maître entier restreint, la vérification indépendante et la CLI structurée. Aucun composant neuronal ni résultat de performance Neural CG n’existe encore.

## Motivation

Le Cutting Stock 1D apparaît notamment lors de la découpe de barres d’aluminium. Une commande regroupe des pièces de plusieurs longueurs et quantités, à produire à partir de barres de longueur fixe. Un plan de coupe doit satisfaire toute la demande avec le moins de barres possible.

Les formulations compactes deviennent difficiles lorsque le nombre de motifs de coupe possibles explose. La génération de colonnes évite de tous les énumérer : elle ne construit que les motifs utiles. Sur les grandes instances, les résolutions successives du problème maître, du pricing et la gestion des colonnes peuvent néanmoins devenir coûteuses.

## Hypothèse de recherche

Le réseau neuronal ne remplacera pas le solveur. Il apprendra à classer ou sélectionner les colonnes candidates les plus utiles au sein de la boucle de génération de colonnes.

Le projet sera considéré comme concluant uniquement si les expériences mesurées montrent simultanément :

- un objectif final identique ou explicitement comparable à celui du solveur classique ;
- un temps mur-à-mur inférieur, et non seulement une inférence rapide ;
- un gain qui se maintient ou augmente sur les instances les plus difficiles ;
- aucune fausse convergence grâce à une vérification exacte du pricing.

## Problème étudié

Une instance contient :

```python
stock_length: float
kerf: float
piece_lengths: list[float]
demands: list[int]
```

Un motif `a = (a_1, ..., a_m)` indique combien de pièces de chaque type sont découpées dans une barre. La convention initiale de trait de scie est volontairement conservative :

```text
sum(piece_length[i] * a[i]) + kerf * sum(a[i]) <= stock_length
```

Une largeur de coupe est donc réservée par pièce, y compris pour la dernière pièce de la barre. `kerf = 0` retrouve la formulation mathématique habituelle. Cette convention est une hypothèse de modélisation, pas un axe de recherche.

L’objectif maître est de minimiser le nombre de barres sous des contraintes de couverture de la demande. La formulation complète, le dual, le pricing et la définition de la perte matière sont détaillés dans [docs/formulation.md](docs/formulation.md).

## Approche classique

Le socle est une génération de colonnes pour le Cutting Stock 1D :

1. construire des motifs initiaux garantissant la faisabilité ;
2. résoudre la relaxation linéaire du Restricted Master Problem (RMP) ;
3. extraire les valeurs duales ;
4. résoudre exactement le pricing sous forme de sac à dos entier ;
5. ajouter tout motif de coût réduit négatif et recommencer ;
6. lorsque le pricing exact ne trouve plus d’amélioration, résoudre le RMP entier sur les motifs générés.

Le pricing exact certifie, à la tolérance numérique déclarée, l’optimalité de la relaxation linéaire du maître complet lorsque aucune colonne améliorante n’est trouvée. Le maître entier final est résolu uniquement sur les colonnes générées : son statut est donc `optimal_over_generated_columns_only`, et ne constitue pas une preuve d’optimalité entière globale sans branch-and-price ou preuve additionnelle. Les contraintes de capacité et de demande sont vérifiées indépendamment du solveur.

## Accélération apprise proposée

À partir de la Phase 4, un modèle simple recevra l’état courant du RMP, les duales et un ensemble de motifs candidats. Il classera les colonnes à ajouter. Une passe de pricing exacte restera obligatoire avant toute déclaration de convergence.

```mermaid
flowchart TD
    A[Instance Cutting Stock 1D] --> B[Motifs initiaux]
    B --> C[Restricted Master LP]
    C --> D[Valeurs duales]
    D --> E[Pricing / pool de candidats]
    E --> F{Mode du solveur}
    F -->|Classique| G[Sélection déterministe]
    F -->|Neural| H[Politique apprise de colonnes]
    G --> I[Ajout au RMP]
    H --> I
    I --> C
    C --> J[Contrôle exact du pricing]
    J -->|Colonne améliorante| I
    J -->|Aucune| K[Integer Restricted Master]
    K --> L[Plan de coupe et métriques]
```

Le même cœur classique devra pouvoir exécuter, sur une même instance, les interfaces prévues :

```bash
uv run python -m neural_cutting_stock ... --solver classical
uv run python -m neural_cutting_stock ... --solver neural
```

Ces commandes seront ajoutées avec les solveurs ; elles ne sont pas encore disponibles dans la phase de fondation.

## Méthodologie de benchmark

Les instances synthétiques seront reproductibles par graine explicite. La difficulté sera étudiée selon plusieurs dimensions indépendantes : nombre de types, demande totale, longueur de barre, distributions des longueurs et demandes, et kerf. Les catégories `SMALL`, `MEDIUM`, `LARGE` et `XL` ne seront figées qu’après profilage du solveur classique.

Les exécutions classique et neuronale seront appariées sur exactement les mêmes instances, configurations et conditions matérielles. Le temps principal est le temps mur-à-mur de résolution. Les décompositions RMP, pricing, maître entier, gestion des colonnes et, plus tard, inférence neuronale servent à expliquer ce temps, jamais à le remplacer.

Le schéma de données, les règles de chronométrage, les statuts et le protocole de comparaison sont spécifiés dans [docs/benchmark_protocol.md](docs/benchmark_protocol.md).

## Résultats

La validation de Phase 1 comporte **68 tests réussis** et un contrôle Ruff réussi, exécutés avec les dépendances verrouillées. Le bilan reproductible, les commandes, l’environnement et les critères vérifiés sont publiés dans [`results/phase-1-summary.md`](results/phase-1-summary.md). Ce nombre décrit la correction testée du code ; il ne constitue pas un résultat de performance.

**Les expériences de performance sont encore en attente.** Aucun speedup ni graphique comparatif n’est publié à ce stade.

Le pipeline de visualisation produira à partir des mesures brutes validées :

- `results/runtime_comparison.png` — temps mur-à-mur de Classical CG et Neural CG par difficulté ;
- `results/speedup_by_size.png` — accélération appariée par catégorie de taille.

Les courbes de runtime ne seront présentées comme succès que pour les paires dont la qualité de solution respecte le seuil déclaré, par défaut une différence de zéro barre.

<!-- À activer uniquement lorsque le fichier provient de mesures réelles :
![Runtime comparison](results/runtime_comparison.png)
-->

## Organisation du dépôt

```text
neural-cutting-stock/
├── AGENTS.md                       # constitution et garde-fous
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── configs/                        # configurations versionnées des expériences
├── docs/
│   ├── benchmark_protocol.md
│   └── formulation.md
├── results/                        # résultats et figures réellement mesurés
├── scripts/                        # points d’entrée fins, sans logique métier
├── src/neural_cutting_stock/
│   ├── problem/                    # modèle et validation d’instance
│   ├── solver/                     # RMP, pricing et orchestration CG
│   ├── benchmarks/                 # génération, exécution et enregistrement
│   └── visualization/              # figures issues des résultats validés
└── tests/
```

Le paquet `learning` ne sera créé qu’au début de la Phase 4, lorsque les profils et le format de trajectoire justifieront son interface.

## Installation de développement

Python 3.11 ou plus récent est requis.

```bash
uv sync --extra dev
uv run pytest
```

Le fichier `uv.lock` fixe l’environnement reproductible. Une installation editable avec `python -m pip install -e ".[dev]"` reste possible sans `uv`.

Le socle de résolution prévu repose sur NumPy et SciPy/HiGHS, sans licence commerciale. PyTorch ne deviendra une dépendance que lorsque le travail d’apprentissage commencera.

## Feuille de route

La progression est pilotée par les cases atomiques de [ROADMAP.md](ROADMAP.md) : une case correspond à une itération et un commit. Les six phases contiennent chacune quinze étapes initiales et peuvent recevoir des sous-étapes non cochées si le travail révèle un besoin réel. La prochaine itération est toujours la première case non cochée de la roadmap ; son identifiant n’est pas recopié ici afin d’éviter toute information obsolète.

## Périmètre

Ce dépôt traite exclusivement du Cutting Stock 1D accéléré par apprentissage au sein d’une génération de colonnes. Les solveurs end-to-end neuronaux, Pointer Networks, métaheuristiques, Cutting Stock 2D/3D et autres problèmes combinatoires ne font pas partie du projet.
