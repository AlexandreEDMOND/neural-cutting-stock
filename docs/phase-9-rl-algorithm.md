# Algorithme d'entraînement RL de l'agent qualité (Phase 9)

Cas ROADMAP P9.05 : entraîner une **politique profonde** sur les familles à marge de la phase 8,
avec journal complet des expériences. Ce document nomme l'algorithme, le justifie face aux
alternatives et fixe les garanties honnêtes associées. L'implémentation se trouve dans
[`src/neural_cutting_stock/learning/rl_policy.py`](../src/neural_cutting_stock/learning/rl_policy.py)
et le point d'entrée dans [`scripts/train_phase9_policy.py`](../scripts/train_phase9_policy.py).

## Identifiant et résumé

- Identifiant : `reinforce-poisson-completion-v1`.
- Famille : gradient de politique Monte-Carlo — REINFORCE avec baseline et avantages standardisés.
- Entraînement : épisodes `quality-refinement-env-v1` construits sur les points de départ
  classiques vérifiés des instances de la partition d'entraînement gelée
  (`data/phase-8-partitions/manifest.json`, familles retenues de la phase 8).

## Formulation MDP

Un épisode est le raffinement itératif d'une solution entière classique, exactement l'environnement
versionné de P9.02 :

| Élément | Définition |
|---|---|
| État | `QualityAgentInput` (`quality-agent-interface-v1`) : vue d'instance, pool de colonnes constant, solution courante |
| Action | un plan complet : un compte d'usage par motif maximal énuméré pour l'instance |
| Récompense | réduction signée de barres **vérifiée** par revue indépendante ; pénalité stricte pour plan invalide |
| Horizon | budget déclaré `max_steps`, seule terminaison honnête |

L'espace d'action est factorisé : la base de motifs maximaux `B` est énumérée une fois par instance
(déterministe, bornée par `MaximalPatternLimits`), et l'agent choisit un vecteur d'usages
`u ∈ ℕ^{|B|}`. Le réseau (`tanh` 28 → caché → 1) émet un taux strictement positif
`μ_c = softplus(score_c) + ε` par candidat et l'action est échantillonnée composante par composante :

```text
u_c ~ Poisson(μ_c),   u_c exécuté = min(u_c, min_{i: p_ci > 0} demand_i)
```

Le plafonnement par les bornes de demande garantit que chaque motif exécuté reste vérifiable ;
la log-probabilité est évaluée sur les comptes exécutés (troncature inactive pour des taux sains).

### Complétion déterministe de couverture

Les comptes échantillonnés sont décodés en plan puis complétés depuis la même base : une passe
en masse ajoute des multiplicités entières tant qu'elles réduisent la demande résiduelle, puis des
passes unitaires épuisent tout résidu (la surproduction est légale, seule la sous-production est
rejetée). Toute proposition exécutée vérifie donc la couverture : chaque pas de formation vit dans
le régime gradué de récompenses vérifiées au lieu du plateau plat de pénalité. Cette complétion est
une partie déclarée de la politique, pas un artifice d'évaluation ; les ablations prévues (P9.09)
l'utilisent à budget égal.

## Estimation du gradient

Pour un épisode de récompenses `r_1..r_T` :

```text
G_t = Σ_{s ≥ t} r_s                          (reward-to-go, crédite chaque action de ses gains futurs)
b_i  = moyenne mobile exponentielle du retour de l'épisode i, suivie PAR INSTANCE
A_t  = G_t − b_i                             (baseline indépendante de l'action : gradient non biaisé)
A'_t = standardisation (moyenne 0, écart 1) des A_t sur toutes les étapes d'une époque
L(θ) = − Σ_t A'_t · log π_θ(a_t | s_t)       (un seul pas Adam par époque)
```

Justifications ponctuelles : le reward-to-go exploite la causalité (la récompense d'un pas ne dépend
que de sa propre proposition) ; la baseline par instance absorbe les écarts d'échelle entre familles
(t3 : ~30 barres, t12 : ~400 barres) ; la standardisation par époque neutralise le bruit d'échelle
résiduel sans biaiser la direction du gradient ; `ε = 1e-8` rend la standardisation sûre quand les
avantages sont tous égaux.

## Pourquoi cet algorithme

- **Épisodes courts, simulateur exact gratuit** : chaque transition passe par `verify_proposal`,
  calcul pur et déterministe. Un algorithme Monte-Carlo on-policy y est naturellement efficace ;
  aucune contrainte mur-à-mur n'existe (constitution, invariant 4), donc la simplicité prime.
- **Action dénombrable vectorielle** : DQN/acteurs-critiques discrets visent des actions indexées ;
  ici l'action est un plan entier à `|B|` dimensions (6 à 78 candidats). La factorisation de
  Poisson donne une loi simple, à support entier, dont le score-function estimator est exact.
- **PPO/TRPO** apportent leur bénéfice principal sur de longs horizons corrélés avec de gros
  réseaux ; ni l'un ni l'autre n'est présent ici. La complexité supplémentaire ne serait pas
  justifiée par un profilage (règle du dépôt : mesurer avant d'optimiser).
- **Continuité avec P9.04** : mêmes features `imitation-candidate` à largeur fixe 28, même base
  d'action énumérée ; la politique RL généralise le clonage d'imitation en exploration stochastique.

## Garanties et honnêteté

- L'agent propose, le vérificateur dispose : toute proposition traverse `verify_proposal` ;
  l'acceptation exige une réduction stricte de barres vérifiée. Aucune optimalité globale n'est
  jamais revendiquée, ni pendant l'entraînement ni après.
- Les contraintes de capacité et de demande ne dépendent jamais d'une prédiction neuronale : la
  complétion puise uniquement dans les motifs maximaux valides de la base.
- Les échecs restent visibles : les plans invalides seraient pénalisés et journalisés (aucun n'est
  produit en pratique grâce à la complétion ; l'invariant est testé).
- Reproductibilité : graine unique via `set_reproducible_seed`, ordre trié des instances, mise à
  jour groupée déterministe ; checkpoint `neural-training-checkpoint-v1` et journal
  `phase-9-training-journal-v1` (courbes complètes, un enregistrement par épisode, environnement
  tracé). L'égalité bit-exact entre machines reste hors périmètre, comme en P9.03.
- Les durées d'entraînement sont journalisées à titre informatif et ne constituent jamais un
  critère de succès.

## Journal des expériences

Chaque exécution publie via `training_journal_payload` :

- la source : manifeste de partitions gelées, `plan_id`, partition, identifiants d'instances,
  chemin et SHA-256 du checkpoint ;
- l'algorithme : identifiant, description de l'espace d'action, de la loi d'échantillonnage, de la
  complétion, de la récompense, de la mise à jour, de la baseline et de l'optimiseur ;
- la configuration complète (graine, hyperparamètres) et les métadonnées d'environnement ;
- les courbes complètes (`training-curves-v1`) : perte de politique, retour moyen par épisode,
  barres gagnées, pas acceptés et invalides, par époque ;
- un enregistrement par épisode : instance, pas, retour, barres gagnées, compteurs d'acceptation
  et d'invalidité, barres initiales et finales.

Le journal publié du run de référence est
[`results/phase-9-training-journal.json`](../results/phase-9-training-journal.json) et le
checkpoint correspondant [`models/phase-9-quality-policy.pt`](../models/phase-9-quality-policy.pt).
