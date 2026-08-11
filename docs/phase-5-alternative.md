# Alternative retenue pour P5.04

## Décision

La politique séquentielle n'est pas retenue. L'alternative conservée est la
sélection supervisée bornée de la Phase 4, spécifiée ici sous le contrat
`bounded-column-selection-v1`. Elle intervient uniquement entre le pricing
classique et l'ajout de colonnes au RMP.

Cette formalisation décrit l'interface opérationnelle de l'alternative. Elle ne
définit pas encore une récompense ni une nouvelle métrique d'entraînement ; ce
point est réservé à P5.05.

## État

À chaque appel de sélection, l'observation est un `PricingState` de
`learning-interface-v1`, identifié par l'instance et l'itération courante. Il
contient notamment :

- les longueurs, demandes et kerf de l'instance 1D ;
- les duales du RMP dans l'ordre des types ;
- les motifs déjà présents dans le RMP ;
- la tolérance de coût réduit et la convention de signe des duales.

L'observation de chaque action candidate est un `PatternCandidate` produit par
le pool déterministe classique. Il contient un motif faisable, nouveau pour le
RMP, et son coût réduit calculé par le pricing classique. La politique ne
génère donc ni motif ni valeur duale.

## Action

L'action est une décision `ColumnSelectionDecision` : classer les candidats et
en retenir au plus `candidate_budget` motifs. Le classement est déterministe à
score égal et le budget est explicite.

Avant l'ajout au RMP, l'orchestration conserve uniquement les motifs retenus
dont le coût réduit est strictement inférieur à `-reduced_cost_tolerance`.
Une sélection vide déclenche le fallback exact. La politique ne déclare jamais
la convergence.

## Horizon

L'horizon de la politique est **un appel de sélection** (`H = 1`) : elle prend
une observation du pricing et produit une décision sans mémoire de ses choix
précédents. Cette décision locale est répétée par la boucle de génération de
colonnes, qui reconstruit l'état après chaque résolution du RMP.

La fin de la génération de colonnes n'est pas une décision de la politique.
Seul le pricing exact, exécuté par le solveur classique après la sélection,
peut établir l'absence de colonne améliorante. Le maître entier restreint et la
vérification indépendante du plan restent inchangés.

## Portée et garde-fous

Cette alternative conserve le périmètre et les garanties de la Phase 4 :

- les contraintes de capacité et de demande restent indépendantes du score ;
- les candidats viennent du pricing classique et sont vérifiés avant insertion ;
- le pricing exact et le fallback exact restent obligatoires avant convergence ;
- les budgets de candidats et les limites de ressources restent explicites ;
- toute comparaison est end-to-end et vérifie la qualité avant d'agréger le temps.

Une politique avec mémoire, une action de génération de motif ou un horizon
supérieur à un appel constitueraient une autre conception et ne sont pas
introduits par P5.04.
