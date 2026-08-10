# Formulation mathématique initiale

## 1. Instance

Une instance comporte :

- une longueur de barre `L > 0` ;
- une largeur de trait de scie `k >= 0` ;
- `m` types de pièces de longueurs `l_i > 0` ;
- des demandes entières `d_i > 0`.

La représentation normalisée regroupera les types de même longueur et les ordonnera de façon déterministe. Chaque type demandé doit tenir seul dans une barre selon la convention de kerf :

```text
l_i + k <= L
```

Les longueurs et le kerf utilisent la même unité, typiquement le millimètre. Les valeurs réelles validées sont converties depuis leur représentation décimale (`str(value)`) pour les calculs de capacité et de division entière ; cela évite qu’un arrondi binaire fasse perdre une pièce exactement admissible. Les solveurs conservent leurs tolérances explicites pour les comparaisons numériques.

## 2. Convention de kerf

Pour un motif `a = (a_1, ..., a_m)`, avec `a_i` le nombre de pièces de type `i`, la consommation de capacité est :

```text
capacity_used(a) = sum_i l_i a_i + k sum_i a_i
```

Le motif est faisable si :

```text
capacity_used(a) <= L
```

Cette convention réserve un kerf par pièce, y compris après la dernière pièce. Elle est légèrement conservative par rapport à certains procédés où le dernier bord ne nécessite pas de coupe. Elle a trois avantages initiaux : formulation linéaire simple, pricing de type sac à dos inchangé et absence d’ambiguïté entre solveur et vérificateur.

`k = 0` est pleinement supporté. Une convention physique plus précise pourra être introduite ultérieurement derrière une version explicite de la formulation, sans changer la question de recherche.

## 3. Motifs faisables

L’ensemble théorique des motifs non vides est :

```text
P = {a in Z_+^m : 1 <= sum_i a_i, a_i <= d_i,
                    sum_i (l_i + k) a_i <= L}
```

La borne `a_i <= d_i` ne retire aucun plan optimal utile : produire dans une seule barre plus que la demande totale d’un type ne peut améliorer la couverture. Elle borne également le pricing.

## 4. Maître entier complet

Pour chaque motif `p in P`, la variable entière `x_p` est le nombre de barres coupées selon ce motif.

```text
minimize    sum_p x_p

subject to  sum_p a_ip x_p >= d_i       for every type i
            x_p in Z_+                   for every pattern p
```

Il s’agit d’une formulation de couverture : une légère surproduction est autorisée. L’objectif primaire est le nombre de barres. Si l’application interdit toute surproduction, cette exigence devra être ajoutée comme une décision de produit explicite et retestée ; elle ne doit pas être glissée silencieusement dans le solveur.

## 5. Restricted Master Problem

L’ensemble `P` étant trop grand pour être énuméré, la génération de colonnes maintient un sous-ensemble `P_R`.

Le RMP linéaire résout :

```text
minimize    sum_{p in P_R} x_p

subject to  sum_{p in P_R} a_ip x_p >= d_i   for every i
            x_p >= 0                         for every p in P_R
```

Les motifs initiaux prévus sont homogènes. Pour chaque type `i`, on ajoute le motif contenant :

```text
min(d_i, floor(L / (l_i + k)))
```

pièces de ce type et aucune autre. Les validations précédentes garantissent que cette quantité est au moins un et donc que le RMP initial couvre chaque type.

## 6. Dual et pricing exact

En notant `pi_i >= 0` la variable duale de la contrainte de demande `i`, le dual du RMP est :

```text
maximize    sum_i d_i pi_i

subject to  sum_i a_ip pi_i <= 1       for every p in P_R
            pi_i >= 0
```

Le coût réduit d’un nouveau motif `a` est :

```text
reduced_cost(a) = 1 - sum_i pi_i a_i
```

Le pricing cherche le motif de valeur duale maximale :

```text
maximize    sum_i pi_i a_i

subject to  sum_i (l_i + k) a_i <= L
            0 <= a_i <= d_i
            a_i integer
```

Une colonne améliore le RMP si sa valeur est strictement supérieure à `1 + epsilon`, soit un coût réduit inférieur à `-epsilon`. `epsilon` sera une configuration enregistrée, jamais une constante invisible.

Le premier pricing sera résolu exactement comme un problème de sac à dos entier avec un solveur open source HiGHS. De petites instances seront aussi vérifiées par énumération exhaustive dans les tests.

## 7. Convergence et solution entière

La boucle s’arrête lorsque le pricing exact ne trouve aucun motif de coût réduit négatif à la tolérance choisie. Cela certifie l’optimalité de la relaxation linéaire du maître complet.

Le solveur résout ensuite le maître **entier restreint** sur `P_R`. Ce plan est faisable pour l’instance originale, mais il n’est pas nécessairement l’optimum entier parmi tous les motifs de `P`. Une preuve d’optimalité entière globale requerrait par exemple du branch-and-price, qui n’est pas nécessaire au premier jalon et ne doit pas être revendiqué implicitement.

Les résultats distingueront au minimum :

- la borne LP à convergence ;
- l’objectif entier restreint ;
- le gap entre ces deux valeurs ;
- le statut du RMP, du pricing et du maître entier ;
- la raison de terminaison.

## 8. Bilan matière

Pour un plan entier, soit `B = sum_p x_p` le nombre de barres.

```text
requested_length       = sum_i l_i d_i
produced_length        = sum_p x_p sum_i l_i a_ip
overproduction_length  = produced_length - requested_length
kerf_loss               = k sum_p x_p sum_i a_ip
trim_loss               = sum_p x_p (L - sum_i (l_i + k) a_ip)
total_waste             = B L - requested_length
```

Avec cette convention de couverture :

```text
total_waste = overproduction_length + kerf_loss + trim_loss
```

`total_waste` mesure donc toute matière achetée qui ne correspond pas à la demande : chute, trait de scie et éventuelle surproduction. Les quatre composantes seront préférées à un unique nombre ambigu dans les rapports détaillés.

## 9. Invariants du vérificateur

Indépendamment des statuts du solveur, tout plan accepté doit vérifier :

- variables de motif entières et non négatives ;
- chaque motif respecte la capacité avec kerf ;
- la couverture produite est supérieure ou égale à la demande ;
- le nombre de barres correspond à la somme des multiplicités ;
- les identités de bilan matière sont satisfaites à la tolérance numérique ;
- chaque colonne déclarée nouvelle est non vide et dédupliquée.
