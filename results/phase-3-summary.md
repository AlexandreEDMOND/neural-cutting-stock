# Bilan de publication de la Phase 3

Source unique : [`manifest.json`](../data/phase-3-corpus/manifest.json), corpus `phase-3-small-v1`, schéma `phase-3-corpus-v1`.

## Corpus validé

- Trajectoires persistées : **3** ; instances : **3**.
- Statuts : `converged`: 3.
- Répartition : train **1**, validation **1**, test **1**.
- Itérations enregistrées : **3** ; colonnes ajoutées : **0** ; motifs sélectionnés : **0**.
- Validation : chaque hash SHA-256 du manifeste correspond au JSON persistant et chaque trajectoire a été rejouée par le solveur classique exact sans erreur.

| Partition | Trajectoires | Itérations | Types de pièces |
|---|---:|---:|---:|
| train | 1 | 1 | 2 |
| validation | 1 | 1 | 3 |
| test | 1 | 1 | 4 |

## Portée et limites

Le corpus est un petit corpus de collecte, pas un benchmark de temps : les durées de collecte ne sont pas agrégées ni interprétées comme une performance. Il ne contient aucune colonne candidate enregistrée, aucune colonne ajoutée et aucun exemple de dataset (`selected_pattern_count = 0`). Il ne permet donc pas d'évaluer un modèle appris, un classement de colonnes ou un speedup.

L'environnement déclaré est `3.11.15`, `numpy,scipy`, `Darwin-arm64` ; le commit déclaré par les trajectoires est `ce3d85831e2802374a6bf18f762015dd9d49493f`. La régénération et le rejeu sont documentés dans [`data/phase-3-corpus/README.md`](../data/phase-3-corpus/README.md).

## Figures

- [`phase3_corpus_structure.png`](phase3_corpus_structure.png) représente les comptes d'itérations et de types de pièces par partition.
- [`phase3_instance_dimensions.png`](phase3_instance_dimensions.png) représente les dimensions des trois instances persistées.

Les deux figures sont descriptives et proviennent exclusivement du manifeste et des trajectoires validées ; elles ne montrent ni runtime, ni comparaison Classical CG/Neural CG, ni résultat scientifique extrapolé.
