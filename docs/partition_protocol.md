# Protocole de partitions

P3.09 fixe les partitions avant toute collecte de trajectoire avec `PartitionPlan` et le schéma
`trajectory-partitions-v1`. Une famille est définie par la configuration du générateur, hors graine,
et reçoit un `family_id` stable ; deux graines différentes d'une même famille partagent donc cet
identifiant.

Chaque partition contient des ensembles disjoints de graines et de familles pour `train`,
`validation` et `test`. Une instance est admissible seulement si sa graine et sa famille désignent
la même partition. Une combinaison inconnue ou croisée est rejetée, ce qui rend explicite tout
risque de fuite au lieu de l'assigner arbitrairement.

Le plan est une configuration de campagne indépendante des trajectoires et se sérialise avec
`PartitionPlan.to_dict()`. L'ordre des générateurs n'intervient pas dans l'identité d'une
assignation ; la collecte et la construction du dataset sont des étapes ultérieures.
