# Protocole du dataset de trajectoires

P3.10 produit `trajectory-dataset-v1` avec `build_dataset`. Chaque source est rejouée par le
pricing classique exact avant la construction du dataset ; une erreur de rejeu arrête la
construction et aucun dataset partiel n'est retourné.

Un exemple correspond à un motif candidat enregistré à une itération. Il contient les duales dans
l'ordre déclaré par la trajectoire, le motif, son coût réduit, la décision `selected` observée et
la partition explicitement fournie par l'appelant. Une itération sans pool candidat ne produit pas
d'exemple, et aucune valeur absente n'est remplacée artificiellement.

Les identifiants de trajectoire sont uniques et triés avant matérialisation afin que la sérialisation
soit indépendante de l'ordre d'entrée. Les partitions doivent couvrir exactement les trajectoires
fournies ; elles ne sont pas déduites des données de l'instance. Une même `instance_id` ne peut pas
apparaître dans plusieurs partitions, afin d'interdire une fuite entre train, validation et test.
