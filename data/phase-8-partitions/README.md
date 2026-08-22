# Manifeste des partitions qualité Phase 8

`manifest.json` gèle les partitions entraînement/validation/test des familles retenues par la
mesure de marges (`family-margins-v1`, voir [`results/phase-8-family-margins.md`](../../results/phase-8-family-margins.md)).
Il est généré puis validé avec :

```bash
uv run python scripts/freeze_phase8_partitions.py
```

L'unité de partitionnement est la cellule mesurée `(family_label, seed)` ; la graine seule décide
de la partition (`train` : 1–3, `validation` : 4, `test` : 5–6), si bien qu'aucune graine — donc
aucun tirage aléatoire — n'est partagée entre deux partitions. Chaque cellule est recroisée à son
générateur déclaré lors de la construction, chaque `instance_id` matérialisé doit être unique dans
tout le plan, et la validation ré-materialise chaque cellule depuis la configuration enregistrée,
indépendamment des specs vivantes. Toute altération du manifeste persisté est détectée par les
tests (`tests/test_phase8_partitions.py`).
