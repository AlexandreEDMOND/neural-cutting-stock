# Manifeste final Phase 6

`manifest.json` est le manifeste déterministe des instances réservées à l'évaluation finale.
Il est généré avec :

```bash
uv run python scripts/generate_phase6_manifest.py
```

Les entrées sont hors du corpus Phase 3, conservent les longueurs et demandes matérialisées, et
sont vérifiées contre leur générateur avant écriture. `target_size_class` désigne une strate de
charge fixée avant l'évaluation (`size-class-v1`) ; la catégorie mesurée par le runtime classique
sera enregistrée plus tard dans `size_class`. Ces deux notions ne doivent pas être confondues.
