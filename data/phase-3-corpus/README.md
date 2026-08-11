# Petit corpus de Phase 3

Ce corpus contient trois trajectoires classiques rejouables, une par partition (`train`,
`validation`, `test`). `manifest.json` fixe les graines, familles, instances, hashes, schémas,
environnement et statistiques calculées depuis les trajectoires persistées.

Régénération depuis la racine du dépôt :

```bash
uv run python scripts/build_phase3_corpus.py --output-dir data/phase-3-corpus
```

Les temps présents dans les trajectoires sont des mesures de collecte d'une exécution particulière;
ils ne constituent pas un benchmark de performance. Le rejeu exact est la validation du corpus.
