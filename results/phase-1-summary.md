# Bilan de validation de la Phase 1

Date d'exécution : 2026-08-11  
Commit mesuré : `713e5acedb5b83348a2bfa67195043f0469b97d5`  
Environnement : Python 3.11, `pytest 8.4.2`, `ruff 0.16.2`, dépendances verrouillées par `uv.lock`.

## Commandes exécutées

```text
uv run pytest -q
68 passed in 1.70s

uv run ruff check .
All checks passed!
```

## Critères vérifiés

- Les 68 tests de la baseline classique passent.
- Le pricing entier exact est comparé par énumération exhaustive sur trois petites instances,
  avec kerf nul, kerf positif et longueurs décimales.
- La relaxation finale de Column Generation atteint la même valeur que le maître complet sur
  le corpus exhaustif de trois instances testé.
- Le maître entier restreint est résolu sur les colonnes générées et sa garantie est limitée à
  `optimal_over_generated_columns_only`.
- Les plans entiers sont vérifiés indépendamment pour la capacité, la couverture, le kerf, le
  bilan matière et le nombre de barres.
- Les statuts de convergence, d'infaisabilité, de limite atteinte, d'erreur solveur et de plan
  invalide sont couverts par les tests dédiés.
- Le contrôle Ruff ne signale aucun problème.

## Conclusion

Les critères de sortie de la Phase 1 sont satisfaits par la validation locale exécutée ci-dessus.
Ce bilan ne contient aucun résultat de performance : aucun benchmark de temps ou de speedup n'est
publié à ce stade.
