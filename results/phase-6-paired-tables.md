# Tableaux appariés de la Phase 6

Sources brutes validées : `results/phase-6-classical-runs.csv` et `results/phase-6-neural-runs.csv` (schéma `benchmark-run-v1`).

## Couverture

- Exécutions : **72** ; paires : **36** ; paires admissibles : **36**.
- Tolérance de différence d'objectif : **0.0 barre(s)**. Les différences d'objectif et les speedups sont recalculés depuis les enregistrements bruts.
- Les médianes par instance n'agrègent que les répétitions admissibles ; chaque paire, y compris échec ou violation, reste conservée dans [`phase-6-paired-tables.json`](phase-6-paired-tables.json).

## Qualité (barres)

| Instance | Types | Répétitions | Admissibles | Objectif Classical médian | Objectif Neural médian | Différence médiane | Violations qualité |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2e8d6ecbaa342a376ff4bf4c2261c4dc9641091f94c4aebbb8364b9e2584f6af | 2 | 3 | 3 | 7 | 7 | 0 | 0 |
| 3eb804881591494848f167ba7b26a2b9267a3ba54e1727e6932ff3b14cd58e50 | 2 | 3 | 3 | 12 | 12 | 0 | 0 |
| e89f36850422c2316a18c0e5c220f81c1d6ab1fbeb7c500622d6d755ac5409b2 | 2 | 3 | 3 | 5 | 5 | 0 | 0 |
| a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2 | 4 | 3 | 3 | 15 | 15 | 0 | 0 |
| b09c9bbfd876bc77c5440191ebfb1a02f5aa41b774b446ae71d1f3ca862891e9 | 4 | 3 | 3 | 14 | 14 | 0 | 0 |
| d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d | 4 | 3 | 3 | 13 | 13 | 0 | 0 |
| 2db8f04b0672b4e39578ab96c31e1989f5e93a806b484a143f83761c7a1f4c50 | 6 | 3 | 3 | 23 | 23 | 0 | 0 |
| 35714b99e7af814094a2bd53d0a82a4c33fb1c3bcdedddf861740c744140d125 | 6 | 3 | 3 | 30 | 30 | 0 | 0 |
| d71a500910a2039d84f38704af75d2036160060c50841a11b3d98be280d6f7fd | 6 | 3 | 3 | 20 | 20 | 0 | 0 |
| 5d7cde942f6f9a9811561fb1c16bb97f0f197f7fd567dcb7f41c33103c0406a0 | 8 | 3 | 3 | 23 | 23 | 0 | 0 |
| 6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13 | 8 | 3 | 3 | 39 | 39 | 0 | 0 |
| b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc | 8 | 3 | 3 | 24 | 24 | 0 | 0 |

## Runtime mur-à-mur (s)

| Instance | Types | Admissibles | Classical médian (s) | Neural médian (s) | Speedup médian |
|---|---:|---:|---:|---:|---:|
| 2e8d6ecbaa342a376ff4bf4c2261c4dc9641091f94c4aebbb8364b9e2584f6af | 2 | 3 | 0.017684 | 0.018910 | 0.983286 |
| 3eb804881591494848f167ba7b26a2b9267a3ba54e1727e6932ff3b14cd58e50 | 2 | 3 | 0.017027 | 0.019756 | 0.856655 |
| e89f36850422c2316a18c0e5c220f81c1d6ab1fbeb7c500622d6d755ac5409b2 | 2 | 3 | 0.016449 | 0.019093 | 0.892169 |
| a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2 | 4 | 3 | 0.154748 | 0.080735 | 1.988822 |
| b09c9bbfd876bc77c5440191ebfb1a02f5aa41b774b446ae71d1f3ca862891e9 | 4 | 3 | 0.127440 | 0.107068 | 1.203069 |
| d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d | 4 | 3 | 0.101177 | 0.107083 | 1.000686 |
| 2db8f04b0672b4e39578ab96c31e1989f5e93a806b484a143f83761c7a1f4c50 | 6 | 3 | 0.054742 | 0.069985 | 0.797016 |
| 35714b99e7af814094a2bd53d0a82a4c33fb1c3bcdedddf861740c744140d125 | 6 | 3 | 0.108184 | 0.125215 | 0.885235 |
| d71a500910a2039d84f38704af75d2036160060c50841a11b3d98be280d6f7fd | 6 | 3 | 0.218294 | 1.064525 | 0.206241 |
| 5d7cde942f6f9a9811561fb1c16bb97f0f197f7fd567dcb7f41c33103c0406a0 | 8 | 3 | 0.133188 | 0.660288 | 0.201712 |
| 6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13 | 8 | 3 | 0.214556 | 0.316907 | 0.677031 |
| b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc | 8 | 3 | 0.450713 | 11.315339 | 0.039832 |

## Mémoire (pic tracemalloc, octets)

| Instance | Types | Admissibles | Pic Classical médian | Pic Neural médian |
|---|---:|---:|---:|---:|
| 2e8d6ecbaa342a376ff4bf4c2261c4dc9641091f94c4aebbb8364b9e2584f6af | 2 | 3 | 15176 | 16381 |
| 3eb804881591494848f167ba7b26a2b9267a3ba54e1727e6932ff3b14cd58e50 | 2 | 3 | 14893 | 15516 |
| e89f36850422c2316a18c0e5c220f81c1d6ab1fbeb7c500622d6d755ac5409b2 | 2 | 3 | 14107 | 14891 |
| a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2 | 4 | 3 | 19457 | 18668 |
| b09c9bbfd876bc77c5440191ebfb1a02f5aa41b774b446ae71d1f3ca862891e9 | 4 | 3 | 17987 | 18746 |
| d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d | 4 | 3 | 17730 | 18083 |
| 2db8f04b0672b4e39578ab96c31e1989f5e93a806b484a143f83761c7a1f4c50 | 6 | 3 | 16637 | 17929 |
| 35714b99e7af814094a2bd53d0a82a4c33fb1c3bcdedddf861740c744140d125 | 6 | 3 | 18818 | 19519 |
| d71a500910a2039d84f38704af75d2036160060c50841a11b3d98be280d6f7fd | 6 | 3 | 23890 | 22797 |
| 5d7cde942f6f9a9811561fb1c16bb97f0f197f7fd567dcb7f41c33103c0406a0 | 8 | 3 | 20257 | 20865 |
| 6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13 | 8 | 3 | 23672 | 18644 |
| b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc | 8 | 3 | 41073 | 35339 |

## Itérations CG

| Instance | Types | Admissibles | Itérations Classical médian | Itérations Neural médian |
|---|---:|---:|---:|---:|
| 2e8d6ecbaa342a376ff4bf4c2261c4dc9641091f94c4aebbb8364b9e2584f6af | 2 | 3 | 1 | 1 |
| 3eb804881591494848f167ba7b26a2b9267a3ba54e1727e6932ff3b14cd58e50 | 2 | 3 | 1 | 1 |
| e89f36850422c2316a18c0e5c220f81c1d6ab1fbeb7c500622d6d755ac5409b2 | 2 | 3 | 1 | 1 |
| a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2 | 4 | 3 | 4 | 3 |
| b09c9bbfd876bc77c5440191ebfb1a02f5aa41b774b446ae71d1f3ca862891e9 | 4 | 3 | 3 | 3 |
| d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d | 4 | 3 | 3 | 3 |
| 2db8f04b0672b4e39578ab96c31e1989f5e93a806b484a143f83761c7a1f4c50 | 6 | 3 | 1 | 1 |
| 35714b99e7af814094a2bd53d0a82a4c33fb1c3bcdedddf861740c744140d125 | 6 | 3 | 2 | 2 |
| d71a500910a2039d84f38704af75d2036160060c50841a11b3d98be280d6f7fd | 6 | 3 | 6 | 5 |
| 5d7cde942f6f9a9811561fb1c16bb97f0f197f7fd567dcb7f41c33103c0406a0 | 8 | 3 | 3 | 3 |
| 6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13 | 8 | 3 | 5 | 2 |
| b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc | 8 | 3 | 12 | 10 |

## Colonnes (C = Classical, N = Neural)

| Instance | Types | Admissibles | Générées C/N | Ajoutées C/N | Finales C/N |
|---|---:|---:|---:|---:|---:|
| 2e8d6ecbaa342a376ff4bf4c2261c4dc9641091f94c4aebbb8364b9e2584f6af | 2 | 3 | 1 / 1 | 0 / 0 | 2 / 2 |
| 3eb804881591494848f167ba7b26a2b9267a3ba54e1727e6932ff3b14cd58e50 | 2 | 3 | 1 / 1 | 0 / 0 | 2 / 2 |
| e89f36850422c2316a18c0e5c220f81c1d6ab1fbeb7c500622d6d755ac5409b2 | 2 | 3 | 1 / 1 | 0 / 0 | 2 / 2 |
| a0828ac52b3d58f01bfdadbd2eee43a27a4dc2fb83b298eb885be71d25ab31a2 | 4 | 3 | 4 / 3 | 3 / 2 | 7 / 6 |
| b09c9bbfd876bc77c5440191ebfb1a02f5aa41b774b446ae71d1f3ca862891e9 | 4 | 3 | 3 / 3 | 2 / 2 | 6 / 6 |
| d98437a9e4d1ee8f330f95435351e503b39e6dc899a005394a24ce46ef0bfc7d | 4 | 3 | 3 / 3 | 2 / 2 | 6 / 6 |
| 2db8f04b0672b4e39578ab96c31e1989f5e93a806b484a143f83761c7a1f4c50 | 6 | 3 | 1 / 1 | 0 / 0 | 6 / 6 |
| 35714b99e7af814094a2bd53d0a82a4c33fb1c3bcdedddf861740c744140d125 | 6 | 3 | 2 / 2 | 1 / 1 | 7 / 7 |
| d71a500910a2039d84f38704af75d2036160060c50841a11b3d98be280d6f7fd | 6 | 3 | 6 / 5 | 5 / 4 | 11 / 10 |
| 5d7cde942f6f9a9811561fb1c16bb97f0f197f7fd567dcb7f41c33103c0406a0 | 8 | 3 | 3 / 3 | 2 / 2 | 10 / 10 |
| 6a97075552eaa33fc90b205108ea38eed24df5e00e18099237a5924d12beab13 | 8 | 3 | 5 / 2 | 4 / 1 | 12 / 9 |
| b7dc915107b5020410fb5597772f3b4a18c1dc4ded8e114a35d3dc5e895e96cc | 8 | 3 | 12 / 10 | 11 / 9 | 19 / 17 |

## Garanties et limites

Les deux maîtres entiers sont résolus sur les colonnes générées uniquement : les objectifs rapportés sont des optimaux sur colonnes générées et ne préjugent pas d'un optimum entier global. Aucune médiane n'agrège une paire hors tolérance de qualité ou incomplètement mesurée ; ces paires restent listées dans le JSON source. La mémoire est le pic d'allocations tracé par `tracemalloc` pendant le solveur, pas la consommation RSS du processus.
