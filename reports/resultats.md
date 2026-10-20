# Résultats

30 000 porteurs de carte, taux de défaut global 22,1 %. Segmentation sur 7 indicateurs de comportement standardisés ; le défaut n'est utilisé qu'après coup, pour interpréter les segments.

## Choix de k

|   k | inertie   |   silhouette | calinski_harabasz   |   davies_bouldin |
|----:|:----------|-------------:|:--------------------|-----------------:|
|   2 | 129 850   |        0,396 | 18 516              |            1,167 |
|   3 | 99 666    |        0,391 | 16 604              |            1,004 |
|   4 | 71 440    |        0,439 | 19 393              |            0,910 |
|   5 | 58 633    |        0,362 | 19 359              |            0,995 |
|   6 | 53 065    |        0,335 | 17 741              |            0,991 |
|   7 | 48 990    |        0,315 | 16 429              |            1,159 |
|   8 | 45 811    |        0,311 | 15 356              |            1,181 |

Stabilité (k = 4, 20 rééchantillons bootstrap) : ARI moyen 0,997, minimum 0,995.

## Segments

| segment                   | clients   | part des clients   | taux de défaut [IC 95 %]   | part des défauts   |   âge médian |
|:--------------------------|:----------|:-------------------|:---------------------------|:-------------------|-------------:|
| Transacteurs              | 6 653     | 22 %               | 16,5 % [15,6 % ; 17,4 %]   | 17 %               |           35 |
| Revolvers                 | 15 359    | 51 %               | 15,8 % [15,2 % ; 16,4 %]   | 36 %               |           33 |
| Peu actifs                | 3 973     | 13 %               | 19,1 % [17,9 % ; 20,4 %]   | 11 %               |           35 |
| En difficulté de paiement | 4 015     | 13 %               | 58,8 % [57,2 % ; 60,3 %]   | 36 %               |           33 |

## Liens statistiques

- Segment × défaut : test du χ², p < 10⁻³⁰⁰, V de Cramér = 0,348.
- Âge selon le segment : Kruskal-Wallis, p = 2,2 × 10⁻⁶⁵, ε² = 0,0101 (différence significative mais négligeable).
- Taille d'effet ε² des variables de segmentation : mois en crédit renouvelable 0,80, mois soldés intégralement 0,72, mois sans utilisation 0,68, taux d'utilisation du plafond 0,61, part des factures remboursée 0,55, mois en retard de paiement 0,49, plafond de crédit (log) 0,16.
