# heart : nos résultats vs l'article (protocole calibré : D4 test=20% seed=0)

Chaque case : **nous** / article (en %). Écart = nous - article. Precision/recall/F1 calculés avec la même moyenne que l'article : classe positive pour le bagging et le classical boosting, pondérée pour les autres.


## accuracy

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **77** / 68 | **70** / 68 | **72** / 70 | **78** / 70 | **75** / 70 | **78** / 77 | **77** / 70 | **75** / 75 | **78** / 73 | **75** / 73 | **68** / 77 | **73** / 70 | **73** / 99 | **75** / 91 | **73** / 75 | 6.4 pts |

Meilleur modèle : D4 nous=RF / article=CStack

## precision

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **63** / 67 | **53** / 67 | **56** / 68 | **69** / 68 | **67** / 68 | **69** / 77 | **63** / 66 | **74** / 80 | **79** / 76 | **75** / 73 | **68** / 77 | **72** / 70 | **72** / 99 | **74** / 91 | **72** / 78 | 7.7 pts |

Meilleur modèle : D4 nous=Cat / article=CStack

## recall

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **63** / 48 | **42** / 48 | **47** / 52 | **58** / 52 | **42** / 52 | **58** / 44 | **63** / 86 | **75** / 75 | **78** / 73 | **75** / 73 | **68** / 77 | **73** / 70 | **73** / 99 | **75** / 90 | **73** / 75 | 9.3 pts |

Meilleur modèle : D4 nous=Cat / article=CStack

## f1

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **63** / 56 | **47** / 56 | **51** / 51 | **63** / 51 | **52** / 51 | **63** / 62 | **63** / 75 | **74** / 73 | **78** / 71 | **75** / 73 | **68** / 76 | **72** / 70 | **71** / 99 | **74** / 94 | **71** / 73 | 7.5 pts |

Meilleur modèle : D4 nous=Cat / article=CStack

## auc

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **79** / 77 | **70** / 68 | **74** / 65 | **81** / 65 | **76** / 65 | **79** / 70 | **77** / 68 | **79** / 70 | **76** / 69 | **75** / 71 | **70** / 79 | **77** / 67 | **74** / 88 | **77** / 84 | **79** / 81 | 8.0 pts |

Meilleur modèle : D4 nous=RF / article=CStack

## auprc

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **64** / 66 | **48** / 58 | **58** / 54 | **68** / 54 | **64** / 54 | **68** / 65 | **62** / 62 | **65** / 64 | **63** / 60 | **58** / 59 | **49** / 64 | **57** / 55 | **63** / 61 | **63** / 64 | **66** / 76 | 5.3 pts |

Meilleur modèle : D4 nous=RF / article=Multi

## Écart moyen absolu par métrique

| accuracy | precision | recall | f1 | auc | auprc |
|---|---|---|---|---|---|
| 6.4 pts | 7.7 pts | 9.3 pts | 7.5 pts | 8.0 pts | 5.3 pts |

Cases d'accuracy à ±3 points de l'article : 40 %