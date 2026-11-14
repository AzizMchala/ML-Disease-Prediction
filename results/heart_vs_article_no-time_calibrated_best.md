# heart : nos résultats vs l'article (protocole calibré : D4 test=20% seed=17)

Chaque case : **nous** / article (en %). Écart = nous - article. Precision/recall/F1 calculés avec la même moyenne que l'article : classe positive pour le bagging et le classical boosting, pondérée pour les autres.


## accuracy

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **72** / 68 | **72** / 68 | **68** / 70 | **75** / 70 | **77** / 70 | **78** / 77 | **73** / 70 | **77** / 75 | **73** / 73 | **73** / 73 | **78** / 77 | **75** / 70 | **70** / 99 | **73** / 91 | **78** / 75 | 5.6 pts |

Meilleur modèle : D4 nous=RSub / article=CStack

## precision

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **57** / 67 | **56** / 67 | **50** / 68 | **64** / 68 | **73** / 68 | **80** / 77 | **59** / 66 | **76** / 80 | **72** / 76 | **72** / 73 | **78** / 77 | **74** / 70 | **79** / 99 | **72** / 91 | **78** / 78 | 7.3 pts |

Meilleur modèle : D4 nous=RSub / article=CStack

## recall

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **42** / 48 | **47** / 48 | **47** / 52 | **47** / 52 | **42** / 52 | **42** / 44 | **53** / 86 | **77** / 75 | **73** / 73 | **73** / 73 | **78** / 77 | **75** / 70 | **70** / 99 | **73** / 90 | **78** / 75 | 7.9 pts |

Meilleur modèle : D4 nous=LGBM / article=CStack

## f1

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **48** / 56 | **51** / 56 | **49** / 51 | **55** / 51 | **53** / 51 | **55** / 62 | **56** / 75 | **75** / 73 | **72** / 71 | **72** / 73 | **77** / 76 | **74** / 70 | **59** / 99 | **71** / 94 | **77** / 73 | 8.1 pts |

Meilleur modèle : D4 nous=LGBM / article=CStack

## auc

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **76** / 77 | **77** / 68 | **69** / 65 | **77** / 65 | **79** / 65 | **75** / 70 | **76** / 68 | **78** / 70 | **79** / 69 | **79** / 71 | **78** / 79 | **77** / 67 | **85** / 88 | **77** / 84 | **79** / 81 | 6.7 pts |

Meilleur modèle : D4 nous=CStack / article=CStack

## auprc

| | CBag | DT | Dagg | RF | ET | RSub | CBoost | Ada | Cat | XGB | LGBM | Logit | CStack | 2-Lvl | Multi | écart moyen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D4 | **63** / 66 | **63** / 58 | **44** / 54 | **66** / 54 | **68** / 54 | **70** / 65 | **62** / 62 | **68** / 64 | **67** / 60 | **65** / 59 | **62** / 64 | **69** / 55 | **75** / 61 | **65** / 64 | **70** / 76 | 6.9 pts |

Meilleur modèle : D4 nous=CStack / article=Multi

## Écart moyen absolu par métrique

| accuracy | precision | recall | f1 | auc | auprc |
|---|---|---|---|---|---|
| 5.6 pts | 7.3 pts | 7.9 pts | 8.1 pts | 6.7 pts | 6.9 pts |

Cases d'accuracy à ±3 points de l'article : 40 %