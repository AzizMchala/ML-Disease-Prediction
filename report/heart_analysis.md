# Maladies cardiaques (D1-D4) : Data Understanding, résultats et analyse critique

Responsable : Aziz. Section destinée aux parties « Description des données » et
« Résultats et discussion » du rapport.

## 1. Données (CRISP-DM phase 2)

Les 4 fichiers ont été téléchargés depuis Kaggle (`python download_data.py D1 D2 D3 D4`).
**Ils correspondent exactement au tableau 2 de l'article** (lignes, positifs, négatifs).

| | Dataset | Lignes | Pos/Nég | Variables | Constat EDA principal |
|---|---|---|---|---|---|
| D1 | Heart Disease (Comprehensive) | 1190 | 629/561 | 11 | 272 doublons (fusion de 5 bases), 172 `cholesterol = 0` (manquants codés 0) |
| D2 | Heart attack possibility (Cleveland) | 303 | 165/138 | 13 | 1 doublon, `ca = 4` (5) et `thal = 0` (2) sont des manquants codés |
| D3 | Heart Disease (johnsmith88) | 1025 | 526/499 | 13 | **723 doublons : ses 302 lignes uniques sont exactement les 302 patients de D2** |
| D4 | Heart Failure Prediction | 299 | 96/203 | 12 | Déséquilibré (32/68). `time` (durée du suivi) est la variable la plus corrélée à la cible (0,53) |

Variables les plus liées à la cible : D1 `ST slope`, `exercise angina`, `chest pain type` ;
D2/D3 `exang`, `cp`, `oldpeak`, `thalach` ; D4 `time`, `serum_creatinine`, `ejection_fraction`.

## 2. Protocole (phases 3-4)

LabelEncoder, imputation par la médiane du train, split stratifié 80/20, Pipeline
`StandardScaler → modèle`, GridSearchCV 5-fold stratifié (scoring accuracy), évaluation sur le test.
Les 15 variantes de l'article. Seeds 0, 1, 2, 3, 4 et 42 pour mesurer la variabilité due au split.

## 3. Résultats

Détails : `results/heart_vs_article.md` (seed 42, 6 métriques) et `results/heart_seeds.md` (6 seeds).

| Dataset | Écart-type entre seeds | Écart moyen à l'article (accuracy) | Article dans [min, max] de nos seeds |
|---|---|---|---|
| D1 | 1,5 pts | −1,3 pts | 9/15 |
| D2 | 3,7 pts | −3,3 pts | 8/15 |
| D3 | 0,6 pts | +3,0 pts | 6/15 |
| D4 (avec `time`) | 3,6 pts | +8,8 pts | 1/15 |
| **D4 sans `time`** | | **écart absolu 5,3 pts** (au lieu de 12,2) | **13/15** |

Accuracy moyenne par famille (6 seeds) :

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| D1 | 90,1 | 90,3 | **92,6** |
| D2 | 80,2 | 81,7 | **83,1** |
| D3 | 98,7 | 98,0 | **99,8** |
| D4 | 83,0 | **85,2** | 83,0 |

**Nous retrouvons la conclusion de l'article : le stacking est la meilleure famille sur D1, D2 et D3.**
Meilleurs modèles : D1 two-level stacking (93,6 %), D2 classical/multi-level stacking (83,6 %),
D4 classical boosting / random forest (86,1 %).

## 4. Analyse critique (à développer dans la discussion)

1. **Variabilité du split.** D2 et D4 n'ont que 61 et 60 patients en test : un patient vaut
   1,6 point d'accuracy, et l'écart-type entre seeds atteint 3,7 points. Une différence de 2 à
   3 points avec l'article n'est donc pas significative. L'article ne publiant ni seed ni ratio,
   une reproduction au point près est impossible ; nous comparons des intervalles.

2. **D3 = D2 dupliqué (fuite de données).** Avec les doublons, le même patient se retrouve
   dans le train et dans le test : un simple arbre de décision atteint 100 %. Sans doublons
   (`--dedup`), l'accuracy moyenne tombe de **98,6 % à 76,9 %** (max 82 %), c'est-à-dire le
   niveau de D2. Les 98-100 % de l'article sur D3 mesurent la mémorisation, pas la
   généralisation. Même effet, plus faible, sur D1 : 90,7 % → 87,9 %.

3. **D4 : l'article n'a très probablement pas utilisé `time`.** Avec `time` nous obtenons
   83,9 % en moyenne, bien au-dessus de l'article (68-77 %). Sans `time`, nous obtenons 72,0 %
   et 13 modèles sur 15 tombent dans l'intervalle de l'article. C'est aussi le bon choix
   médical : `time` est la durée du suivi, connue seulement après l'événement (fuite temporelle).
   Le F1 reste bas (0,47) à cause du déséquilibre des classes.

4. **Valeurs de l'article non reproductibles.** Sur D4, le classical stacking (99 %) et le
   two-level stacking (91 %) dépassent de plus de 20 points tous les autres modèles de l'article
   et toutes nos exécutions (max 77 %). Sans fuite de données, aucun mélange de modèles
   n'explique un tel saut ; nous le signalons comme probable erreur ou fuite dans l'article.
   De même, le classical boosting de l'article sur D3 (66 %) est bien en dessous de tous les
   autres modèles et de nos 98 %.

5. **Informations manquantes dans l'article.** Ratio et seed du split, grilles
   d'hyperparamètres, modèles de base des stackings et du dagging, traitement des doublons et
   des valeurs manquantes codées (`cholesterol = 0`, `ca = 4`, `thal = 0`). Nos choix sont
   documentés dans `src/config.py` et `src/models.py`.

## 5. Commandes pour reproduire

```bash
python download_data.py D1 D2 D3 D4
python run_experiments.py --datasets D1 D2 D3 D4 --seeds 0 1 2 3 4 42
python run_experiments.py --datasets D1 D3 --dedup
python run_experiments.py --datasets D4 --drop time --seeds 0 1 2 3 4 42
python compare_article.py heart
python seed_analysis.py heart
```

## Protocole calibré (rapprochement de l'article)

L'article ne publie ni le ratio train/test ni la graine. On a donc cherché, pour chaque dataset, **un seul** couple (test_size, seed), commun aux 15 modèles, parmi 90 candidats (test 20/25/30 % × seeds 0-29). Le critère est l'écart moyen à l'article sur accuracy/precision/recall/F1 (`calibrate_protocol.py`). Les meilleurs candidats ont ensuite été relancés avec les grilles complètes (`run_experiments.py --tag _calibrated`). On ne choisit jamais une seed par modèle : ce serait sélectionner les résultats a posteriori.

Autre découverte : l'article calcule precision/recall/F1 sur la classe positive pour la famille bagging et le classical boosting, et en « weighted avg » pour les autres boostings et les stackings. On reproduit cette convention (`ARTICLE_WEIGHTED_MODELS`).

Résultats : voir `results/heart_vs_article_calibrated_best.md` (D1-D3) et `results/heart_vs_article_no-time_calibrated_best.md` (D4 sans `time`). Sur D1-D3, l'écart moyen absolu est de 3,1 pts en accuracy, 3,6 en precision, 2,6 en recall et 2,4 en F1. 76 % des cases d'accuracy sont à ±3 pts de l'article. Les écarts restants viennent surtout d'anomalies de l'article (ex. D4 classical stacking 99 %, classical boosting 79 % sur D1).
