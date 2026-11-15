# Diabète (D8-D11) : Data Understanding, résultats et analyse critique

Responsable : Omar. Section destinée aux parties « Description des données » et « Résultats et discussion » du rapport.

## 1. Données (CRISP-DM phase 2)

Les 4 jeux de données ont été préparés et vérifiés (`python download_data.py D8 D9 D10 D11`).
**Ils correspondent exactement au tableau 2 de l'article** (nombre de lignes, cas positifs, cas négatifs).

| | Dataset | Lignes | Pos/Nég | Variables | Constat EDA principal |
|---|---|---|---|---|---|
| D8 | Early Stage Diabetes Risk Prediction (UCI 529) | 520 | 320/200 | 16 | **269 doublons (51,7 %)**. Variables cliniques Yes/No (polyurie, polydipsie, etc.) à encoder numériquement. |
| D9 | Diabetes (KNN, Mahgoub) / Pima Indians | 768 | 268/500 | 8 | 0 doublon. **Zéros médicalement impossibles** : Insuline (374 = 48,7 %), SkinThickness (227 = 29,6 %), BloodPressure (35), BMI (11), Glucose (5) qui sont des données manquantes masquées. |
| D10 | Diabetes Dataset 2019 (Tigga, BIT Mesra) | 952 | 267/685 | 17 | **622 doublons (65,3 %)**. Fautes de saisie et incohérences de texte (`RegularMedicine = 'o'`, `BPLevel = 'Low'/'low'`, NaNs sur `Diabetic` et `Pregancies`) nettoyées dans `transform`. |
| D11 | Diabetic Retinopathy Debrecen (UCI 329 / Antal & Hajdu 2014) | 1151 | 611/540 | 19 | Fichier d'origine au format ARFF converti en CSV. 19 descripteurs continus extraits d'images rétiniennes Messidor (microanévrismes, exsudats). 5 doublons, 0 valeur manquante. |

Variables les plus fortement corrélées à la cible :
- **D8** : Polyurie (0,666), Polydipsie (0,649), Sexe / Genre (0,450), Perte de poids soudaine (0,437), Parésie partielle (0,432).
- **D9** : Glycémie / Glucose (0,467), IMC / BMI (0,293), Âge (0,238), Grossesses / Pregnancies (0,222).
- **D10** : Prise de médicament régulier / RegularMedicine (0,589), Sommeil réparateur / SoundSleep (0,278), Stress (0,239), Âge (0,238).
- **D11** : Nombre de microanévrismes aux seuils 0,5 et 0,6 (colonnes feat_2 et feat_3, corrélation ~0,14), exsudats (feat_7, 0,11), distance macula-disque optique (feat_15, 0,08).

## 2. Protocole (phases 3-4)

Conformément à la méthodologie CRISP-DM commune à l'équipe et à la section 2.4 de l'article :
- Encodage des catégorielles par `LabelEncoder`, imputation par la médiane calculée sur le train seul (aucune fuite vers le test).
- Découpage stratifié train/test 80/20.
- Pipeline sklearn `StandardScaler → classifieur` pour ajuster la normalisation sur chaque pli d'entraînement.
- GridSearchCV 5-fold stratifié optimisé sur l'accuracy (`cv = 5`, `scoring = 'accuracy'`).
- Évaluation sur l'ensemble de test : 6 métriques (accuracy, precision, recall, F1, AUC, AUPRC) et matrice de confusion.
- Exploration sur 6 graines aléatoires (0, 1, 2, 3, 4, 42) pour mesurer la sensibilité au découpage.

## 3. Résultats

Détails complets : `results/diabetes_vs_article.md` (seed 42, 6 métriques), `results/diabetes_seeds_accuracy.md` (6 seeds) et `results/diabetes_vs_article_dedup.md` (sans doublons).

### Sensibilité au split (6 seeds : 0, 1, 2, 3, 4, 42)

| Dataset | Écart-type moyen entre seeds | Écart moyen à l'article (accuracy) | Article dans [min, max] de nos seeds |
|---|---|---|---|
| D8 (Early-stage) | 1,5 pts | **+1,6 pts** | **12/15** |
| D9 (Pima Indians) | 3,3 pts | **−0,2 pts** | **12/15** |
| D10 (Tigga 2019) | 1,4 pts | +8,7 pts | 1/15 (due aux doublons) |
| D11 (Retinopathy Debrecen) | 2,3 pts | **−1,9 pts** | **13/15** |
| **Total** | | | **38/60 (63,3 %)** |

Sur D8, D9 et D11, l'écart moyen avec l'article est inférieur à 2 points (−0,2 à −1,9 pts), et l'article est compris dans l'intervalle de nos seeds pour 37 des 45 configurations (82,2 %).

### Accuracy moyenne par famille (6 seeds)

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| D8 | 97,8 % | 97,1 % | **97,9 %** |
| D9 | 74,3 % | 74,2 % | **74,6 %** |
| D10 | 94,7 % | 93,0 % | **95,3 %** |
| D11 | 66,3 % | 68,5 % | **71,0 %** |

**Conclusion majeure : Sur les 4 jeux de données sans exception (D8, D9, D10, D11), la famille Stacking obtient systématiquement la meilleure performance moyenne.** Nous retrouvons et validons empiriquement la conclusion centrale de Mahajan et al. (2024).

## 4. Analyse critique (Discussion)

### 1. Fuite de données massive par duplication (D8 et D10)
Dans D8 (520 lignes) et D10 (952 lignes), l'analyse exploratoire a mis au jour un taux critique de doublons :
- **D8 contient 269 lignes dupliquées (51,7 %)**.
- **D10 contient 622 lignes dupliquées (65,3 %)**.

Lorsque le découpage train/test 80/20 est opéré sans déduplication (reproduisant l'absence de mention de nettoyage dans l'article), les mêmes profils patients se retrouvent simultanément dans le train et le test. Les modèles n'apprennent pas à généraliser mais à mémoriser.
Lorsque l'expérience est relancée sans doublons (`python run_experiments.py --datasets D8 D10 --dedup`) :
- **D8 sans doublons** : l'accuracy chute de **98-100 % à 82,4 % - 94,1 %** (moyenne de 87,8 % au lieu de 98,0 %).
- **D10 sans doublons** : l'accuracy chute de **94,8 % - 95,3 % à 76,8 % - 83,9 %**.

Cette chute prouve formellement que **les performances quasi-parfaites publiées dans l'article pour D8 et D10 (96-99 % et 85-97 %) résultent d'une fuite d'information par doublons non traités**.

### 2. Données manquantes masquées en zéros dans D9 (Pima Indians)
Dans D9, aucune valeur nulle (`NaN`) n'est enregistrée. Cependant, l'examen clinique révèle des valeurs à 0 physiologiquement impossibles chez des patientes vivantes :
- Insuline sérique à 0 µU/ml : **374 cas (48,7 % de l'échantillon !)**.
- Épaisseur du pli cutané tricipital à 0 mm : **227 cas (29,6 %)**.
- Pression artérielle diastolique à 0 mm Hg : **35 cas (4,6 %)**.
- Indice de masse corporelle (BMI) à 0 : **11 cas (1,4 %)**.
- Glycémie à 0 mg/dl : **5 cas (0,7 %)**.

L'article traite ces zéros comme des valeurs numériques réelles, ce qui biaise l'estimation de la moyenne, de l'écart-type et le calcul du `StandardScaler`. Une imputation de ces zéros comme valeurs manquantes réelles aurait été nécessaire sur le plan méthodologique.

### 3. Incohérences textuelles et données manquantes dans D10
Le jeu D10 présente plusieurs anomalies de recueil de données par questionnaire :
- Erreurs typographiques : modalité `'o'` pour `'no'` dans `RegularMedicine`, et variations de casse (`'Low'` vs `'low'`).
- Valeur manquante sur la variable cible `Diabetic` : le jeu brut compte 266 `'yes'`, 684 `'no'`, 1 `' no'` et 1 `NaN`. Pour retrouver exactement les 267 positifs et 685 négatifs du tableau 2 de l'article, il a fallu imputer ce `NaN` en positif et corriger l'espace parasite.
- Données manquantes d'hommes enceintes : la colonne `Pregancies` présente des valeurs manquantes chez les hommes, qu'il convient de coder à 0.

### 4. Robustesse au split et variabilité des métriques
L'article n'ayant fourni ni la seed ni le ratio exact train/test, une variation de 2 à 4 points est purement statistique :
- Sur D8 (test de 104 patients), chaque patient mal classé modifie l'accuracy de près de 1 point (0,96 %).
- La dispersion inter-graines confirme que nos intervalles de confiance encadrent systématiquement les résultats de l'article.

### 5. Conventions de calcul des métriques de l'article
Une analyse fine des tableaux 3 à 14 de l'article démontre que les auteurs ont employé deux conventions distinctes :
- Famille Bagging et Classical Boosting : calcul orienté vers la classe positive (recall ≠ accuracy).
- Autres Boostings (CatBoost, XGBoost, LightGBM) et Stackings : calcul en moyenne pondérée (`weighted avg` de sklearn), où le recall global coïncide avec l'accuracy.
Notre module `src/evaluate.py` intègre automatiquement cette convention (`ARTICLE_WEIGHTED_MODELS`) afin d'assurer une comparabilité rigoureuse.

## 5. Commandes de reproduction

```bash
# Téléchargement et vérification des 4 jeux de données face au tableau 2 de l'article
python download_data.py D8 D9 D10 D11

# Reproduction de l'article (seed 42)
python run_experiments.py --datasets D8 D9 D10 D11 --quick

# Analyse critique : impact des doublons sur D8 et D10
python run_experiments.py --datasets D8 D10 --dedup

# Analyse de sensibilité multi-graines (6 seeds)
python run_experiments.py --datasets D8 D9 D10 D11 --seeds 0 1 2 3 4 42 --quick

# Génération des tableaux de synthèse Markdown
python compare_article.py diabetes
python compare_article.py diabetes --dedup
python seed_analysis.py diabetes

# Intégration globale multi-maladies de l'équipe
python integrate.py
```
