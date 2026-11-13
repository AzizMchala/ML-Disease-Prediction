# Prédiction de maladies par apprentissage d'ensemble

Reproduction et analyse critique de :
> Mahajan P., Uddin S., Hajati F., Moni M.A., Gide E. *A comparative evaluation of machine learning
> ensemble approaches for disease prediction using multiple datasets.* Health and Technology 14, 597–613 (2024).
> https://doi.org/10.1007/s12553-024-00835-w

15 variantes d'ensemble (bagging, boosting, stacking) × 16 datasets × 6 métriques
(accuracy, precision, recall, F1, AUC, AUPRC), plus le RPI et le nombre de victoires par modèle.

**Méthodologie : CRISP-DM** (Business Understanding, Data Understanding, Data Preparation,
Modeling, Evaluation, Deployment).

## Équipe

| Membre | Maladie | Datasets | Branche |
|---|---|---|---|
| Aziz | Cardiaque | D1-D4 | `feature/heart-aziz` |
| Fedi | Foie | D5-D7 | `feature/liver-fedi` |
| Omar | Diabète | D8-D11 | `feature/diabetes-omar` |
| Rami | Rein | D12-D14 | `feature/kidney-rami` |
| Khalil, Elyesse | Peau | D15-D16 | `feature/skin-khalil-elyesse` |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Structure

```
data/raw/<maladie>/        CSV d'origine, renommés D1.csv, D2.csv... (non versionnés, voir .gitignore)
notebooks/<maladie>/       EDA : un notebook par dataset (phase 2)
src/config.py              seed, split, CV, liste des modèles et des métriques (commun)
src/preprocessing.py       nettoyage, encodage, imputation, split stratifié (phase 3)
src/models.py              les 15 variantes + grilles GridSearchCV (phase 4)
src/evaluate.py            métriques, matrice de confusion, RPI, best count (phase 5)
src/datasets/<maladie>.py  déclaration des datasets de chaque membre (_template.py = modèle)
src/article_reference.csv  valeurs des tableaux 3, 5, 7, 9, 11, 13 de l'article
run_experiments.py         entraîne et évalue -> results/<maladie>_results.csv
integrate.py               fusionne tous les résultats -> results/summary/
report/                    rapport PDF
```

## Utilisation

```bash
python download_data.py D1 D2 D3 D4                        # télécharge depuis Kaggle + vérifie le tableau 2
python run_experiments.py --datasets D1 D2 D3 D4 --quick   # test rapide du pipeline
python run_experiments.py --datasets D1 D2 D3 D4           # protocole complet
python run_experiments.py --datasets D3 --dedup            # sans doublons
python run_experiments.py --datasets D4 --drop time        # retirer une colonne
python run_experiments.py --datasets D4 --seeds 0 1 42     # sensibilité au split
python seed_analysis.py heart                              # moyenne ± écart-type sur les seeds
python integrate.py                                        # tableaux finaux équipe
```

## Protocole (section 2.4 de l'article)

1. Encodage des variables catégorielles (LabelEncoder), imputation par la médiane du train
2. Split train/test stratifié 80/20, `random_state=42` (l'article ne précise pas : à documenter)
3. Pipeline `StandardScaler -> modèle`, GridSearchCV 5-fold stratifié, scoring = accuracy
4. Évaluation sur le test : 6 métriques + matrice de confusion, comparaison à l'article (`delta_*`)

Ce que l'article ne précise pas (et que nous avons dû choisir) : ratio du split, seed, grilles
d'hyperparamètres, modèles de base des stackings, modèle de base du dagging, traitement des doublons.

## Règles Git

- `main` est protégée : on y fusionne seulement par Pull Request relue par un autre membre.
- Une branche par membre. Commits courts et explicites, par exemple `EDA D3: 723 doublons détectés`.
- On ne modifie `src/config.py`, `src/models.py` ou `src/evaluate.py` qu'après accord du groupe :
  ce sont eux qui garantissent que tous les résultats sont comparables.
- On ne versionne pas les données (licences Kaggle) : chaque README de dossier donne la source.
