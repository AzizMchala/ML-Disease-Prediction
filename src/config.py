"""Paramètres communs à toute l'équipe (ne pas modifier sans prévenir le groupe).

Protocole tiré de la section 2.4 de l'article de référence
(Mahajan et al., Health and Technology, 2024, doi:10.1007/s12553-024-00835-w) :
encodage des catégorielles, normalisation, split train/test, GridSearchCV 5-fold
optimisé sur l'accuracy, évaluation sur le jeu de test.
"""
import os
from pathlib import Path

# Évite les avertissements de joblib/loky sous Windows (détection des cœurs physiques)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
ARTICLE_REF = ROOT / "src" / "article_reference.csv"

SEED = 42            # graine fixe -> expériences reproductibles
TEST_SIZE = 0.2      # l'article ne donne pas le ratio : 80/20 par défaut, à faire varier
CV_FOLDS = 5         # "Fivefold cross-validation" (article, section 2.4)
SCORING = "accuracy" # "the model with the highest accuracy is chosen"
N_JOBS = -1

MODELS = [
    "classical_bagging", "decision_tree", "dagging", "random_forest", "extra_trees",
    "random_subspace", "classical_boosting", "adaboost", "catboost", "xgboost",
    "lightgbm", "logitboost", "classical_stacking", "two_level_stacking",
    "multi_level_stacking",
]
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "auprc"]

# Moyenne utilisée par l'article pour precision / recall / F1, déduite de ses tableaux :
# - famille bagging + classical boosting : classe positive seule (recall souvent ≠ accuracy)
# - autres boostings + stackings : "weighted avg" du classification_report (recall = accuracy)
# Les deux versions sont calculées ; *_w = moyenne pondérée.
ARTICLE_WEIGHTED_MODELS = {
    "adaboost", "catboost", "xgboost", "lightgbm", "logitboost",
    "classical_stacking", "two_level_stacking", "multi_level_stacking",
}
