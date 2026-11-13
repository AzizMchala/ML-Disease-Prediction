"""Les 15 variantes d'ensemble de l'article (tableau 1) et leurs grilles d'hyperparamètres.

L'article ne publie ni les modèles de base des stackings, ni les grilles exactes :
les choix ci-dessous sont les nôtres et doivent être justifiés dans le rapport.

Chaque modèle est un Pipeline (StandardScaler -> classifieur) : la normalisation est
ainsi ré-ajustée sur chaque pli du GridSearchCV, sans fuite vers la validation.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

from .config import SEED


# --------------------------------------------------------------------------- #
# Variantes absentes de scikit-learn                                           #
# --------------------------------------------------------------------------- #
class DaggingClassifier(ClassifierMixin, BaseEstimator):
    """Dagging (Ting & Witten, 1997) : le train est découpé en k parts DISJOINTES
    stratifiées, un modèle est appris par part, puis les probabilités sont moyennées.
    Différence avec le bagging : pas de tirage avec remise."""

    def __init__(self, estimator=None, n_estimators=5, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier()
        skf = StratifiedKFold(self.n_estimators, shuffle=True, random_state=self.random_state)
        self.estimators_ = [clone(base).fit(X[idx], y[idx]) for _, idx in skf.split(X, y)]
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        P = np.zeros((len(X), len(self.classes_)))
        for m in self.estimators_:
            cols = np.searchsorted(self.classes_, m.classes_)
            P[:, cols] += m.predict_proba(X)
        return P / len(self.estimators_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class LogitBoostClassifier(ClassifierMixin, BaseEstimator):
    """LogitBoost (Friedman, Hastie & Tibshirani, 2000) : boosting par régression
    logistique additive, arbres de régression comme apprenants faibles.
    Multiclasse traité en un-contre-tous."""

    def __init__(self, n_estimators=100, learning_rate=0.5, max_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _fit_binary(self, X, y01):
        F = np.zeros(len(X))
        trees = []
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-2.0 * F))
            w = np.clip(p * (1 - p), 1e-10, None)
            z = np.clip((y01 - p) / w, -4, 4)
            t = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            t.fit(X, z, sample_weight=w)
            F += 0.5 * self.learning_rate * t.predict(X)
            trees.append(t)
        return trees

    def _decision(self, X, trees):
        return sum(0.5 * self.learning_rate * t.predict(X) for t in trees)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        targets = [self.classes_[1]] if len(self.classes_) == 2 else list(self.classes_)
        self.models_ = [self._fit_binary(X, (y == c).astype(float)) for c in targets]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        S = np.column_stack([1 / (1 + np.exp(-2 * self._decision(X, m))) for m in self.models_])
        if len(self.classes_) == 2:
            return np.column_stack([1 - S[:, 0], S[:, 0]])
        return S / S.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# --------------------------------------------------------------------------- #
# Fabrique des 15 modèles                                                      #
# --------------------------------------------------------------------------- #
def _dt():
    return DecisionTreeClassifier(random_state=SEED)


def _lr():
    return LogisticRegression(max_iter=2000, random_state=SEED)


def _classical_stacking():
    # Stacking hétérogène classique : modèles de base simples et variés + méta-modèle LR
    return StackingClassifier(
        estimators=[("dt", _dt()), ("knn", KNeighborsClassifier()),
                    ("svc", SVC(probability=True, random_state=SEED)), ("nb", GaussianNB())],
        final_estimator=_lr(), cv=5, n_jobs=1)


def _two_level_stacking():
    # Niveau 1 : ensembles forts ; niveau 2 : méta-modèle sur predict_proba (article, 2.4)
    return StackingClassifier(
        estimators=[("rf", RandomForestClassifier(random_state=SEED)),
                    ("et", ExtraTreesClassifier(random_state=SEED)),
                    ("xgb", XGBClassifier(random_state=SEED, verbosity=0)),
                    ("ada", AdaBoostClassifier(algorithm="SAMME", random_state=SEED))],
        final_estimator=_lr(), stack_method="predict_proba", cv=5, n_jobs=1)


def _multi_level_stacking():
    # Niveau 1 : deux stackings différents ; niveau 2 : leurs sorties ; niveau 3 : méta-modèle
    stack_a = StackingClassifier(
        estimators=[("dt", _dt()), ("knn", KNeighborsClassifier()), ("nb", GaussianNB())],
        final_estimator=_lr(), cv=5, n_jobs=1)
    stack_b = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(random_state=SEED)),
                    ("xgb", XGBClassifier(random_state=SEED, verbosity=0)),
                    ("svc", SVC(probability=True, random_state=SEED))],
        final_estimator=_lr(), cv=5, n_jobs=1)
    return StackingClassifier(estimators=[("stack_a", stack_a), ("stack_b", stack_b)],
                              final_estimator=_lr(), stack_method="predict_proba",
                              cv=5, n_jobs=1)


def _catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)


def _lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1)


# (estimateur, grille complète, grille rapide). Les clés sont préfixées "clf__"
# au moment de construire le Pipeline.
_REGISTRY = {
    "classical_bagging": (
        lambda: BaggingClassifier(estimator=_dt(), random_state=SEED),
        {"n_estimators": [10, 50, 100], "max_samples": [0.5, 0.8, 1.0]},
        {"n_estimators": [50]}),
    "decision_tree": (
        _dt,
        {"criterion": ["gini", "entropy"], "max_depth": [None, 3, 5, 10],
         "min_samples_split": [2, 5, 10]},
        {"max_depth": [None, 5]}),
    "dagging": (
        lambda: DaggingClassifier(estimator=_dt(), random_state=SEED),
        {"n_estimators": [3, 5, 10]},
        {"n_estimators": [5]}),
    "random_forest": (
        lambda: RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 300], "max_depth": [None, 5, 10],
         "max_features": ["sqrt", "log2"]},
        {"n_estimators": [100]}),
    "extra_trees": (
        lambda: ExtraTreesClassifier(random_state=SEED),
        {"n_estimators": [100, 300], "max_depth": [None, 5, 10],
         "max_features": ["sqrt", "log2"]},
        {"n_estimators": [100]}),
    "random_subspace": (
        # Random Subspace (Ho, 1998) : pas de bootstrap des lignes, tirage des colonnes
        lambda: BaggingClassifier(estimator=_dt(), bootstrap=False, bootstrap_features=True,
                                  random_state=SEED),
        {"n_estimators": [10, 50, 100], "max_features": [0.5, 0.7]},
        {"n_estimators": [50], "max_features": [0.5]}),
    "classical_boosting": (
        lambda: GradientBoostingClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1], "max_depth": [2, 3]},
        {"n_estimators": [100]}),
    "adaboost": (
        lambda: AdaBoostClassifier(algorithm="SAMME", random_state=SEED),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
        {"n_estimators": [100]}),
    "catboost": (
        _catboost,
        {"iterations": [200, 500], "depth": [4, 6], "learning_rate": [0.03, 0.1]},
        {"iterations": [200]}),
    "xgboost": (
        lambda: XGBClassifier(random_state=SEED, verbosity=0),
        {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1],
         "subsample": [0.8, 1.0]},
        {"n_estimators": [100]}),
    "lightgbm": (
        _lightgbm,
        {"n_estimators": [100, 300], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
        {"n_estimators": [100]}),
    "logitboost": (
        lambda: LogitBoostClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5], "max_depth": [1, 2]},
        {"n_estimators": [100]}),
    "classical_stacking": (
        _classical_stacking,
        {"final_estimator__C": [0.1, 1, 10], "knn__n_neighbors": [5, 9]},
        {"final_estimator__C": [1]}),
    "two_level_stacking": (
        _two_level_stacking,
        {"final_estimator__C": [0.1, 1, 10], "rf__n_estimators": [100, 300]},
        {"final_estimator__C": [1]}),
    "multi_level_stacking": (
        _multi_level_stacking,
        {"final_estimator__C": [0.1, 1, 10]},
        {"final_estimator__C": [1]}),
}


def get_model(name: str, quick: bool = False):
    """Retourne (pipeline, grille) prêts pour GridSearchCV."""
    build, grid, quick_grid = _REGISTRY[name]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", build())])
    g = quick_grid if quick else grid
    return pipe, {f"clf__{k}": v for k, v in g.items()}
