"""Reconstitution du protocole non publié par l'article (ratio du test + seed).

L'article ne donne ni le ratio train/test ni la graine aléatoire. On cherche donc, pour
chaque dataset, UN SEUL couple (test_size, seed), commun aux 15 modèles, qui minimise
l'écart moyen absolu avec l'article sur accuracy, precision, recall et F1.

Règle d'honnêteté : on ne choisit jamais une seed différente par modèle (ce serait
sélectionner les résultats a posteriori). Le protocole retenu s'applique à tous les modèles.

Étape 1 (ici) : recherche rapide (grilles réduites, sans les 2 stackings les plus lourds).
Étape 2 : relancer le protocole complet sur les meilleurs candidats avec run_experiments.py.

    python calibrate_protocol.py --datasets D1 D2 D3 --seeds 30
    python calibrate_protocol.py --datasets D4 --drop time --seeds 30

Sortie : results/calibration_<dataset>.csv (une ligne par couple testé)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 1 thread par tâche, parallélisme par config

import argparse
import itertools
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.datasets import load
from src.evaluate import compute_metrics, load_article_reference
from src.models import get_model
from src.preprocessing import prepare

FIT_METRICS = ["accuracy", "precision", "recall", "f1"]
HEAVY = {"two_level_stacking", "multi_level_stacking"}


def evaluate_config(df, target, dataset_id, test_size, seed, models, ref):
    warnings.filterwarnings("ignore")
    X_tr, X_te, y_tr, y_te = prepare(df, target, test_size=test_size, seed=seed)
    gaps, rows = [], {}
    for name in models:
        pipe, grid = get_model(name, quick=True)
        cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
        s = GridSearchCV(pipe, grid, cv=cv, scoring=config.SCORING, n_jobs=1).fit(X_tr, y_tr)
        m = compute_metrics(y_te, s.predict(X_te), s.predict_proba(X_te))
        rows[name] = m["accuracy"]
        w = name in config.ARTICLE_WEIGHTED_MODELS  # même moyenne que l'article
        gaps += [abs(m[f"{k}_w" if (w and k != "accuracy") else k] - ref[(name, k)])
                 for k in FIT_METRICS]
    out = {"dataset": dataset_id, "test_size": test_size, "seed": seed,
           "mean_abs_gap": sum(gaps) / len(gaps)}
    out.update({f"acc_{k}": v for k, v in rows.items()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=30, help="seeds 0..N-1")
    p.add_argument("--test-sizes", nargs="+", type=float, default=[0.2, 0.25, 0.3])
    p.add_argument("--drop", nargs="+", default=[])
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    a = p.parse_args()

    models = [m for m in config.MODELS if m not in HEAVY]
    article = load_article_reference()
    for ds in a.datasets:
        df, target = load(ds)
        df = df.drop(columns=[c for c in a.drop if c in df.columns])
        ref = {(r.model, r.metric): r.value for r in article[article.dataset == ds].itertuples()}
        grid = list(itertools.product(a.test_sizes, range(a.seeds)))
        print(f"== {ds} : {len(grid)} protocoles × {len(models)} modèles")
        res = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(evaluate_config)(df, target, ds, ts, sd, models, ref) for ts, sd in grid)
        res = pd.DataFrame(res).sort_values("mean_abs_gap")
        suffix = "".join(f"_no-{c}" for c in a.drop)
        out = config.RESULTS_DIR / f"calibration_{ds}{suffix}.csv"
        res.to_csv(out, index=False)
        print(res[["test_size", "seed", "mean_abs_gap"]].head(5).to_string(index=False))
        print(f"   pire : {res['mean_abs_gap'].max():.3f} | médiane : {res['mean_abs_gap'].median():.3f}")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
