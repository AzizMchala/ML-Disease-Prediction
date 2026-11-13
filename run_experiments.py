"""Phase 4-5 de CRISP-DM : entraîne les 15 variantes sur un ou plusieurs datasets.

Exemples :
    python run_experiments.py --datasets D1 D2 D3 D4            # protocole complet
    python run_experiments.py --datasets D3 --dedup             # sans doublons (analyse critique)
    python run_experiments.py --datasets D4 --seeds 0 1 42      # sensibilité au split
    python run_experiments.py --datasets D1 --quick             # test rapide (petites grilles)
    python run_experiments.py --datasets D1 --models xgboost lightgbm

Sortie : results/<maladie>_results[_dedup].csv (format commun à toute l'équipe)
         results/figures/<dataset>_<modèle>_cm.png (matrices de confusion)
"""
import argparse
import time
import warnings

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.datasets import REGISTRY, load
from src.evaluate import add_article_comparison, compute_metrics, save_confusion_matrix
from src.models import get_model
from src.preprocessing import prepare

warnings.filterwarnings("ignore")


def run(dataset_id, models, seed, test_size, dedup, quick):
    df, target = load(dataset_id)
    X_tr, X_te, y_tr, y_te = prepare(df, target, drop_duplicates=dedup,
                                     test_size=test_size, seed=seed)
    rows = []
    for name in models:
        t0 = time.time()
        pipe, grid = get_model(name, quick=quick)
        cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid, cv=cv, scoring=config.SCORING, n_jobs=config.N_JOBS)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)
        m = compute_metrics(y_te, y_pred, y_proba)
        tag = f"{dataset_id}_{name}" + ("_dedup" if dedup else "")
        cm = save_confusion_matrix(y_te, y_pred, f"{dataset_id} - {name}",
                                   config.FIG_DIR / f"{tag}_cm.png")
        rows.append({
            "dataset": dataset_id, "disease": REGISTRY[dataset_id]["disease"], "model": name,
            "seed": seed, "test_size": test_size, "dedup": dedup,
            "n_train": len(y_tr), "n_test": len(y_te), **m,
            "cv_accuracy": search.best_score_, "best_params": str(search.best_params_),
            "confusion_matrix": cm.tolist(), "fit_seconds": round(time.time() - t0, 1),
        })
        print(f"  {dataset_id} {name:22s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
              f"auc={m['auc']:.3f} ({rows[-1]['fit_seconds']}s)")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=config.MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[config.SEED])
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p.add_argument("--dedup", action="store_true", help="supprimer les doublons avant le split")
    p.add_argument("--quick", action="store_true", help="grilles réduites (test du pipeline)")
    args = p.parse_args()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            print(f"== {ds} | seed={seed} | test_size={args.test_size} | dedup={args.dedup}")
            rows += run(ds, args.models, seed, args.test_size, args.dedup, args.quick)

    res = add_article_comparison(pd.DataFrame(rows))
    for disease, part in res.groupby("disease"):
        out = config.RESULTS_DIR / f"{disease}_results{'_dedup' if args.dedup else ''}.csv"
        if out.exists():  # on remplace seulement les lignes recalculées
            old = pd.read_csv(out)
            key = ["dataset", "model", "seed", "test_size"]
            old = old.merge(part[key], on=key, how="left", indicator=True)
            old = old[old["_merge"] == "left_only"].drop(columns="_merge")
            part = pd.concat([old, part], ignore_index=True)
        part.to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
