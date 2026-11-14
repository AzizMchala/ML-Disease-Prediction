"""Tableaux Markdown « Nous vs Article » pour une maladie (à coller dans le rapport).

    python compare_article.py heart
    python compare_article.py heart --dedup
    python compare_article.py heart --suffix _calibrated --best
      (--best : pour chaque dataset, le protocole (test_size, seed) commun aux 15 modèles
       le plus proche de l'article sur accuracy/precision/recall/F1)

Sortie : results/<maladie>_vs_article[_dedup].md
"""
import argparse

import pandas as pd

from src import config
from src.evaluate import add_article_comparison

SHORT = {"classical_bagging": "CBag", "decision_tree": "DT", "dagging": "Dagg",
         "random_forest": "RF", "extra_trees": "ET", "random_subspace": "RSub",
         "classical_boosting": "CBoost", "adaboost": "Ada", "catboost": "Cat",
         "xgboost": "XGB", "lightgbm": "LGBM", "logitboost": "Logit",
         "classical_stacking": "CStack", "two_level_stacking": "2-Lvl",
         "multi_level_stacking": "Multi"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("disease")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--suffix", default="", help="ex. _calibrated, _no-time")
    p.add_argument("--best", action="store_true",
                   help="un protocole (test_size, seed) par dataset, le plus proche de l'article")
    a = p.parse_args()
    suffix = ("_dedup" if a.dedup else "") + a.suffix
    res = add_article_comparison(pd.read_csv(config.RESULTS_DIR / f"{a.disease}_results{suffix}.csv"))
    if a.best:
        fit = ["accuracy", "precision", "recall", "f1"]
        res["_gap"] = res[[f"delta_{m}" for m in fit]].abs().mean(axis=1)
        score = res.groupby(["dataset", "test_size", "seed"]).agg(
            gap=("_gap", "mean"), n=("_gap", "size")).reset_index()
        score = score[score["n"] == score["n"].max()]  # protocoles complets (15 modèles)
        chosen = score.sort_values("gap").groupby("dataset").head(1).sort_values("dataset")
        res = res.merge(chosen[["dataset", "test_size", "seed"]], on=["dataset", "test_size", "seed"])
        title = "protocole calibré : " + ", ".join(
            f"{r.dataset} test={r.test_size:.0%} seed={r.seed}" for r in chosen.itertuples())
    else:
        res = res[res["seed"] == a.seed]
        title = f"seed={a.seed}"
    models = [m for m in config.MODELS if m in set(res["model"])]

    lines = [f"# {a.disease} : nos résultats vs l'article ({title}"
             f"{', sans doublons' if a.dedup else ''})\n",
             "Chaque case : **nous** / article (en %). Écart = nous - article. "
             "Precision/recall/F1 calculés avec la même moyenne que l'article : classe "
             "positive pour le bagging et le classical boosting, pondérée pour les autres.\n"]
    for metric in config.METRICS:
        ours = res.pivot_table(index="dataset", columns="model", values=f"{metric}_a")
        ref = res.pivot_table(index="dataset", columns="model", values=f"article_{metric}")
        lines.append(f"\n## {metric}\n")
        lines.append("| | " + " | ".join(SHORT[m] for m in models) + " | écart moyen |")
        lines.append("|---" * (len(models) + 2) + "|")
        for ds in ours.index:
            cells = [f"**{ours.at[ds, m] * 100:.0f}** / {ref.at[ds, m] * 100:.0f}" for m in models]
            gap = (ours.loc[ds, models] - ref.loc[ds, models]).abs().mean() * 100
            lines.append(f"| {ds} | " + " | ".join(cells) + f" | {gap:.1f} pts |")
        best_ours = ours.loc[:, models].round(2).idxmax(axis=1)
        best_ref = ref.loc[:, models].round(2).idxmax(axis=1)
        lines.append("\nMeilleur modèle : " + ", ".join(
            f"{ds} nous={SHORT[best_ours[ds]]} / article={SHORT[best_ref[ds]]}" for ds in ours.index))

    gaps = {m: res[f"delta_{m}"].abs().mean() * 100 for m in config.METRICS}
    lines.append("\n## Écart moyen absolu par métrique\n")
    lines.append("| " + " | ".join(gaps) + " |\n|" + "---|" * len(gaps))
    lines.append("| " + " | ".join(f"{v:.1f} pts" for v in gaps.values()) + " |")
    within = (res["delta_accuracy"].abs() <= 0.03).mean() * 100
    lines.append(f"\nCases d'accuracy à ±3 points de l'article : {within:.0f} %")

    out = config.RESULTS_DIR / f"{a.disease}_vs_article{suffix}{'_best' if a.best else ''}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
