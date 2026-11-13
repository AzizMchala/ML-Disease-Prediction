"""Intégration finale : fusionne results/*_results.csv de toute l'équipe et produit
les équivalents de nos tableaux 3 à 16 de l'article.

    python integrate.py            # seed par défaut (config.SEED)
    python integrate.py --seed 0

Sortie : results/summary/ (un CSV par tableau) + affichage console.
"""
import argparse

import pandas as pd

from src import config
from src.evaluate import best_count, load_article_reference, rpi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    files = sorted(f for f in config.RESULTS_DIR.glob("*_results.csv"))
    if not files:
        raise SystemExit("Aucun fichier results/*_results.csv : lancer run_experiments.py d'abord.")
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    res = res[res["seed"] == args.seed]
    out_dir = config.RESULTS_DIR / "summary"
    out_dir.mkdir(exist_ok=True)

    order = lambda t: t.reindex(columns=[m for m in config.MODELS if m in t.columns])
    for metric in config.METRICS:
        ours = order(res.pivot_table(index="dataset", columns="model", values=metric)).round(2)
        ours.to_csv(out_dir / f"table_{metric}_ours.csv")
        if f"delta_{metric}" in res:
            delta = order(res.pivot_table(index="dataset", columns="model",
                                          values=f"delta_{metric}")).round(2)
            delta.to_csv(out_dir / f"table_{metric}_delta_vs_article.csv")
            print(f"\n=== {metric} : écart moyen absolu vs article = "
                  f"{delta.abs().stack().mean():.3f}")

    # Tableau 15 (RPI) et 16 (best count), nous vs article
    ref = load_article_reference()
    ref = ref[ref["dataset"].isin(res["dataset"].unique())]
    t15 = pd.DataFrame({m: rpi(res, m) for m in config.METRICS}).T
    t16 = pd.DataFrame({m: best_count(res, m) for m in config.METRICS}).T
    t16.loc["count"] = t16.sum()
    a16 = pd.DataFrame({m: best_count(ref[ref.metric == m], "value") for m in config.METRICS}).T
    a16.loc["count"] = a16.sum()
    order(t15).to_csv(out_dir / "table15_rpi_ours.csv")
    order(t16).to_csv(out_dir / "table16_best_count_ours.csv")
    order(a16).to_csv(out_dir / "table16_best_count_article_same_datasets.csv")
    print("\n=== Best count (nous) ===\n", order(t16).loc["count"].sort_values(ascending=False))
    print("\n=== Best count (article, mêmes datasets) ===\n",
          order(a16).loc["count"].sort_values(ascending=False))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
