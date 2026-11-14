"""Sensibilité au découpage train/test : moyenne ± écart-type sur plusieurs seeds,
et l'article est-il dans l'intervalle [min, max] de nos seeds ?

    python seed_analysis.py heart

Sortie : results/<maladie>_seeds.md
"""
import argparse

import pandas as pd

from src import config
from src.evaluate import add_article_comparison
from compare_article import SHORT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("disease")
    p.add_argument("--metric", default="accuracy")
    p.add_argument("--suffix", default="", help="ex. _no-time, _dedup")
    a = p.parse_args()
    res = add_article_comparison(pd.read_csv(config.RESULTS_DIR / f"{a.disease}_results{a.suffix}.csv"))
    m = a.metric
    g = res.groupby(["dataset", "model"]).agg(
        mean=(f"{m}_a", "mean"), std=(f"{m}_a", "std"), lo=(f"{m}_a", "min"),
        hi=(f"{m}_a", "max"), n=(f"{m}_a", "size"), article=(f"article_{m}", "first")).reset_index()
    g["in_range"] = (g["article"] >= g["lo"] - 0.005) & (g["article"] <= g["hi"] + 0.005)
    g["gap"] = g["mean"] - g["article"]
    seeds = sorted(res["seed"].unique())

    lines = [f"# {a.disease} : {m} sur {len(seeds)} seeds {seeds} (split 80/20)\n",
             "Case : moyenne ± écart-type (article). ✓ = article dans [min, max] de nos seeds.\n"]
    models = [x for x in config.MODELS if x in set(g["model"])]
    lines.append("| | " + " | ".join(SHORT[x] for x in models) + " |")
    lines.append("|---" * (len(models) + 1) + "|")
    for ds, part in g.groupby("dataset"):
        part = part.set_index("model")
        cells = [f"{part.at[x, 'mean'] * 100:.0f}±{part.at[x, 'std'] * 100:.0f} "
                 f"({part.at[x, 'article'] * 100:.0f}){' ✓' if part.at[x, 'in_range'] else ''}"
                 for x in models]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines.append("\n## Synthèse par dataset\n")
    lines.append("| Dataset | écart-type moyen | écart moyen à l'article | article dans nos seeds |")
    lines.append("|---|---|---|---|")
    for ds, part in g.groupby("dataset"):
        lines.append(f"| {ds} | {part['std'].mean() * 100:.1f} pts | "
                     f"{part['gap'].mean() * 100:+.1f} pts | "
                     f"{part['in_range'].sum()}/{len(part)} |")
    lines.append(f"\nTotal : article dans l'intervalle de nos seeds pour "
                 f"{g['in_range'].sum()}/{len(g)} couples (dataset, modèle).")

    out = config.RESULTS_DIR / f"{a.disease}_seeds{a.suffix}_{m}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
