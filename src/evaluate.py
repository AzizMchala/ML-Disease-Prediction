"""Évaluation (phase 5 de CRISP-DM) : 6 métriques de l'article, matrice de confusion,
RPI (section 2.2) et nombre de victoires (« best count »)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize

from . import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Binaire : classe positive = 1. Multiclasse (peau) : moyenne pondérée,
    AUC un-contre-tous, AUPRC macro."""
    classes = np.unique(y_true)
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    for k, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        out[f"{k}_w"] = fn(y_true, y_pred, average="weighted", zero_division=0)
    if binary:
        out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        out["auprc"] = average_precision_score(y_true, y_proba[:, 1])
    else:
        Y = label_binarize(y_true, classes=np.arange(y_proba.shape[1]))
        out["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                                   labels=np.arange(y_proba.shape[1]))
        out["auprc"] = average_precision_score(Y, y_proba, average="macro")
    return out


def save_confusion_matrix(y_true, y_pred, title: str, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def load_article_reference() -> pd.DataFrame:
    """Valeurs des tableaux 3, 5, 7, 9, 11 et 13 de l'article (format long)."""
    return pd.read_csv(config.ARTICLE_REF)


def weighted_from_confusion(cm) -> dict:
    """precision / recall / F1 pondérés à partir d'une matrice de confusion (lignes = vrai)."""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(axis=0))
        rec = np.nan_to_num(np.diag(cm) / support)
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    w = support / support.sum()
    return {"precision_w": (prec * w).sum(), "recall_w": (rec * w).sum(), "f1_w": (f1 * w).sum()}


def ensure_weighted(results: pd.DataFrame) -> pd.DataFrame:
    """Complète les colonnes *_w des anciens résultats grâce aux matrices de confusion."""
    if "precision_w" in results and results["precision_w"].notna().all():
        return results
    import ast
    filled = results["confusion_matrix"].map(
        lambda s: weighted_from_confusion(ast.literal_eval(s) if isinstance(s, str) else s))
    w = pd.DataFrame(list(filled), index=results.index)
    for c in w:
        results[c] = results[c].fillna(w[c]) if c in results else w[c]
    return results


def article_view(results: pd.DataFrame) -> pd.DataFrame:
    """Colonnes *_a : la métrique calculée comme l'article (binaire ou pondérée selon le modèle)."""
    results = ensure_weighted(results.copy())
    weighted = results["model"].isin(config.ARTICLE_WEIGHTED_MODELS)
    for m in ["precision", "recall", "f1"]:
        results[f"{m}_a"] = np.where(weighted, results[f"{m}_w"], results[m])
    for m in ["accuracy", "auc", "auprc"]:
        results[f"{m}_a"] = results[m]
    return results


def add_article_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Ajoute pour chaque métrique la valeur de l'article et l'écart (nous - article),
    en utilisant la même moyenne que l'article (colonnes *_a)."""
    ref = load_article_reference().pivot_table(
        index=["dataset", "model"], columns="metric", values="value").reset_index()
    ref.columns = ["dataset", "model"] + [f"article_{m}" for m in ref.columns[2:]]
    results = results.drop(columns=[c for c in results if c.startswith(("article_", "delta_"))])
    out = article_view(results).merge(ref, on=["dataset", "model"], how="left")
    for m in config.METRICS:
        if f"article_{m}" in out:
            out[f"delta_{m}"] = out[f"{m}_a"] - out[f"article_{m}"]
    return out


def rpi(results: pd.DataFrame, metric: str) -> pd.Series:
    """Relative Performance Index (article, section 2.2) :
    RPI = somme_i (a_i - a*_i) / d, a*_i = minimum des modèles sur le dataset i."""
    table = results.pivot_table(index="dataset", columns="model", values=metric)
    return table.sub(table.min(axis=1), axis=0).mean(axis=0).sort_values(ascending=False)


def best_count(results: pd.DataFrame, metric: str, decimals: int = 2) -> pd.Series:
    """Nombre de datasets où chaque modèle atteint le meilleur score (ex æquo compris),
    arrondi au pourcent comme dans l'article."""
    table = results.pivot_table(index="dataset", columns="model", values=metric).round(decimals)
    return table.eq(table.max(axis=1), axis=0).sum().sort_values(ascending=False)
