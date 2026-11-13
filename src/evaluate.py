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


def add_article_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Ajoute pour chaque métrique la valeur de l'article et l'écart (nous - article)."""
    ref = load_article_reference().pivot_table(
        index=["dataset", "model"], columns="metric", values="value").reset_index()
    ref.columns = ["dataset", "model"] + [f"article_{m}" for m in ref.columns[2:]]
    out = results.merge(ref, on=["dataset", "model"], how="left")
    for m in config.METRICS:
        if f"article_{m}" in out:
            out[f"delta_{m}"] = out[m] - out[f"article_{m}"]
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
