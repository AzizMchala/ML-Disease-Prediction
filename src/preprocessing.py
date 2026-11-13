"""Préparation des données (phase 3 de CRISP-DM), identique pour tous les datasets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config


def clean(df: pd.DataFrame, target: str, drop_duplicates: bool = False) -> pd.DataFrame:
    """Nettoyage minimal et traçable.

    drop_duplicates=False reproduit l'article (qui ne mentionne pas la suppression
    des doublons). Lancer aussi avec True pour l'analyse critique (fuite de données).
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=[target])
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode(df: pd.DataFrame, target: str):
    """LabelEncoder sur les colonnes texte (article : "labelling categorical features")."""
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Médiane calculée sur le train uniquement (pas de fuite vers le test)."""
    med = X_train.median(numeric_only=True)
    return X_train.fillna(med), X_test.fillna(med)


def prepare(df: pd.DataFrame, target: str, drop_duplicates: bool = False,
            test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Retourne X_train, X_test, y_train, y_test.

    La normalisation (StandardScaler) n'est PAS faite ici : elle est dans le
    Pipeline du modèle, donc ajustée sur chaque pli d'entraînement du GridSearchCV.
    """
    df = clean(df, target, drop_duplicates)
    df, _ = encode(df, target)
    X = df.drop(columns=[target])
    # Classes renumérotées 0..k-1 (exigé par XGBoost). Chaque loader doit déjà
    # mettre la classe positive (malade) à 1 pour un problème binaire.
    y = LabelEncoder().fit_transform(df[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_test = impute(X_train, X_test)
    return X_train, X_test, y_train, y_test
