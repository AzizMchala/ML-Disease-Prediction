"""Diabète : D8 à D11 (responsable : Omar).

Valeurs "expected" = tableau 2 de l'article : (lignes, positifs, négatifs).
Placer les CSV dans data/raw/diabetes/ (nom D8.csv, D9.csv...).
"""
import pandas as pd


def _transform_d8(df: pd.DataFrame) -> pd.DataFrame:
    # Variables Yes/No et Gender en texte : encodage automatique par LabelEncoder dans preprocessing.
    # Cible : 'class' ('Positive' -> 1, 'Negative' -> 0)
    df = df.copy()
    df["class"] = (df["class"].astype(str).str.strip().str.lower() == "positive").astype(int)
    return df


def _transform_d9(df: pd.DataFrame) -> pd.DataFrame:
    # Cible déjà 0/1 ('Outcome')
    return df.copy()


def _transform_d10(df: pd.DataFrame) -> pd.DataFrame:
    # Nettoyage des fautes de frappe, espaces et valeurs manquantes dans les chaînes
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Correction de BPLevel ('Low' -> 'low', 'High' -> 'high')
    if "BPLevel" in df.columns:
        df["BPLevel"] = df["BPLevel"].str.lower()

    # Correction de RegularMedicine ('o' -> 'no')
    if "RegularMedicine" in df.columns:
        df["RegularMedicine"] = df["RegularMedicine"].replace({"o": "no", "O": "no"})

    # Correction de Pdiabetes ('0' -> 'no')
    if "Pdiabetes" in df.columns:
        df["Pdiabetes"] = df["Pdiabetes"].replace({"0": "no", "nan": "no"}).fillna("no")

    # Pregancies : NaN chez les hommes -> 0
    if "Pregancies" in df.columns:
        df["Pregancies"] = df["Pregancies"].fillna(0)

    # Cible 'Diabetic' : 'yes' (266) + 'no' (684) + ' no' (1) + NaN (1).
    # L'article a 267 positifs et 685 négatifs (total 952).
    # ' no' -> 'no' (685), NaN -> 'yes' (267).
    df["Diabetic"] = df["Diabetic"].fillna("yes").astype(str).str.strip().str.lower()
    df["Diabetic"] = (df["Diabetic"] == "yes").astype(int)
    return df


def _transform_d11(df: pd.DataFrame) -> pd.DataFrame:
    # Cible 'Class' ou colonne 19 : déjà 0/1 (1 = retinopathy, 0 = normal)
    df = df.copy()
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
    elif "19" in df.columns:
        df["Class"] = df["19"].astype(int)
        df = df.drop(columns=["19"])
    return df


DATASETS = {
    "D8": {
        "disease": "diabetes",
        "name": "Early Stage Diabetes Risk Prediction",
        "url": "https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset",
        "files": ["D8.csv", "diabetes_data_upload.csv"],
        "target": "class",
        "expected": (520, 320, 200),
        "transform": _transform_d8,
    },
    "D9": {
        "disease": "diabetes",
        "name": "Diabetes (KNN, Mahgoub) / Pima Indians",
        "url": "https://www.kaggle.com/abdallamahgoub/diabetes",
        "files": ["D9.csv", "diabetes.csv"],
        "target": "Outcome",
        "expected": (768, 268, 500),
        "transform": _transform_d9,
    },
    "D10": {
        "disease": "diabetes",
        "name": "Diabetes Dataset 2019",
        "url": "https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019",
        "files": ["D10.csv", "diabetes_dataset__2019.csv"],
        "target": "Diabetic",
        "expected": (952, 267, 685),
        "transform": _transform_d10,
    },
    "D11": {
        "disease": "diabetes",
        "name": "Diabetic Retinopathy Debrecen",
        "url": "https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen",
        "files": ["D11.csv", "messidor_features.csv"],
        "target": "Class",
        "expected": (1151, 611, 540),
        "transform": _transform_d11,
    },
}
