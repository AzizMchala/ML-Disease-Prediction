"""MODÈLE à copier : src/datasets/<maladie>.py (liver, diabetes, kidney, skin).

Règles :
- "expected" = (lignes, positifs, négatifs) du tableau 2 de l'article
  (pour la peau, multiclasse : (lignes, None, None)).
- La classe positive (malade) doit valoir 1. Si la cible est du texte ("ckd",
  "yes", 2/1...), la convertir dans "transform".
"""


def _transform_example(df):
    # Exemple : cible texte -> 1 = malade, 0 = sain
    df["classification"] = (df["classification"].str.strip() == "ckd").astype(int)
    return df


DATASETS = {
    "D12": {
        "disease": "kidney",
        "name": "Chronic Kidney Dataset",
        "url": "https://www.kaggle.com/datasets/mansoordaku/ckdisease",
        "files": ["D12.csv", "kidney_disease.csv"],
        "target": "classification",
        # L'article écrit 150/250 mais le fichier contient 250 "ckd" : si le check
        # affiche DIFFÉRENT, noter dans le rapport quelle classe l'article appelle positive.
        "expected": (400, 150, 250),
        "transform": _transform_example,
    },
}
