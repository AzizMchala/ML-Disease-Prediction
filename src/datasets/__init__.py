"""Registre commun des 16 datasets. Chaque membre déclare les siens dans son module
(heart.py, liver.py, diabetes.py, kidney.py, skin.py) en suivant _template.py."""
import importlib

import pandas as pd

from .. import config

_MODULES = ["heart", "liver", "diabetes", "kidney", "skin"]

REGISTRY = {}
for _name in _MODULES:
    try:
        REGISTRY.update(importlib.import_module(f".{_name}", __name__).DATASETS)
    except ModuleNotFoundError:
        pass  # module pas encore écrit par son responsable


def load(dataset_id: str, check: bool = True):
    """Charge un dataset brut. Retourne (DataFrame, nom de la colonne cible)."""
    spec = REGISTRY[dataset_id]
    folder = config.DATA_DIR / spec["disease"]
    path = next((folder / f for f in spec["files"] if (folder / f).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"{dataset_id}: placer l'un de {spec['files']} dans {folder} (source : {spec['url']})")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "transform" in spec:
        df = spec["transform"](df)
    target = spec["target"]
    if check:
        check_against_article(dataset_id, df, target)
    return df, target


def check_against_article(dataset_id: str, df: pd.DataFrame, target: str):
    """Compare la taille et la répartition des classes au tableau 2 de l'article."""
    spec = REGISTRY[dataset_id]
    rows, pos, neg = spec["expected"]
    counts = df[target].value_counts()
    got = (len(df), counts.get(1), counts.get(0))
    ok = got == (rows, pos, neg) if pos is not None else len(df) == rows
    status = "OK" if ok else "DIFFÉRENT de l'article"
    print(f"[{dataset_id}] lignes={got[0]} pos={got[1]} neg={got[2]} | "
          f"article: lignes={rows} pos={pos} neg={neg} -> {status}")
    return ok
