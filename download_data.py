"""Télécharge les datasets publics Kaggle déclarés dans src/datasets/*.py.

    python download_data.py D1 D2 D3 D4
    python download_data.py --all

Les CSV ne sont pas versionnés (licences Kaggle) : chacun les retélécharge avec ce script.
Le premier fichier de "files" est le nom local attendu (D1.csv...).
"""
import argparse
import io
import urllib.request
import zipfile

from src import config
from src.datasets import REGISTRY, check_against_article

import pandas as pd


def kaggle_slug(url: str) -> str:
    return url.split("kaggle.com/datasets/")[1].strip("/")


def download(dataset_id: str):
    spec = REGISTRY[dataset_id]
    out = config.DATA_DIR / spec["disease"] / spec["files"][0]
    if out.exists():
        print(f"[{dataset_id}] déjà présent : {out}")
        return
    if "kaggle.com/datasets/" not in spec["url"]:
        print(f"[{dataset_id}] source non Kaggle, télécharger à la main : {spec['url']}")
        return
    api = f"https://www.kaggle.com/api/v1/datasets/download/{kaggle_slug(spec['url'])}"
    with urllib.request.urlopen(api) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    member = spec.get("zip_member", csvs[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(z.read(member))
    df = pd.read_csv(out)
    df.columns = [c.strip() for c in df.columns]
    if "transform" in spec:
        df = spec["transform"](df)
    print(f"[{dataset_id}] {member} -> {out}")
    check_against_article(dataset_id, df, spec["target"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    for ds in (REGISTRY if a.all else a.datasets):
        download(ds)
