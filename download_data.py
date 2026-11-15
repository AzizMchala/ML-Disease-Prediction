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


DIRECT_SOURCES = {
    "D8": {
        "url": "https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip",
        "type": "zip_csv",
        "zip_member": "diabetes_data_upload.csv",
    },
    "D9": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "type": "csv",
    },
    "D10": {
        "url": "https://raw.githubusercontent.com/bernardtse/diabetes-prediction/master/resources/diabetes_dataset__2019.csv",
        "type": "csv",
    },
    "D11": {
        "url": "https://archive.ics.uci.edu/static/public/329/diabetic+retinopathy+debrecen.zip",
        "type": "arff_zip",
        "zip_member": "messidor_features.arff",
    },
}


def kaggle_slug(url: str) -> str:
    return url.split("kaggle.com/datasets/")[1].strip("/")


def download(dataset_id: str):
    spec = REGISTRY[dataset_id]
    out = config.DATA_DIR / spec["disease"] / spec["files"][0]
    if out.exists():
        print(f"[{dataset_id}] déjà présent : {out}")
        df = pd.read_csv(out)
        df.columns = [c.strip() for c in df.columns]
        if "transform" in spec:
            df = spec["transform"](df)
        check_against_article(dataset_id, df, spec["target"])
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    req_headers = {"User-Agent": "Mozilla/5.0"}

    if dataset_id in DIRECT_SOURCES:
        ds_info = DIRECT_SOURCES[dataset_id]
        req = urllib.request.Request(ds_info["url"], headers=req_headers)
        with urllib.request.urlopen(req) as r:
            raw_data = r.read()

        if ds_info["type"] == "zip_csv":
            z = zipfile.ZipFile(io.BytesIO(raw_data))
            out.write_bytes(z.read(ds_info["zip_member"]))
        elif ds_info["type"] == "csv":
            out.write_bytes(raw_data)
        elif ds_info["type"] == "arff_zip":
            z = zipfile.ZipFile(io.BytesIO(raw_data))
            content = z.read(ds_info["zip_member"]).decode("utf-8", errors="ignore")
            lines = [l for l in content.splitlines() if l.strip() and not l.startswith("%")]
            data_idx = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data")
            cols = [f"feat_{i}" for i in range(19)] + ["Class"]
            df_arff = pd.read_csv(io.StringIO("\n".join(lines[data_idx + 1:])), header=None, names=cols)
            df_arff.to_csv(out, index=False)
        print(f"[{dataset_id}] téléchargé -> {out}")
    elif "kaggle.com/datasets/" in spec["url"]:
        api = f"https://www.kaggle.com/api/v1/datasets/download/{kaggle_slug(spec['url'])}"
        req = urllib.request.Request(api, headers=req_headers)
        with urllib.request.urlopen(req) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        member = spec.get("zip_member", csvs[0])
        out.write_bytes(z.read(member))
        print(f"[{dataset_id}] {member} -> {out}")
    else:
        print(f"[{dataset_id}] source non Kaggle, télécharger à la main : {spec['url']}")
        return

    df = pd.read_csv(out)
    df.columns = [c.strip() for c in df.columns]
    if "transform" in spec:
        df = spec["transform"](df)
    check_against_article(dataset_id, df, spec["target"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    for ds in (REGISTRY if a.all else a.datasets):
        download(ds)
