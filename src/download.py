"""Download the RECON-SL dataset (LIRNEasia) from Kaggle.

Usage:
    python -m src.download --out data/raw/recon_sl.zip
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/lirneasia/sri-lankan-residential-electricity-consumption"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/recon_sl.zip")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"Downloading {URL}", flush=True)
    with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=8 << 20)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise SystemExit("Download did not return a zip file (Kaggle may require login from this network).")
    tmp.replace(out)
    print(f"Saved {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
