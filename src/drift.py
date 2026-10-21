"""Data-drift monitoring with the Population Stability Index (PSI).

At training time `reference_stats` stores decile bins of key features for the training households.
Later, `psi_report` compares new household-days against them:
    PSI < 0.1 stable · 0.1-0.2 moderate shift · > 0.2 significant drift (consider retraining).

Usage:
    python -m src.drift --data data/processed/household_days.parquet --reference models/reference_stats.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.features import HOURS, make_features

DRIFT_FEATURES = ["total_kwh", "daytime_min_share", "daytime_min_to_night_min", "midday_to_evening",
                  "evening_kwh", "night_kwh", "peak_hour"]
EPS = 1e-6


def reference_stats(X: pd.DataFrame) -> dict:
    stats = {}
    for f in DRIFT_FEATURES:
        edges = np.unique(np.quantile(X[f], np.linspace(0.1, 0.9, 9)))
        counts = np.bincount(np.searchsorted(edges, X[f], side="right"), minlength=len(edges) + 1)
        stats[f] = {"edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}
    return {"n_days": int(len(X)), "features": stats}


def psi(expected, actual) -> float:
    e, a = np.clip(np.asarray(expected), EPS, None), np.clip(np.asarray(actual), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(X: pd.DataFrame, reference: dict) -> pd.DataFrame:
    rows = []
    for f, s in reference["features"].items():
        edges = np.asarray(s["edges"])
        counts = np.bincount(np.searchsorted(edges, X[f], side="right"), minlength=len(edges) + 1)
        value = psi(s["proportions"], counts / max(counts.sum(), 1))
        rows.append({"feature": f, "psi": round(value, 4),
                     "status": "drift" if value > 0.2 else "moderate" if value > 0.1 else "stable"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/household_days.parquet")
    ap.add_argument("--reference", default="models/reference_stats.json")
    ap.add_argument("--out", default="reports/drift.json")
    a = ap.parse_args()
    hd = pd.read_parquet(a.data)
    rep = psi_report(make_features(hd[HOURS]), json.load(open(a.reference)))
    print(rep.to_string(index=False))
    json.dump(rep.to_dict(orient="records"), open(a.out, "w"), indent=2)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        table = "| feature | PSI | status |\n|---|---|---|\n" + "".join(
            f"| {r.feature} | {r.psi:.4f} | {r.status} |\n" for r in rep.itertuples())
        with open(summary, "a") as fh:
            fh.write("### Data drift (PSI vs. training data)\n\n" + table + "\n")


if __name__ == "__main__":
    main()
