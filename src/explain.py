"""SHAP, LIME and partial dependence for the Rotation Forest solar detector.

The trees of a Rotation Forest split on rotated axes (mixtures of features), so tree-SHAP would explain those
mixtures, not the features a person can read. All explanations here are therefore model-agnostic and work on
the original 64 features:

* SHAP (permutation explainer): global importance and per-day contributions for the whole forest.
* LIME: a local linear surrogate around single days.
* Partial dependence (PDP + ICE): how P(solar) moves with one feature when the others are held as observed.

Usage:
    python -m src.explain
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

from src.features import HOURS, make_features

PDP_FEATURES = ["daytime_min_share", "midday_to_evening", "daytime_zero_hours", "bowl_depth"]


def shap_explainer(detector, background):
    """Permutation SHAP for the whole Rotation Forest; values are in probability units (P(solar))."""
    cols = list(background.columns)
    f = lambda x: detector.day_model.predict_proba(pd.DataFrame(x, columns=cols))[:, 1]
    return shap.PermutationExplainer(f, background)


def shap_values(explainer, X):
    return explainer(X, max_evals=2 * X.shape[1] + 1, silent=True)


def lime_explainer(X_train, seed=42):
    return LimeTabularExplainer(X_train.values, feature_names=list(X_train.columns), class_names=["no solar", "solar"],
                                discretize_continuous=True, random_state=seed, mode="classification")


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/household_days.parquet")
    ap.add_argument("--model", default="models/solar_detector.joblib")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/explain")
    ap.add_argument("--sample", type=int, default=400, help="test days explained with SHAP")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):  # drop plots of earlier models
        old.unlink()

    det = joblib.load(a.model)
    hd = pd.read_parquet(a.data)
    split = json.load(open(Path(a.reports) / "split.json"))
    tr = hd[hd.hh.isin(split["train_households"])]
    te = hd[hd.hh.isin(split["test_households"])].reset_index(drop=True)
    Xtr = make_features(tr[HOURS])[det.feature_names]
    Xte = make_features(te[HOURS])[det.feature_names]
    rng = np.random.default_rng(0)
    Xs = Xte.iloc[rng.choice(len(Xte), min(a.sample, len(Xte)), replace=False)]

    # Global SHAP for the whole forest.
    bg = shap.sample(Xtr, 50, random_state=0)
    explainer = shap_explainer(det, bg)
    sv = shap_values(explainer, Xs)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP summary — Rotation Forest (test days)")
    save(out / "shap_beeswarm.png")
    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP| — Rotation Forest")
    save(out / "shap_bar.png")
    for feat in ["daytime_min_share", "midday_to_evening"]:
        plt.figure()
        shap.plots.scatter(sv[:, feat], color=sv[:, "total_kwh"], show=False)
        save(out / f"shap_dependence_{feat}.png")
    importance = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns).sort_values(ascending=False)
    importance.to_csv(out / "shap_importance.csv", header=["mean_abs_shap"])

    # Partial dependence with individual curves (ICE) on a sample of training days.
    pdp_X = Xtr.sample(1000, random_state=0).astype(float)  # PDP refuses integer columns (daytime_zero_hours)
    fig, ax = plt.subplots(1, len(PDP_FEATURES), figsize=(4 * len(PDP_FEATURES), 3.6))
    PartialDependenceDisplay.from_estimator(det.day_model, pdp_X, PDP_FEATURES, kind="both", subsample=60,
                                            grid_resolution=25, percentiles=(0.02, 0.98), random_state=0, ax=ax,
                                            ice_lines_kw={"alpha": 0.15}, pd_line_kw={"color": "#E8A33D", "lw": 2.5})
    fig.suptitle("Partial dependence of P(solar) — Rotation Forest (orange = average, grey = single days)")
    save(out / "pdp.png")

    # Local explanations: a confident solar day, a confident non-solar day, and the most uncertain day.
    p = det.day_model.predict_proba(Xte)[:, 1]
    picks = {"solar_day": int(np.argmax(np.where(te.solar_installed.values == 1, p, -1))),
             "non_solar_day": int(np.argmin(np.where(te.solar_installed.values == 0, p, 2))),
             "uncertain_day": int(np.argmin(np.abs(p - 0.5)))}
    sv_loc = shap_values(explainer, Xte.iloc[list(picks.values())])
    lime = lime_explainer(Xtr.sample(20000, random_state=0))
    lime_rows = []
    for k, (name, i) in enumerate(picks.items()):
        plt.figure()
        shap.plots.waterfall(sv_loc[k], max_display=12, show=False)
        plt.title(f"SHAP waterfall — {name} (true={te.solar_installed[i]}, p={p[i]:.3f})")
        save(out / f"shap_waterfall_{name}.png")
        exp = lime.explain_instance(Xte.iloc[i].values, lambda x: det.day_model.predict_proba(
            pd.DataFrame(x, columns=Xte.columns)), num_features=10, num_samples=3000)
        fig = exp.as_pyplot_figure(label=1)
        fig.suptitle(f"LIME — {name} (true={te.solar_installed[i]}, p={p[i]:.3f})")
        save(out / f"lime_{name}.png")
        for rule, weight in exp.as_list(label=1):
            lime_rows.append({"case": name, "hh": te.hh[i], "day": str(te.day[i].date()), "true": int(te.solar_installed[i]),
                              "p_solar": float(p[i]), "rule": rule, "weight": weight})
    pd.DataFrame(lime_rows).to_csv(out / "lime_explanations.csv", index=False)
    print("Top SHAP features (Rotation Forest):\n", importance.head(10).round(4))


if __name__ == "__main__":
    main()
