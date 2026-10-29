"""SHAP and LIME explanations for the solar detector.

* SHAP (TreeExplainer) explains a tuned gradient-boosting base learner globally and per day (tree SHAP is exact
  and fast).
* SHAP (permutation) explains the full stacking ensemble on a sample, to check it relies on the same features.
* LIME explains individual ensemble predictions with a local linear surrogate.

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

from src.features import HOURS, make_features


def base_learner(detector, name="lightgbm"):
    stack = detector.day_model
    names = [n for n, _ in stack.estimators]
    return stack.estimators_[names.index(name)]


def tree_shap(model, X):
    explainer = shap.TreeExplainer(model)
    values = explainer(X)
    if values.values.ndim == 3:  # some versions return one slice per class
        values = values[..., 1]
    return explainer, values


def stack_weights(detector):
    stack = detector.day_model
    names = [n for n, _ in stack.estimators]
    coef = stack.final_estimator_.coef_.ravel()
    # predict_proba stacking gives 1 column per estimator for binary problems
    return pd.Series(coef[: len(names)], index=names).sort_values(ascending=False)


def lime_explainer(X_train, seed=42):
    return LimeTabularExplainer(X_train.values, feature_names=list(X_train.columns), class_names=["no solar", "solar"],
                                discretize_continuous=True, random_state=seed, mode="classification")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/household_days.parquet")
    ap.add_argument("--model", default="models/solar_detector.joblib")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/explain")
    ap.add_argument("--sample", type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    det = joblib.load(a.model)
    hd = pd.read_parquet(a.data)
    split = json.load(open(Path(a.reports) / "split.json"))
    tr = hd[hd.hh.isin(split["train_households"])]
    te = hd[hd.hh.isin(split["test_households"])].reset_index(drop=True)
    Xtr = make_features(tr[HOURS])[det.feature_names]
    Xte = make_features(te[HOURS])[det.feature_names]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), min(a.sample, len(Xte)), replace=False)
    Xs = Xte.iloc[idx]

    w = stack_weights(det)
    w.to_csv(out / "stacking_weights.csv")
    print("Stacking meta-learner weights:\n", w.round(3))

    # Global SHAP on the LightGBM base learner.
    lgbm = base_learner(det, "lightgbm")
    _, sv = tree_shap(lgbm, Xs)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP summary — LightGBM base learner (test days)")
    plt.tight_layout(); plt.savefig(out / "shap_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close("all")
    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP| — LightGBM base learner")
    plt.tight_layout(); plt.savefig(out / "shap_bar.png", dpi=130, bbox_inches="tight"); plt.close("all")
    for feat in ["daytime_min_share", "midday_to_evening"]:
        if feat in Xs.columns:
            plt.figure()
            shap.plots.scatter(sv[:, feat], color=sv[:, "total_kwh"], show=False)
            plt.tight_layout(); plt.savefig(out / f"shap_dependence_{feat}.png", dpi=130, bbox_inches="tight")
            plt.close("all")
    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns).sort_values(ascending=False)
    mean_abs.to_csv(out / "shap_importance_lightgbm.csv")

    # SHAP on the full ensemble (model-agnostic, smaller sample) to confirm the same drivers.
    f = lambda x: det.day_model.predict_proba(pd.DataFrame(x, columns=Xs.columns))[:, 1]
    bg = shap.sample(Xtr, 100, random_state=0)
    ens = shap.PermutationExplainer(f, bg)
    sv_ens = ens(Xs.iloc[:200], max_evals=2 * Xs.shape[1] + 1)
    ens_imp = pd.Series(np.abs(sv_ens.values).mean(0), index=Xs.columns).sort_values(ascending=False)
    ens_imp.to_csv(out / "shap_importance_ensemble.csv")
    plt.figure()
    shap.plots.bar(sv_ens, max_display=15, show=False)
    plt.title("Mean |SHAP| — full stacking ensemble (200 test days)")
    plt.tight_layout(); plt.savefig(out / "shap_bar_ensemble.png", dpi=130, bbox_inches="tight"); plt.close("all")

    # Local explanations: a confident solar day, a confident non-solar day, and the most uncertain day.
    p = det.day_model.predict_proba(Xte)[:, 1]
    picks = {"solar_day": int(np.argmax(np.where(te.solar_installed.values == 1, p, -1))),
             "non_solar_day": int(np.argmin(np.where(te.solar_installed.values == 0, p, 2))),
             "uncertain_day": int(np.argmin(np.abs(p - 0.5)))}
    _, sv_loc = tree_shap(lgbm, Xte.iloc[list(picks.values())])
    lime = lime_explainer(Xtr.sample(20000, random_state=0))
    lime_rows = []
    for k, (name, i) in enumerate(picks.items()):
        plt.figure()
        shap.plots.waterfall(sv_loc[k], max_display=12, show=False)
        plt.title(f"SHAP waterfall — {name} (true={te.solar_installed[i]}, p={p[i]:.3f})")
        plt.tight_layout(); plt.savefig(out / f"shap_waterfall_{name}.png", dpi=130, bbox_inches="tight")
        plt.close("all")
        exp = lime.explain_instance(Xte.iloc[i].values, lambda x: det.day_model.predict_proba(
            pd.DataFrame(x, columns=Xte.columns)), num_features=10, num_samples=3000)
        fig = exp.as_pyplot_figure(label=1)
        fig.suptitle(f"LIME — {name} (true={te.solar_installed[i]}, p={p[i]:.3f})")
        fig.tight_layout(); fig.savefig(out / f"lime_{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        for rule, weight in exp.as_list(label=1):
            lime_rows.append({"case": name, "hh": te.hh[i], "day": str(te.day[i].date()), "true": int(te.solar_installed[i]),
                              "p_solar": float(p[i]), "rule": rule, "weight": weight})
    pd.DataFrame(lime_rows).to_csv(out / "lime_explanations.csv", index=False)
    print("Top SHAP features (LightGBM):\n", mean_abs.head(10).round(4))
    print("Top SHAP features (ensemble):\n", ens_imp.head(10).round(4))


if __name__ == "__main__":
    main()
