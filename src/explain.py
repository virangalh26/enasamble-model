"""Explanations for the NGBoost poverty-targeting model: SHAP, LIME, partial dependence and permutation
importance, on the held-out test households.

* SHAP (TreeExplainer, exact for tree ensembles) on the predicted mean log consumption (mu): which features
  push a household towards or away from poverty, globally and for single households. The same is done for
  the predicted spread (sigma): which features make the model *unsure*.
* LIME: local linear surrogate of P(poor) for three chosen households.
* Partial dependence (PDP + ICE) of P(poor) for the most important numeric features.
* Permutation importance on test PR-AUC, as a model-agnostic check of the SHAP ranking.
One-hot columns are summed back to their original feature (e.g. all `walls=k` -> `walls`).

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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.features import CATEGORICAL

DPI = 130


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def raw_name(col: str) -> str:
    return col.split("=")[0] if col.split("=")[0] in CATEGORICAL else col


def grouped(values: pd.DataFrame) -> pd.DataFrame:
    """Sum one-hot columns back to their original feature."""
    return values.T.groupby(raw_name).sum().T


def shap_values(targeter, X, output):
    explainer = shap.TreeExplainer(targeter.model, model_output=output)
    return pd.DataFrame(explainer.shap_values(X.values), columns=X.columns, index=X.index)


def plot_bar(series, title, xlabel, path, top=20, color="tab:blue"):
    s = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save(fig, path)


def plot_local(contrib, title, path, top=12):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=["tab:red" if v < 0 else "tab:green" for v in s.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("SHAP on log consumption (red = towards poverty)")
    save(fig, path)


def pdp(targeter, X, feature, grid_size=20, ice=60, seed=0):
    vals = np.unique(np.quantile(X[feature], np.linspace(0.02, 0.98, grid_size)))
    sample = X.sample(min(len(X), 400), random_state=seed)
    curves = []
    for v in vals:
        Xv = sample.copy()
        Xv[feature] = v
        curves.append(targeter.p_poor_from_matrix(Xv))
    curves = np.array(curves)  # grid x households
    return vals, curves.mean(axis=1), curves[:, :ice]


def permutation_importance(targeter, X, y, features, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, targeter.p_poor_from_matrix(X))
    rows = []
    for f in features:
        cols = [c for c in X.columns if raw_name(c) == f]
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]
            drops.append(base - average_precision_score(y, targeter.p_poor_from_matrix(Xp)))
        rows.append({"feature": f, "pr_auc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/poverty_targeter.joblib")
    ap.add_argument("--data", default="data/processed/households.parquet")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/explain")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t = joblib.load(a.model)
    hh = pd.read_parquet(a.data)
    test_psus = set(json.load(open(Path(a.reports) / "split.json"))["test_psus"])
    tr, te = hh[~hh.psu.isin(test_psus)], hh[hh.psu.isin(test_psus)].reset_index(drop=True)
    Xtr, Xte = t.matrix(tr), t.matrix(te)
    p = t.p_poor_from_matrix(Xte)
    y = te["poor"].values
    summary = {}

    # SHAP on mu (log consumption) and on sigma (uncertainty).
    sv_mu = shap_values(t, Xte, 0)
    sv_sd = shap_values(t, Xte, 1)
    imp_mu = grouped(sv_mu.abs()).mean().sort_values(ascending=False)
    imp_sd = grouped(sv_sd.abs()).mean().sort_values(ascending=False)
    pd.DataFrame({"mean_abs_shap_mu": imp_mu, "mean_abs_shap_sigma": imp_sd}).sort_values(
        "mean_abs_shap_mu", ascending=False).to_csv(out / "shap_importance.csv")
    plot_bar(imp_mu, "What drives predicted consumption (SHAP, test households)", "mean |SHAP| on log consumption",
             out / "shap_importance_mu.png")
    plot_bar(imp_sd, "What makes the model unsure (SHAP on the predicted spread)", "mean |SHAP| on log sigma",
             out / "shap_importance_sigma.png", color="tab:purple")
    top_cols = sv_mu.abs().mean().sort_values(ascending=False).index[:15]
    shap.summary_plot(sv_mu[top_cols].values, Xte[top_cols], show=False, max_display=15)
    save(plt.gcf(), out / "shap_beeswarm_mu.png")

    # Direction check against domain knowledge: does more of a feature raise or lower consumption?
    direction = {}
    for f in imp_mu.index[:15]:
        if f in Xte and Xte[f].nunique() > 1:
            direction[f] = round(float(spearmanr(Xte[f], sv_mu[f]).statistic), 3)
    pd.Series(direction, name="spearman(feature value, SHAP on mu)").to_csv(out / "shap_direction.csv")

    # Three households: a poor one correctly flagged, a poor one missed, a non-poor one wrongly flagged.
    flag = p >= t.threshold
    cases = {"poor_flagged": np.where((y == 1) & flag)[0], "poor_missed": np.where((y == 1) & ~flag)[0],
             "nonpoor_flagged": np.where((y == 0) & flag)[0]}
    picks = {k: int(v[np.argmax(p[v])] if k != "poor_missed" else v[np.argmin(p[v])]) for k, v in cases.items()
             if len(v)}
    lime = LimeTabularExplainer(Xtr.values, feature_names=list(Xtr.columns), class_names=["not poor", "poor"],
                                mode="classification", discretize_continuous=True, random_state=0)
    proba = lambda x: np.c_[1 - (q := t.p_poor_from_matrix(pd.DataFrame(x, columns=Xtr.columns))), q]
    lime_rows = []
    for name, i in picks.items():
        row = te.iloc[i]
        title = (f"{name}: {row.district_name}, {row.sector_name}, P(poor)={p[i]:.2f}, "
                 f"actual Rs {row.pc_consumption:,.0f}/person")
        plot_local(grouped(sv_mu.iloc[[i]]).iloc[0], title, out / f"shap_local_{name}.png")
        exp = lime.explain_instance(Xte.values[i], proba, num_features=10, num_samples=3000)
        for rule, w in exp.as_list():
            lime_rows.append({"case": name, "rule": rule, "weight_on_p_poor": w})
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME, {name}", fontsize=10)
        save(fig, out / f"lime_{name}.png")
    pd.DataFrame(lime_rows).to_csv(out / "lime_explanations.csv", index=False)
    summary["cases"] = {k: {"row": i, "p_poor": float(p[i]), "district": te.iloc[i].district_name}
                        for k, i in picks.items()}

    # Partial dependence of P(poor) for the strongest numeric features.
    numeric = [f for f in imp_mu.index if f not in CATEGORICAL and f in Xte and Xte[f].nunique() > 2][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), numeric):
        vals, mean, ice = pdp(t, Xte, f)
        ax.plot(vals, ice, color="tab:blue", alpha=0.07)
        ax.plot(vals, mean, color="tab:red", lw=2.5, label="PDP (average)")
        ax.set_title(f, fontsize=10)
        ax.set_ylabel("P(poor)")
    axes.ravel()[0].legend()
    save(fig, out / "pdp_ice.png")

    # Permutation importance as an independent check of the SHAP ranking.
    perm = permutation_importance(t, Xte, y, list(imp_mu.index[:20]))
    perm.to_csv(out / "permutation_importance.csv", index=False)
    plot_bar(perm.set_index("feature")["pr_auc_drop"], "Permutation importance (drop in test PR-AUC)",
             "PR-AUC drop", out / "permutation_importance.png", color="tab:gray")
    rho = spearmanr(imp_mu.reindex(perm.feature), perm.pr_auc_drop).statistic
    summary.update({"top_shap_mu": list(imp_mu.index[:10]), "top_shap_sigma": list(imp_sd.index[:10]),
                    "top_permutation": list(perm.feature[:10]), "shap_vs_permutation_spearman": float(rho)})
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
