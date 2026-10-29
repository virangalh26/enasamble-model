"""Train and evaluate the Rotation Forest solar detector on RECON-SL household-days, against lecture baselines.

The main model is a Rotation Forest (src/rotation_forest.py), an ensemble method not covered in the lectures.
The algorithms taught in lectures (logistic regression, KNN, decision tree, random forest, gradient boosting,
XGBoost, LightGBM, CatBoost, voting, stacking) are trained on exactly the same data as baselines.

Evaluation rules (to keep the score honest):
* A household's days never appear in both training and evaluation folds (grouped splits).
* 20% of households are held out as a test set and touched once, at the end.
* Hyperparameters and the decision threshold are chosen on the training households only.
* Every model trains on the same sample of at most `days_per_household` days per training household: days of one
  home are near-duplicates, so more of them add training time but little information.

Usage:
    python -m src.train --data data/processed/household_days.parquet
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.drift import reference_stats
from src.features import FEATURE_NAMES, HOURS, make_features
from src.model import SolarDetector
from src.rotation_forest import RotationForestClassifier

SEED = 42
MAIN = "Rotation Forest"

# Grid searched on the tuning households (3-fold grouped CV, log loss). Kept small: every tree is expensive.
GRID = {
    "subset_size": [3, 6],               # features per PCA subset (the paper's K)
    "max_features": [None, "sqrt"],      # features each tree split may look at (None = all, as in the paper)
    "min_samples_leaf": [5, 20],         # leaf size: smoothness of each tree
}


# ---------------------------------------------------------------- metrics
def household_scores(y_day, p_day, groups, threshold=0.5):
    hh = pd.DataFrame({"g": groups, "y": y_day, "p": p_day}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    pred = (hh.p >= threshold).astype(int)
    return {
        "hh_accuracy": accuracy_score(hh.y, pred),
        "hh_f1": f1_score(hh.y, pred, zero_division=0),
        "hh_precision": precision_score(hh.y, pred, zero_division=0),
        "hh_recall": recall_score(hh.y, pred, zero_division=0),
        "hh_auc": roc_auc_score(hh.y, hh.p),
        "hh_errors": int((pred != hh.y).sum()),
        "hh_n": int(len(hh)),
        "day_accuracy": accuracy_score(y_day, p_day >= 0.5),
        "day_auc": roc_auc_score(y_day, p_day),
        "day_logloss": log_loss(y_day, np.clip(p_day, 1e-6, 1 - 1e-6)),
    }


def best_threshold(y_day, p_day, groups):
    """Pick the household threshold that maximises training accuracy (ties: closest to 0.5)."""
    hh = pd.DataFrame({"g": groups, "y": y_day, "p": p_day}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    acc = np.array([accuracy_score(hh.y, hh.p >= t) for t in grid])
    best = grid[acc == acc.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])


# ---------------------------------------------------------------- models
def baseline_models():
    """Algorithms covered in the lectures: comparison baselines only."""
    return {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "KNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=-1,
                                 random_state=SEED, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=-1, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.08, random_seed=SEED, verbose=0,
                                       thread_count=-1),
    }


def oof_and_test(model, Xtr, ytr, Xte, cv):
    """Out-of-fold probabilities on the training households, and test probabilities from a refit on all of them."""
    p = np.zeros(len(Xtr))
    for tr, va in cv:
        p[va] = clone(model).fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[va])[:, 1]
    fitted = clone(model).fit(Xtr, ytr)
    return p, fitted.predict_proba(Xte)[:, 1], fitted


def sample_days(hh, n, seed=SEED):
    """Row positions of at most n random days per household."""
    return (pd.Series(np.arange(len(hh)), index=hh).groupby(level=0, group_keys=False)
            .apply(lambda s: s.sample(min(n, len(s)), random_state=seed)).sort_values().to_numpy())


# ---------------------------------------------------------------- tuning
def grid_search(X, y, g, n_estimators, seed=SEED):
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(X, y, g))
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        s = time.time()
        model = RotationForestClassifier(n_estimators=n_estimators, random_state=seed, **params)
        losses, aucs = [], []
        for tr, va in cv:
            p = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            losses.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)))
            aucs.append(roc_auc_score(y[va], p))
        rows.append({**params, "cv_logloss": float(np.mean(losses)), "cv_logloss_std": float(np.std(losses)),
                     "cv_day_auc": float(np.mean(aucs)), "seconds": round(time.time() - s, 1)})
        print(f"[grid] {params} logloss={rows[-1]['cv_logloss']:.4f} AUC={rows[-1]['cv_day_auc']:.4f} "
              f"({rows[-1]['seconds']:.0f}s)", flush=True)
    table = pd.DataFrame(rows).sort_values("cv_logloss").reset_index(drop=True)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in table.loc[0, list(GRID)].items()}
    return best, table


def trees_curve(model, X, y, Xva, yva, out):
    """Validation log loss and AUC as trees are added: shows where the ensemble stops improving."""
    fitted = clone(model).fit(X, y)
    rows = [{"trees": k, "logloss": log_loss(yva, np.clip(p[:, 1], 1e-6, 1 - 1e-6)), "auc": roc_auc_score(yva, p[:, 1])}
            for k, p in enumerate(fitted.staged_predict_proba(Xva), start=1)]
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(curve.trees, curve.logloss, color="#5170FF")
    ax.set_xlabel("number of rotated trees"); ax.set_ylabel("validation log loss (days)")
    ax.set_title("Rotation Forest: adding trees (one grouped CV fold)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return curve


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/household_days.parquet")
    ap.add_argument("--trees", type=int, default=100, help="trees in the final Rotation Forest")
    ap.add_argument("--search-trees", type=int, default=30, help="trees per Rotation Forest in the grid search")
    ap.add_argument("--days-per-household", type=int, default=40)
    ap.add_argument("--tune-households", type=float, default=0.5,
                    help="fraction of training households used for the grid search (speed)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    t0 = time.time()
    models_dir, reports_dir = Path(a.models_dir), Path(a.reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    hd = pd.read_parquet(a.data)
    X = make_features(hd[HOURS])[FEATURE_NAMES].reset_index(drop=True)
    y = hd["solar_installed"].to_numpy()
    g = hd["hh"].to_numpy()      # split groups: whole households
    u = hd["unit"].to_numpy()    # evaluation units: household-periods with a constant label
    print(f"{len(X):,} household-days, {pd.Series(g).nunique():,} households, {X.shape[1]} features")

    # Hold out 20% of households once. The test set keeps every one of their days.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(outer.split(X, y, g))
    assert not set(g[tr_idx]) & set(g[te_idx]), "household leaked across train/test"
    json.dump({"train_households": sorted(set(g[tr_idx])), "test_households": sorted(set(g[te_idx]))},
              open(reports_dir / "split.json", "w"))
    tr_idx = tr_idx[sample_days(g[tr_idx], a.days_per_household)]
    Xtr, ytr, gtr, utr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], g[tr_idx], u[tr_idx]
    Xte, yte, ute = X.iloc[te_idx].reset_index(drop=True), y[te_idx], u[te_idx]
    print(f"train: {pd.Series(gtr).nunique()} households ({len(Xtr):,} sampled days) / "
          f"test: {pd.Series(g[te_idx]).nunique()} households ({len(Xte):,} days)")

    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, ytr, gtr))
    rows, oof, test_p = [], {}, {}

    def record(name, stage, p_cv, p_te, seconds):
        cv_s, te_s = household_scores(ytr, p_cv, utr), household_scores(yte, p_te, ute)
        rows.append({"model": name, "stage": stage, **{f"cv_{k}": v for k, v in cv_s.items()},
                     **{f"test_{k}": v for k, v in te_s.items()}, "seconds": round(seconds, 1)})
        oof[name], test_p[name] = p_cv, p_te
        print(f"[{stage:9s}] {name:34s} CV hh_acc={cv_s['hh_accuracy']:.4f} AUC={cv_s['hh_auc']:.4f} | "
              f"test hh_acc={te_s['hh_accuracy']:.4f} errors={te_s['hh_errors']} ({seconds:.0f}s)", flush=True)

    # 1) Lecture algorithms as baselines (default settings).
    for name, model in baseline_models().items():
        s = time.time()
        p_cv, p_te, _ = oof_and_test(model, Xtr, ytr, Xte, cv)
        record(name, "baseline", p_cv, p_te, time.time() - s)

    # Lecture ensembles built from those baselines: soft voting and stacking (LR meta-learner).
    s = time.time()
    members = ["Extra Trees", "XGBoost", "LightGBM", "CatBoost"]
    base_cv, base_te = np.column_stack([oof[n] for n in members]), np.column_stack([test_p[n] for n in members])
    record("Soft voting (ET, XGB, LGBM, CatBoost)", "baseline", base_cv.mean(1), base_te.mean(1), time.time() - s)
    meta_cv = np.zeros(len(Xtr))
    for tr, va in cv:
        meta_cv[va] = LogisticRegression(max_iter=1000).fit(base_cv[tr], ytr[tr]).predict_proba(base_cv[va])[:, 1]
    meta_te = LogisticRegression(max_iter=1000).fit(base_cv, ytr).predict_proba(base_te)[:, 1]
    record("Stacking (ET, XGB, LGBM, CatBoost -> LR)", "baseline", meta_cv, meta_te, time.time() - s)

    # 2) Rotation Forest: grid search on a subset of the training households.
    rng = np.random.default_rng(SEED)
    tune_hh = rng.choice(sorted(set(gtr)), int(len(set(gtr)) * a.tune_households), replace=False)
    tmask = np.isin(gtr, tune_hh)
    best, grid = grid_search(Xtr[tmask].reset_index(drop=True), ytr[tmask], gtr[tmask], a.search_trees)
    grid.to_csv(reports_dir / "rotation_forest_grid_search.csv", index=False)
    print(f"Best Rotation Forest settings: {best}")

    # Ablation: the same ensemble with the rotation switched off (axis-aligned trees on the same features).
    s = time.time()
    no_rot = RotationForestClassifier(n_estimators=a.trees, rotate=False, random_state=SEED, **best)
    p_cv, p_te, _ = oof_and_test(no_rot, Xtr, ytr, Xte, cv)
    record("Rotation Forest without rotation (ablation)", "ablation", p_cv, p_te, time.time() - s)

    # 3) Final Rotation Forest.
    s = time.time()
    rotf = RotationForestClassifier(n_estimators=a.trees, random_state=SEED, **best)
    p_cv, p_te, final = oof_and_test(rotf, Xtr, ytr, Xte, cv)
    record(MAIN, "main", p_cv, p_te, time.time() - s)
    tr0, va0 = cv[0]
    curve = trees_curve(rotf, Xtr.iloc[tr0], ytr[tr0], Xtr.iloc[va0], ytr[va0], reports_dir / "figures" / "trees_curve.png")
    curve.to_csv(reports_dir / "trees_curve.csv", index=False)

    thr = best_threshold(ytr, p_cv, utr)
    final_cv = household_scores(ytr, p_cv, utr, thr)
    final_te = household_scores(yte, p_te, ute, thr)
    print(f"Threshold (train OOF) = {thr:.2f}; test hh_acc={final_te['hh_accuracy']:.5f} "
          f"errors={final_te['hh_errors']}/{final_te['hh_n']}")

    comp = pd.DataFrame(rows)
    comp.to_csv(reports_dir / "model_comparison.csv", index=False)
    json.dump({"rotation_forest": {**best, "n_estimators": a.trees, "sample_fraction": 0.75},
               "search": {"grid": GRID, "trees": a.search_trees, "tune_households": a.tune_households,
                          "folds": 3, "objective": "log loss"}},
              open(reports_dir / "best_params.json", "w"), indent=2)
    comparison_figure(comp, reports_dir / "figures" / "model_comparison.png")

    hd_te = hd.iloc[te_idx].reset_index(drop=True)
    hh_te = (pd.DataFrame({"unit": ute, "hh": g[te_idx], "y": yte, "p_day": p_te,
                           "export_kwh": hd_te["export_kwh"].values, "first_export_day": hd_te["first_export_day"].values})
             .groupby("unit").agg(hh=("hh", "first"), solar_installed=("y", "first"), p_solar=("p_day", "mean"),
                                  n_days=("p_day", "size"), export_kwh=("export_kwh", "first"),
                                  first_export_day=("first_export_day", "first")))
    hh_te["pred"] = (hh_te.p_solar >= thr).astype(int)
    hh_te.to_csv(reports_dir / "test_household_predictions.csv")
    pd.DataFrame({"hh": g[te_idx], "unit": ute, "day": hd_te["day"].values, "y": yte, "p": p_te}).to_parquet(
        reports_dir / "test_day_predictions.parquet", index=False)

    detector = SolarDetector(day_model=final, feature_names=FEATURE_NAMES, threshold=thr, info={
        "trained_on_households": int(pd.Series(gtr).nunique()), "evaluation_unit": "household-period",
        "best_params": best, "cv": final_cv, "test": final_te,
        "background": Xtr.sample(50, random_state=SEED).reset_index(drop=True)})  # SHAP background for the app
    joblib.dump(detector, models_dir / "solar_detector.joblib", compress=3)
    json.dump({"model": f"Rotation Forest ({a.trees} PCA-rotated decision trees)",
               "params": {**best, "n_estimators": a.trees}, "threshold": thr, "features": FEATURE_NAMES,
               "hours": HOURS, "cv_train_households": final_cv, "test_households": final_te,
               "export_threshold_kwh": 1.0, "minutes": round((time.time() - t0) / 60, 1),
               "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(models_dir / "metadata.json", "w"), indent=2, default=float)
    write_tracking_files(final_cv, final_te, thr, Xtr, models_dir, reports_dir)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


def comparison_figure(comp, out):
    c = comp[comp.model != "Dummy (majority)"].sort_values("test_hh_auc")
    colors = ["#E8A33D" if s == "main" else "#9AA5B1" if s == "ablation" else "#5170FF" for s in c.stage]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (col, title) in zip(axes, [("cv_hh_auc", "Household AUC — grouped CV (train)"),
                                       ("test_hh_auc", "Household AUC — held-out test")]):
        ax.barh(c.model, c[col], color=colors)
        ax.set_xlim(max(0.9, c[col].min() - 0.005), 1.0)
        ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Rotation Forest (orange) vs lecture baselines (blue) and the no-rotation ablation (grey)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def write_tracking_files(cv_scores, test_scores, threshold, X_train, models_dir, reports_dir):
    """Metrics for `dvc metrics diff` / the quality gate, and drift reference statistics."""
    json.dump({"threshold": threshold, "cv": cv_scores, "test": test_scores},
              open(Path(reports_dir) / "metrics.json", "w"), indent=2, default=float)
    json.dump(reference_stats(X_train), open(Path(models_dir) / "reference_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
