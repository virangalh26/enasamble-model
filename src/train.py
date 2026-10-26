"""Train and evaluate the NGBoost poverty-targeting model on HIES 2019 households, and compare it with
algorithms covered in lectures.

Evaluation rules:
* Households of one survey cluster (PSU) are never split between training and evaluation, because
  neighbours share prices, roads and wealth (grouped, stratified splits).
* 20% of PSUs are held out as a test set and touched once, at the end.
* Hyperparameters and the decision threshold are chosen on the training households only.

Usage:
    python -m src.train --data data/processed/households.parquet
"""
import argparse
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

from src.drift import reference_stats
from src.features import CATEGORICAL, model_columns, to_matrix
from src.model import PovertyTargeter

MODEL_NAME = "NGBoost (Normal on log consumption)"
GRID = [{"max_depth": d, "learning_rate": lr, "minibatch_frac": mb}
        for d, lr, mb in product([2, 3, 4], [0.02, 0.05], [0.7, 1.0])]
MAX_ROUNDS, EARLY_STOP = 2000, 50
FAIRNESS_GROUPS = ["sector_name", "head_female", "head_ethnicity", "district_name"]


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def ngboost(max_depth=3, learning_rate=0.03, minibatch_frac=1.0, n_estimators=MAX_ROUNDS,
            natural_gradient=True, seed=42):
    base = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    return NGBRegressor(Dist=Normal, Base=base, n_estimators=n_estimators, learning_rate=learning_rate,
                        minibatch_frac=minibatch_frac, natural_gradient=natural_gradient,
                        verbose=False, random_state=seed)


def fit_ngb(params, X, y_log, seed, early_stop=True):
    """Fit NGBoost; with early stopping, 15% of the given rows are set aside to pick the number of rounds."""
    m = ngboost(**params, seed=seed)
    if not early_stop:
        return m.fit(X, y_log)
    rng = np.random.default_rng(seed)
    val = rng.random(len(X)) < 0.15
    m.fit(X[~val], y_log[~val], X_val=X[val], Y_val=y_log[val], early_stopping_rounds=EARLY_STOP)
    return m


def p_poor(model, X, log_z):
    d = model.pred_dist(X)
    return norm.cdf((log_z - d.params["loc"]) / d.params["scale"])


class LinearPMT:
    """The classic proxy means test: OLS on log consumption, Normal residuals -> P(poor)."""

    def __init__(self, log_z):
        self.log_z = log_z

    def fit(self, X, y_log):
        self.sc = StandardScaler().fit(X)
        self.lr = LinearRegression().fit(self.sc.transform(X), y_log)
        self.sd = float(np.std(y_log - self.lr.predict(self.sc.transform(X))))
        return self

    def predict_proba(self, X):
        p = norm.cdf((self.log_z - self.lr.predict(self.sc.transform(X))) / self.sd)
        return np.c_[1 - p, p]


def lecture_models(seed, pos_weight):
    """Algorithms covered in lectures, used as baselines (same rows, same splits)."""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
        "KNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=seed,
                                 eval_metric="logloss", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def scores(y, p, threshold):
    """Classification metrics plus the two welfare-targeting errors at a fixed budget.

    Budget = as many households as are truly poor in the set. Exclusion error: share of poor households
    not selected. Inclusion error: share of selected households that are not poor.
    """
    y, p = np.asarray(y), np.asarray(p)
    k = int(y.sum())
    top = np.zeros_like(y)
    top[np.argsort(-p)[:k]] = 1
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "exclusion_error_at_budget": 1 - top[y == 1].mean(), "inclusion_error_at_budget": 1 - y[top == 1].mean(),
    }


def regression_scores(y_log, mu, sigma, interval=0.8):
    zq = norm.ppf(0.5 + interval / 2)
    return {"rmse_log": float(np.sqrt(np.mean((y_log - mu) ** 2))),
            "nll": float(-norm.logpdf(y_log, mu, sigma).mean()),
            f"coverage_{int(interval * 100)}": float(np.mean(np.abs(y_log - mu) <= zq * sigma)),
            "mean_sigma": float(np.mean(sigma))}


def fairness(test: pd.DataFrame, p, threshold):
    rows = []
    for g in FAIRNESS_GROUPS:
        for val, d in test.assign(p=p).groupby(g):
            y, flag = d["poor"].values, (d["p"].values >= threshold).astype(int)
            rows.append({"group": g, "value": val, "households": len(d), "poor": int(y.sum()),
                         "flagged_share": flag.mean(),
                         "exclusion_error": (1 - flag[y == 1].mean()) if y.sum() else np.nan,
                         "false_positive_rate": flag[y == 0].mean() if (y == 0).any() else np.nan})
    return pd.DataFrame(rows)


def form_defaults(train: pd.DataFrame, Xtr: pd.DataFrame, seed: int) -> dict:
    """Typical training household (median / most common answer), used by the app for fields not asked,
    plus a background sample for LIME in the app."""
    raw = train[model_columns(train)]
    values = {c: (float(raw[c].mode().iloc[0]) if c in CATEGORICAL else float(raw[c].median())) for c in raw}
    cats = {c: sorted(float(v) for v in raw[c].dropna().unique()) for c in CATEGORICAL if c in raw}
    background = Xtr.sample(min(len(Xtr), 300), random_state=seed).round(4).values.tolist()
    return {"values": values, "categories": cats, "lime_background": background}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/households.parquet")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--quick", action="store_true", help="tiny grid, for tests")
    args = ap.parse_args()
    prm = yaml.safe_load(open(args.params))
    seed, z = prm["train"]["seed"], prm["data"]["poverty_line"]
    log_z = float(np.log(z))
    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    fh = open(reports_dir / "train_log.txt", "w")
    t0 = time.time()

    hh = pd.read_parquet(args.data).reset_index(drop=True)
    y, y_log, groups = hh["poor"].values, hh["log_pc_consumption"].values, hh["psu"].values
    outer = StratifiedGroupKFold(n_splits=round(1 / prm["train"]["test_share"]), shuffle=True, random_state=seed)
    tr_idx, te_idx = next(outer.split(hh, y, groups))
    train, test = hh.iloc[tr_idx].reset_index(drop=True), hh.iloc[te_idx].reset_index(drop=True)
    Xtr = to_matrix(train)
    cols = list(Xtr.columns)
    Xte = to_matrix(test, cols)
    ytr, yte, ltr, lte = train["poor"].values, test["poor"].values, train["log_pc_consumption"].values, \
        test["log_pc_consumption"].values
    cv = list(StratifiedGroupKFold(n_splits=prm["train"]["cv_folds"], shuffle=True, random_state=seed)
              .split(train, ytr, train["psu"].values))
    json.dump({"test_psus": sorted(map(int, test["psu"].unique()))}, open(reports_dir / "split.json", "w"))
    log(f"{len(hh):,} households ({y.mean():.1%} poor), {Xtr.shape[1]} model features | "
        f"train {len(train):,} ({train['psu'].nunique()} PSUs) | test {len(test):,} ({test['psu'].nunique()} PSUs)", fh)

    # 1) NGBoost hyperparameter search: grouped CV on training households, early stopping inside each fold.
    grid = GRID[:2] if args.quick else GRID
    search = []
    for g in grid:
        s, p_oof, rounds = time.time(), np.zeros(len(train)), []
        for tr, va in cv:
            m = fit_ngb(g, Xtr.iloc[tr].values, ltr[tr], seed)
            rounds.append(m.best_val_loss_itr or m.n_estimators)
            p_oof[va] = p_poor(m, Xtr.iloc[va].values, log_z)
        row = {**g, "cv_pr_auc": average_precision_score(ytr, p_oof), "cv_roc_auc": roc_auc_score(ytr, p_oof),
               "rounds": int(np.mean(rounds)), "seconds": round(time.time() - s, 1)}
        search.append(row)
        log(f"[search] {g} PR-AUC={row['cv_pr_auc']:.4f} ROC-AUC={row['cv_roc_auc']:.4f} rounds={row['rounds']}", fh)
    search = pd.DataFrame(search).sort_values("cv_pr_auc", ascending=False)
    search.to_csv(reports_dir / "ngboost_grid_search.csv", index=False)
    best = {k: search.iloc[0][k] for k in ["max_depth", "learning_rate", "minibatch_frac"]}
    best = {"max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
            "minibatch_frac": float(best["minibatch_frac"]), "n_estimators": int(search.iloc[0]["rounds"])}
    json.dump({"NGBoost": best}, open(reports_dir / "best_params.json", "w"), indent=2)
    log(f"best NGBoost parameters: {best}", fh)

    # 2) NGBoost, its ordinary-gradient ablation, the linear PMT and lecture baselines: same CV, then one test fit.
    pos_weight = float((ytr == 0).sum() / max(ytr.sum(), 1))
    candidates = {MODEL_NAME: ("ngb", {**best, "natural_gradient": True}),
                  "NGBoost, ordinary gradient (ablation)": ("ngb", {**best, "natural_gradient": False}),
                  "Linear Regression PMT (standard practice)": ("pmt", None),
                  **{k: ("clf", v) for k, v in lecture_models(seed, pos_weight).items()}}
    if args.quick:
        candidates = dict(list(candidates.items())[:4])
    rows, fitted, test_p = [], {}, {}
    for name, (kind, spec) in candidates.items():
        s, p_oof = time.time(), np.zeros(len(train))

        def fit(idx):
            if kind == "ngb":
                return ngboost(**spec, seed=seed).fit(Xtr.iloc[idx].values, ltr[idx])
            if kind == "pmt":
                return LinearPMT(log_z).fit(Xtr.iloc[idx].values, ltr[idx])
            return clone(spec).fit(Xtr.iloc[idx], ytr[idx])

        def prob(m, X):
            return p_poor(m, X.values, log_z) if kind == "ngb" else m.predict_proba(X.values if kind == "pmt" else X)[:, 1]

        for tr, va in cv:
            p_oof[va] = prob(fit(tr), Xtr.iloc[va])
        thr = best_f1_threshold(ytr, p_oof)
        m = fit(np.arange(len(train)))
        p_te = prob(m, Xte)
        cv_s, te_s = scores(ytr, p_oof, thr), scores(yte, p_te, thr)
        rows.append({"model": name, "family": "new (NGBoost)" if name.startswith("NGBoost") else
                     "baseline (standard PMT)" if kind == "pmt" else "lecture", "threshold": thr,
                     **{f"cv_{k}": v for k, v in cv_s.items()}, **{f"test_{k}": v for k, v in te_s.items()},
                     "seconds": round(time.time() - s, 1)})
        fitted[name], test_p[name] = (m, thr), p_te
        log(f"{name:44s} CV PR-AUC={cv_s['pr_auc']:.4f} ROC-AUC={cv_s['roc_auc']:.4f} | test PR-AUC={te_s['pr_auc']:.4f} "
            f"ROC-AUC={te_s['roc_auc']:.4f} F1={te_s['f1']:.4f} excl@budget={te_s['exclusion_error_at_budget']:.3f} "
            f"({time.time() - s:.0f}s)", fh)
    comp = pd.DataFrame(rows)
    comp.to_csv(reports_dir / "model_comparison.csv", index=False)

    # 3) Final model, its uncertainty quality and fairness on the test set.
    ngb, thr = fitted[MODEL_NAME]
    targeter = PovertyTargeter(model=ngb, columns=cols, poverty_line=z, threshold=thr)
    mu, sigma = targeter.distribution(Xte)
    reg = regression_scores(lte, mu, sigma)
    final = comp.set_index("model").loc[MODEL_NAME]
    metrics = {"cv": {k[3:]: float(final[k]) for k in comp if k.startswith("cv_")},
               "test": {k[5:]: float(final[k]) for k in comp if k.startswith("test_")},
               "test_regression": reg, "threshold": thr}
    json.dump(metrics, open(reports_dir / "metrics.json", "w"), indent=2)
    fairness(test, test_p[MODEL_NAME], thr).to_csv(reports_dir / "fairness.csv", index=False)
    pred = targeter.predict(test)
    pd.concat([test[["psu", "snumber", "hhno", "district_name", "sector_name", "poor", "pc_consumption"]], pred],
              axis=1).to_csv(reports_dir / "test_predictions.csv", index=False)
    targeter.info = {"model_name": MODEL_NAME, "best_params": best, "test": metrics["test"],
                     "test_regression": reg, "trained_households": len(train)}
    joblib.dump(targeter, models_dir / "poverty_targeter.joblib", compress=3)
    json.dump({"model": MODEL_NAME, "threshold": thr, "poverty_line": z, "features": cols,
               "best_params": best, "test": metrics["test"], "test_regression": reg,
               "baselines": comp[["model", "test_pr_auc", "test_roc_auc", "test_f1"]].to_dict(orient="records")},
              open(models_dir / "metadata.json", "w"), indent=2)
    json.dump(reference_stats(train), open(models_dir / "reference_stats.json", "w"), indent=2)
    json.dump(form_defaults(train, Xtr, seed), open(models_dir / "form_defaults.json", "w"))
    log(f"test regression: {reg}", fh)
    log(f"done in {time.time() - t0:.0f}s", fh)


if __name__ == "__main__":
    main()
