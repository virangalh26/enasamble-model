"""Train, tune, compare and ensemble rooftop-solar detectors on RECON-SL household-days.

Evaluation rules (to keep the score honest):
* A household's days never appear in both training and evaluation folds (grouped splits).
* 20% of households are held out as a test set and touched once, at the end.
* Hyperparameters and the decision threshold are chosen on the training households only.

Usage:
    python -m src.train --data data/processed/household_days.parquet
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES, HOURS, make_features
from src.model import SolarDetector

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
def base_models():
    return {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "KNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, max_samples=0.5,
                                                n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, max_samples=0.5, bootstrap=True,
                                            n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=-1,
                                 random_state=SEED, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=-1, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.08, random_seed=SEED, verbose=0,
                                       thread_count=-1),
    }


SUBSAMPLE = {"KNN (k=25)": 60_000}  # KNN prediction cost grows with training size


def fit(model, X, y, name=None, rng=np.random.default_rng(SEED)):
    n = SUBSAMPLE.get(name)
    if n and len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        X, y = X.iloc[idx], y[idx]
    return model.fit(X, y)


def oof_predict(make_model, X, y, splits, name=None):
    p = np.zeros(len(X))
    for tr, va in splits:
        m = fit(make_model(), X.iloc[tr], y[tr], name)
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


# ---------------------------------------------------------------- tuning
def tune(name, X, y, splits, n_trials):
    def lgbm(t):
        return LGBMClassifier(
            n_estimators=t.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            n_jobs=-1, random_state=SEED, verbose=-1)

    def xgb(t):
        return XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=t.suggest_int("max_depth", 3, 10),
            min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss")

    def cat(t):
        return CatBoostClassifier(
            iterations=t.suggest_int("iterations", 300, 1200, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            depth=t.suggest_int("depth", 4, 9),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 20, log=True),
            random_seed=SEED, verbose=0, thread_count=-1)

    def et(t):
        return ExtraTreesClassifier(
            n_estimators=t.suggest_int("n_estimators", 200, 600, step=100),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20, log=True),
            max_features=t.suggest_float("max_features", 0.1, 0.8),
            max_samples=0.5, bootstrap=True, n_jobs=-1, random_state=SEED)

    space = {"LightGBM": lgbm, "XGBoost": xgb, "CatBoost": cat, "Extra Trees": et}[name]

    def objective(t):
        model = space(t)
        losses = []
        for tr, va in splits:
            p = model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            losses.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = optuna.trial.FixedTrial(study.best_params)
    return space(best), study.best_params, study.best_value


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/household_days.parquet")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tune-households", type=float, default=0.5,
                    help="fraction of training households used for Optuna search (speed)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    t0 = time.time()
    models_dir, reports_dir = Path(a.models_dir), Path(a.reports_dir)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    hd = pd.read_parquet(a.data)
    X = make_features(hd[HOURS])[FEATURE_NAMES].reset_index(drop=True)
    y = hd["solar_installed"].to_numpy()
    g = hd["hh"].to_numpy()      # split groups: whole households
    u = hd["unit"].to_numpy()    # evaluation units: household-periods with a constant label
    print(f"{len(X):,} household-days, {pd.Series(g).nunique():,} households, {X.shape[1]} features")

    # Hold out 20% of households once.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(outer.split(X, y, g))
    Xtr, ytr, gtr, utr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], g[tr_idx], u[tr_idx]
    Xte, yte, gte, ute = X.iloc[te_idx].reset_index(drop=True), y[te_idx], g[te_idx], u[te_idx]
    assert not set(gtr) & set(gte), "household leaked across train/test"
    json.dump({"train_households": sorted(set(gtr)), "test_households": sorted(set(gte))},
              open(reports_dir / "split.json", "w"))
    print(f"train: {pd.Series(gtr).nunique()} households / test: {pd.Series(gte).nunique()} households")

    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, ytr, gtr))

    # 1) Compare model families with default settings.
    rows, oof = [], {}
    for name, model in base_models().items():
        s = time.time()
        make = lambda m=model: __import__("sklearn").base.clone(m)
        p_oof = oof_predict(make, Xtr, ytr, cv, name)
        m = fit(make(), Xtr, ytr, name)
        p_te = m.predict_proba(Xte)[:, 1]
        cv_s, te_s = household_scores(ytr, p_oof, utr), household_scores(yte, p_te, ute)
        rows.append({"model": name, "stage": "default",
                     **{f"cv_{k}": v for k, v in cv_s.items()}, **{f"test_{k}": v for k, v in te_s.items()},
                     "seconds": round(time.time() - s, 1)})
        oof[name] = p_oof
        print(f"[default] {name:22s} CV hh_acc={cv_s['hh_accuracy']:.4f} AUC={cv_s['hh_auc']:.4f} | "
              f"test hh_acc={te_s['hh_accuracy']:.4f} errors={te_s['hh_errors']} ({time.time() - s:.0f}s)", flush=True)

    # 2) Tune the strongest families with Optuna on grouped folds of the training households.
    rng = np.random.default_rng(SEED)
    tune_hh = set(rng.choice(sorted(set(gtr)), int(len(set(gtr)) * a.tune_households), replace=False))
    tmask = np.isin(gtr, list(tune_hh))
    Xt, yt, gt = Xtr[tmask].reset_index(drop=True), ytr[tmask], gtr[tmask]
    tcv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1).split(Xt, yt, gt))
    tuned, params = {}, {}
    for name, n_trials in [("LightGBM", a.trials), ("XGBoost", a.trials),
                           ("CatBoost", max(6, a.trials // 2)), ("Extra Trees", max(6, a.trials // 2))]:
        s = time.time()
        model, best_params, best_loss = tune(name, Xt, yt, tcv, n_trials)
        tuned[name], params[name] = model, best_params
        clone = __import__("sklearn").base.clone
        p_oof = oof_predict(lambda m=model: clone(m), Xtr, ytr, cv)
        p_te = clone(model).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        cv_s, te_s = household_scores(ytr, p_oof, utr), household_scores(yte, p_te, ute)
        rows.append({"model": f"{name} (tuned)", "stage": "tuned",
                     **{f"cv_{k}": v for k, v in cv_s.items()}, **{f"test_{k}": v for k, v in te_s.items()},
                     "seconds": round(time.time() - s, 1)})
        oof[f"{name} (tuned)"] = p_oof
        print(f"[tuned]   {name:22s} best logloss={best_loss:.4f} CV hh_acc={cv_s['hh_accuracy']:.4f} | "
              f"test hh_acc={te_s['hh_accuracy']:.4f} errors={te_s['hh_errors']} ({time.time() - s:.0f}s)", flush=True)
    json.dump(params, open(reports_dir / "best_params.json", "w"), indent=2)

    # 3) Stacked ensemble: tuned base models + logistic regression meta-learner,
    #    with the meta-features produced on household-grouped folds (no within-household leakage).
    clone = __import__("sklearn").base.clone
    names = list(tuned)
    base_oof = np.column_stack([oof[f"{n} (tuned)"] for n in names])
    meta_oof = np.zeros(len(Xtr))
    for tr, va in cv:  # second-level CV on the OOF probabilities estimates the stack's CV score
        meta_oof[va] = LogisticRegression(max_iter=1000).fit(base_oof[tr], ytr[tr]).predict_proba(base_oof[va])[:, 1]
    vote_oof = base_oof.mean(axis=1)

    s = time.time()
    stack = StackingClassifier(
        estimators=[(n.replace(" ", "_").lower(), clone(tuned[n])) for n in names],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv, stack_method="predict_proba", n_jobs=1, passthrough=False)
    stack.fit(Xtr, ytr)
    p_stack = stack.predict_proba(Xte)[:, 1]
    p_vote = np.column_stack([e.predict_proba(Xte)[:, 1] for e in stack.estimators_]).mean(axis=1)
    for label, p_cv, p_test in [("Soft-voting ensemble", vote_oof, p_vote),
                                ("Stacking ensemble (LR meta)", meta_oof, p_stack)]:
        cv_s, te_s = household_scores(ytr, p_cv, utr), household_scores(yte, p_test, ute)
        rows.append({"model": label, "stage": "ensemble",
                     **{f"cv_{k}": v for k, v in cv_s.items()}, **{f"test_{k}": v for k, v in te_s.items()},
                     "seconds": round(time.time() - s, 1)})
        print(f"[ensemble] {label:22s} CV hh_acc={cv_s['hh_accuracy']:.4f} | test hh_acc={te_s['hh_accuracy']:.4f} "
              f"errors={te_s['hh_errors']}", flush=True)

    # 4) Final model = stacking ensemble; threshold chosen on training OOF only.
    thr = best_threshold(ytr, meta_oof, utr)
    final_cv = household_scores(ytr, meta_oof, utr, thr)
    final_te = household_scores(yte, p_stack, ute, thr)
    print(f"Threshold (train OOF) = {thr:.2f}; test hh_acc={final_te['hh_accuracy']:.5f} "
          f"errors={final_te['hh_errors']}/{final_te['hh_n']}")

    comp = pd.DataFrame(rows)
    comp.to_csv(reports_dir / "model_comparison.csv", index=False)

    hd_te = hd.iloc[te_idx].reset_index(drop=True)
    hh_te = (pd.DataFrame({"unit": ute, "hh": gte, "y": yte, "p_day": p_stack,
                           "export_kwh": hd_te["export_kwh"].values, "first_export_day": hd_te["first_export_day"].values})
             .groupby("unit").agg(hh=("hh", "first"), solar_installed=("y", "first"), p_solar=("p_day", "mean"),
                                  n_days=("p_day", "size"), export_kwh=("export_kwh", "first"),
                                  first_export_day=("first_export_day", "first")))
    hh_te["pred"] = (hh_te.p_solar >= thr).astype(int)
    hh_te.to_csv(reports_dir / "test_household_predictions.csv")
    pd.DataFrame({"hh": gte, "unit": ute, "day": hd_te["day"].values, "y": yte, "p": p_stack}).to_parquet(
        reports_dir / "test_day_predictions.parquet", index=False)

    detector = SolarDetector(day_model=stack, feature_names=FEATURE_NAMES, threshold=thr, info={
        "trained_on_households": int(pd.Series(gtr).nunique()), "evaluation_unit": "household-period", "base_models": names, "best_params": params,
        "cv": final_cv, "test": final_te})
    joblib.dump(detector, models_dir / "solar_detector.joblib", compress=3)
    json.dump({"model": "Stacking ensemble (LightGBM, XGBoost, CatBoost, Extra Trees -> Logistic Regression)",
               "threshold": thr, "features": FEATURE_NAMES, "hours": HOURS,
               "cv_train_households": final_cv, "test_households": final_te,
               "export_threshold_kwh": 1.0, "minutes": round((time.time() - t0) / 60, 1)},
              open(models_dir / "metadata.json", "w"), indent=2, default=float)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
