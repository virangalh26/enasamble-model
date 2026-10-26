"""Behaviour of the trained NGBoost targeter (trained on synthetic data in the fixture)."""
import json

import joblib
import numpy as np
import pandas as pd


def test_outputs_are_probabilities_with_ordered_interval(pipeline):
    t = joblib.load(pipeline["models"] / "poverty_targeter.joblib")
    out = t.predict(pipeline["households"])
    assert out["p_poor"].between(0, 1).all()
    assert (out["pc_low"] < out["pred_pc_consumption"]).all() and (out["pred_pc_consumption"] < out["pc_high"]).all()
    assert set(out["flag_for_support"].unique()) <= {0, 1}


def test_p_poor_matches_the_normal_distribution(pipeline):
    from scipy.stats import norm
    t = joblib.load(pipeline["models"] / "poverty_targeter.joblib")
    X = t.matrix(pipeline["households"].head(20))
    mu, sigma = t.distribution(X)
    assert np.allclose(t.p_poor_from_matrix(X), norm.cdf((np.log(t.poverty_line) - mu) / sigma))


def test_poorer_households_get_higher_scores(pipeline):
    t = joblib.load(pipeline["models"] / "poverty_targeter.joblib")
    hh = pipeline["households"]
    p = t.predict(hh)["p_poor"]
    assert p[hh["poor"] == 1].mean() > p[hh["poor"] == 0].mean() + 0.2


def test_reports_written(pipeline):
    m = json.load(open(pipeline["reports"] / "metrics.json"))
    assert 0 < m["test"]["pr_auc"] <= 1 and 0 < m["threshold"] < 1
    assert 0 <= m["test_regression"]["coverage_80"] <= 1
    comp = pd.read_csv(pipeline["reports"] / "model_comparison.csv")
    assert comp["model"].str.startswith("NGBoost").sum() >= 2  # model + natural-gradient ablation
    fair = pd.read_csv(pipeline["reports"] / "fairness.csv")
    assert {"sector_name", "head_female"} <= set(fair["group"])
