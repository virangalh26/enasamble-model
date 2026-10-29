"""Behavioural tests for the saved model. Skipped when the model file is not present."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.features import HOURS

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "solar_detector.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not available")

# Median hourly import of training days with and without solar (same values as the app presets).
SOLAR = [0.429, 0.385, 0.361, 0.349, 0.359, 0.421, 0.312, 0.059, 0.001, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.047, 0.220, 0.424, 0.479, 0.497, 0.528, 0.540, 0.484]
FLAT = [0.116, 0.111, 0.108, 0.107, 0.110, 0.120, 0.120, 0.100, 0.088, 0.087, 0.088,
        0.091, 0.096, 0.097, 0.099, 0.099, 0.099, 0.105, 0.132, 0.148, 0.152, 0.150, 0.139, 0.123]


@pytest.fixture(scope="module")
def detector():
    return joblib.load(MODEL)


def test_typical_solar_day_is_flagged(detector):
    p = detector.predict_day_proba(pd.DataFrame([SOLAR], columns=HOURS))[0]
    assert p >= detector.threshold


def test_typical_non_solar_day_is_not_flagged(detector):
    p = detector.predict_day_proba(pd.DataFrame([FLAT], columns=HOURS))[0]
    assert p < detector.threshold


def test_household_aggregation(detector):
    days = pd.DataFrame([SOLAR, SOLAR, FLAT, FLAT], columns=HOURS)
    days["hh"] = ["a", "a", "b", "b"]
    out = detector.predict_household(days)
    assert out.loc["a", "has_solar_pred"] == 1 and out.loc["b", "has_solar_pred"] == 0
    assert (out["n_days"] == 2).all()


def test_metadata_matches_model(detector):
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert meta["features"] == detector.feature_names
    assert abs(meta["threshold"] - detector.threshold) < 1e-9
