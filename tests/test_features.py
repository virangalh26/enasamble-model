import numpy as np
import pandas as pd

from src.data import readings_to_days
from src.features import FEATURE_NAMES, HOURS, make_features


def day(values):
    return pd.DataFrame([values], columns=HOURS)


def test_feature_names_are_stable_and_import_only():
    X = make_features(day([0.2] * 24))
    assert list(X.columns) == FEATURE_NAMES
    assert not any("export" in c for c in FEATURE_NAMES), "export readings must never be features"
    assert "month" not in FEATURE_NAMES


def test_hour_shares_sum_to_one():
    X = make_features(day(np.random.default_rng(0).uniform(0.1, 1.0, 24)))
    shares = X[[f"{h}_share" for h in HOURS]].sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_zero_day_has_no_nans_or_infs():
    X = make_features(day([0.0] * 24))
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_solar_shape_scores_lower_midday_ratio():
    solar = make_features(day([0.6] * 8 + [0.02] * 8 + [0.7] * 8))
    flat = make_features(day([0.3] * 24))
    assert solar["midday_to_night"].iloc[0] < flat["midday_to_night"].iloc[0]
    assert solar["bowl_depth"].iloc[0] > flat["bowl_depth"].iloc[0]


def test_readings_to_days_builds_hourly_profiles():
    ts = pd.date_range("2024-05-01 00:15", periods=96 * 2, freq="15min")
    reg = np.cumsum(np.full(len(ts), 0.05)) + 1000.0
    days = readings_to_days(pd.DataFrame({"timestamp": ts, "import_kwh": reg}))
    # The first reading has no previous one, so day 1 has 95 clean steps (kept, >= 90) and day 2 has 96.
    assert len(days) == 2
    assert np.allclose(days[HOURS].sum(axis=1), [0.05 * 95, 0.05 * 96])


def test_readings_to_days_drops_glitches_and_incomplete_days():
    ts = pd.date_range("2024-05-01 00:15", periods=96, freq="15min")
    reg = np.cumsum(np.full(len(ts), 0.05)) + 1000.0
    reg[50:] -= 500  # a meter reset: the negative jump must be discarded, not counted
    days = readings_to_days(pd.DataFrame({"timestamp": ts, "import_kwh": reg}))
    assert (days[HOURS].to_numpy() >= 0).all()
    short = readings_to_days(pd.DataFrame({"timestamp": ts[:40], "import_kwh": reg[:40]}))
    assert short.empty
