"""Feature engineering shared by training and the Streamlit app.

Input is one household-day of grid *import* energy per hour (24 values, kWh).
Export readings are never used as features: they would reveal the label directly.
Calendar month is deliberately left out: in RECON-SL the share of solar days per month (7%-53%) reflects which
meters happened to report that month, not sunlight, so a month feature would learn sampling artefacts.
"""
import numpy as np
import pandas as pd

HOURS = [f"h{h:02d}" for h in range(24)]
DAYTIME = ["h09", "h10", "h11", "h12", "h13", "h14", "h15"]
MIDDAY = ["h10", "h11", "h12", "h13", "h14"]
MORNING = ["h06", "h07", "h08"]
EVENING = ["h18", "h19", "h20", "h21"]
NIGHT = ["h00", "h01", "h02", "h03", "h04"]
EPS = 1e-3


def make_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Turn a frame with columns h00..h23 (kWh imported per hour) into model features."""
    hd = hourly[HOURS].astype(float)
    total = hd.sum(axis=1)
    X = pd.DataFrame(index=hd.index)
    X["total_kwh"] = total
    for c in HOURS:
        X[f"{c}_kwh"] = hd[c]
    for c in HOURS:
        X[f"{c}_share"] = hd[c] / total.clip(lower=EPS)

    midday = hd[MIDDAY].sum(axis=1)
    night = hd[NIGHT].sum(axis=1)
    evening = hd[EVENING].sum(axis=1)
    morning = hd[MORNING].sum(axis=1)
    X["midday_kwh"] = midday
    X["night_kwh"] = night
    X["evening_kwh"] = evening
    X["midday_to_night"] = midday / night.clip(lower=EPS)
    X["midday_to_evening"] = midday / evening.clip(lower=EPS)
    X["midday_to_morning"] = midday / morning.clip(lower=EPS)
    X["daytime_min_kwh"] = hd[DAYTIME].min(axis=1)
    X["daytime_min_share"] = X["daytime_min_kwh"] / total.clip(lower=EPS)
    X["daytime_zero_hours"] = (hd[DAYTIME] < 0.01).sum(axis=1)
    X["night_min_kwh"] = hd[NIGHT].min(axis=1)
    X["daytime_min_to_night_min"] = X["daytime_min_kwh"] / X["night_min_kwh"].clip(lower=EPS)
    X["peak_hour"] = hd.values.argmax(axis=1)
    X["trough_hour"] = hd.values.argmin(axis=1)
    X["daytime_std"] = hd[DAYTIME].std(axis=1)
    # A solar "bowl": imports fall from morning into midday and recover in the afternoon.
    X["bowl_depth"] = (hd[["h08", "h16"]].mean(axis=1) - hd[["h11", "h12", "h13"]].mean(axis=1)) / total.clip(lower=EPS)
    return X


FEATURE_NAMES = list(make_features(pd.DataFrame([[1.0] * 24], columns=HOURS)).columns)
