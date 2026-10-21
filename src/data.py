"""Build the household-day dataset from the RECON-SL 15-minute smart meter files.

Source: LIRNEasia, "Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)",
https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption

Usage:
    python -m src.data --zip data/raw/recon_sl.zip --out data/processed/household_days.parquet
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import HOURS

MEMBERS = [
    f"data/data/consumption_data/smart_meter/15min_interval/smart_15min_{i}.csv" for i in (1, 2, 3)
]
STEP_MIN = 15
MAX_STEP_KWH = 10.0  # a 15-minute step above this is a meter glitch, not consumption
EXPORT_THRESHOLD_KWH = 1.0  # fixed before modelling: a household "has solar" if it exported more than this
FIRST_EXPORT_KWH = 0.1  # a day with more than this exported marks the system as installed
MIN_STEPS_PER_DAY = 90  # out of 96; keeps only near-complete days
EPOCH = pd.Timestamp("2023-01-01")


def read_member(zf: zipfile.ZipFile, member: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Read one 15-minute file with compact dtypes: household, minute index, import and export kWh."""
    cols = ["household_ID", "date", "time", "importkwh(kwh)", "exportkwh(kwh)"]
    parts = []
    with zf.open(member) as fh:
        for ch in pd.read_csv(fh, usecols=cols, chunksize=chunksize,
                              dtype={"importkwh(kwh)": "float32", "exportkwh(kwh)": "float32"}):
            ts = pd.to_datetime(ch["date"] + " " + ch["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            ok = ts.notna()
            parts.append(pd.DataFrame({
                "hh": ch.loc[ok, "household_ID"].astype("category"),
                "t": ((ts[ok] - EPOCH).dt.total_seconds() // 60).astype("int32"),
                "imp": ch.loc[ok, "importkwh(kwh)"],
                "exp": ch.loc[ok, "exportkwh(kwh)"],
            }))
            print(f"  {member.rsplit('/', 1)[-1]}: {sum(len(p) for p in parts):,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["hh"] = df["hh"].astype(str).astype("category")
    return df


def to_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative meter registers into clean 15-minute energy steps."""
    df = df.sort_values(["hh", "t"]).drop_duplicates(["hh", "t"])
    g = df.groupby("hh", observed=True)
    gap = g["t"].diff()
    d_imp = g["imp"].diff()
    d_exp = g["exp"].diff()
    clean = gap.eq(STEP_MIN)
    df["d_imp"] = d_imp.where(clean & d_imp.between(0, MAX_STEP_KWH))
    df["d_exp"] = d_exp.where(clean & d_exp.between(0, MAX_STEP_KWH))
    # Each step is the energy of the 15 minutes *ending* at t, so attribute it to t - 1 minute.
    end = EPOCH + pd.to_timedelta(df["t"] - 1, unit="m")
    df["day"] = end.dt.normalize()
    df["hour"] = end.dt.hour.astype("int8")
    return df


def household_days(steps: pd.DataFrame, min_steps: int = MIN_STEPS_PER_DAY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate to one row per household-day (24 hourly import values) plus per-household export totals."""
    s = steps.dropna(subset=["d_imp"])
    counts = s.groupby(["hh", "day"], observed=True)["d_imp"].count()
    hourly = s.pivot_table(index=["hh", "day"], columns="hour", values="d_imp", aggfunc="sum", observed=True)
    hourly = hourly.reindex(columns=range(24))
    hourly.columns = HOURS
    hourly = hourly[counts.reindex(hourly.index) >= min_steps].dropna()

    daily_exp = steps.groupby(["hh", "day"], observed=True)["d_exp"].sum(min_count=1)
    hourly["export_kwh_day"] = daily_exp.reindex(hourly.index).fillna(0.0)
    exp_tot = steps.groupby("hh", observed=True)["d_exp"].agg(export_kwh="sum", export_steps="count")
    return hourly, exp_tot


def readings_to_days(readings: pd.DataFrame, min_steps: int = MIN_STEPS_PER_DAY) -> pd.DataFrame:
    """Turn one household's raw 15-minute cumulative import readings into hourly day profiles.

    Expects columns `timestamp` and `import_kwh` (the cumulative import register). Used by the app.
    """
    ts = pd.to_datetime(readings["timestamp"], errors="coerce")
    df = pd.DataFrame({"hh": "uploaded", "t": ((ts - EPOCH).dt.total_seconds() // 60),
                       "imp": pd.to_numeric(readings["import_kwh"], errors="coerce"), "exp": 0.0}).dropna()
    df["t"] = df["t"].astype("int64")
    hd, _ = household_days(to_steps(df), min_steps=min_steps)
    return hd.reset_index().drop(columns=["export_kwh_day"])


def build(zip_path: str, export_threshold_kwh: float = EXPORT_THRESHOLD_KWH,
          first_export_kwh: float = FIRST_EXPORT_KWH, min_steps_per_day: int = MIN_STEPS_PER_DAY) -> pd.DataFrame:
    days, exports = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for member in MEMBERS:
            print(f"Reading {member}", flush=True)
            steps = to_steps(read_member(zf, member))
            hd, ex = household_days(steps, min_steps=min_steps_per_day)
            del steps
            days.append(hd)
            exports.append(ex)
            print(f"  -> {len(hd):,} household-days, {hd.index.get_level_values('hh').nunique():,} households", flush=True)

    hd = pd.concat(days)
    hd = hd[~hd.index.duplicated(keep="first")].reset_index()
    hd["hh"] = hd["hh"].astype(str)
    exp = pd.concat(exports).reset_index()
    exp["hh"] = exp["hh"].astype(str)
    exp = exp.groupby("hh")[["export_kwh", "export_steps"]].sum()

    # The label comes only from the export register; it never enters the feature matrix.
    exp["has_solar"] = (exp["export_kwh"] > export_threshold_kwh).astype("int8")
    hd = hd.merge(exp, left_on="hh", right_index=True, how="left")
    hd["has_solar"] = hd["has_solar"].fillna(0).astype("int8")
    # Some households install solar part-way through the 15 months. A day counts as "solar installed"
    # from the first day the meter exported energy; earlier days of the same household are non-solar.
    first_export = hd[hd["export_kwh_day"] > first_export_kwh].groupby("hh")["day"].min()
    hd["first_export_day"] = hd["hh"].map(first_export)
    hd["solar_installed"] = ((hd["has_solar"] == 1) & (hd["day"] >= hd["first_export_day"])).astype("int8")
    # Evaluation unit: a household-period with a constant label (one or two per household).
    hd["unit"] = hd["hh"] + ":" + hd["solar_installed"].astype(str)
    hd["month"] = hd["day"].dt.month.astype("int8")
    for c in HOURS:
        hd[c] = hd[c].astype("float32")
    return hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/recon_sl.zip")
    ap.add_argument("--out", default="data/processed/household_days.parquet")
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()
    params = {}
    if Path(a.params).exists():
        import yaml
        params = yaml.safe_load(open(a.params)).get("data", {})
    hd = build(a.zip, **params)
    hd.to_parquet(a.out, index=False)
    hh = hd.groupby("hh")["has_solar"].first()
    print(f"Saved {len(hd):,} household-days for {len(hh):,} households "
          f"({int(hh.sum())} with solar, {hh.mean():.1%}) to {a.out}")
    print(f"Solar-installed days: {hd['solar_installed'].mean():.1%}; evaluation units: {hd['unit'].nunique():,}")


if __name__ == "__main__":
    main()
