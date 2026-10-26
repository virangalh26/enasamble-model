"""Build the household table from the HIES 2019 public-use microdata (25% sample).

Source: Department of Census and Statistics (DCS), Lanka Datta catalog 476,
https://nada.statistics.gov.lk/index.php/catalog/476 (free registration; files are not redistributed).

One row per household with
* proxy features a welfare officer can observe on a home visit (housing, facilities, durable goods,
  land, livestock, household members, location), built by `src.features`, and
* the target: monthly consumption expenditure per person and `poor` = below the official poverty line
  (Rs. 6,966 per person per month in 2019, DCS "Poverty indicators-2019").
Expenditure is used only to build the target, never as a feature.

Usage:
    python -m src.hies --raw data/raw/hies2019 --out data/processed/households.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import build_features

KEY = ["psu", "snumber", "hhno"]  # identifies one household in every HIES file

# File stems as listed in the Lanka Datta data dictionary; the files may come as .csv, .sav or .dta.
FILES = {
    "demographic": "SEC_1_DEMOGRAPHIC",
    "food": "SEC_4_1_FOOD_EXP",
    "nonfood": "SEC_4_2_NONFOOD",
    "durables": "SEC_6_A_DURABLE_GOODS",
    "facilities": "SEC_7_BASIC_FACILITIES",
    "housing": "SEC_8_HOUSING",
    "land_flag": "SEC_9_IS_A_LAND",
    "land": "SEC_9_LAND_ANIMAL",
    "livestock": "SEC_9_2_OWNED_LIVESTOCKS",
    "weight": "weight_2019",
}

# DCS district codes (first digit = province).
DISTRICTS = {
    11: "Colombo", 12: "Gampaha", 13: "Kalutara", 21: "Kandy", 22: "Matale", 23: "Nuwara Eliya",
    31: "Galle", 32: "Matara", 33: "Hambantota", 41: "Jaffna", 42: "Mannar", 43: "Vavuniya",
    44: "Mullaitivu", 45: "Kilinochchi", 51: "Batticaloa", 52: "Ampara", 53: "Trincomalee",
    61: "Kurunegala", 62: "Puttalam", 71: "Anuradhapura", 72: "Polonnaruwa", 81: "Badulla",
    82: "Moneragala", 91: "Ratnapura", 92: "Kegalle",
}
SECTORS = {1: "Urban", 2: "Rural", 3: "Estate"}

# DCS "Poverty indicators-2019", Table 06: % of households below the updated poverty line, by district.
# Used only to check that our poverty label reproduces the official figures.
DCS_HH_POVERTY_2019 = {
    "Sri Lanka": 11.9, "Colombo": 1.8, "Gampaha": 4.2, "Kalutara": 9.5, "Kandy": 11.5, "Matale": 15.6,
    "Nuwara Eliya": 23.9, "Galle": 10.8, "Matara": 9.5, "Hambantota": 12.0, "Jaffna": 20.8, "Mannar": 6.3,
    "Vavuniya": 12.1, "Mullaitivu": 39.5, "Kilinochchi": 23.0, "Batticaloa": 16.2, "Ampara": 14.2,
    "Trincomalee": 15.0, "Kurunegala": 10.7, "Puttalam": 8.1, "Anuradhapura": 7.2, "Polonnaruwa": 13.8,
    "Badulla": 28.0, "Moneragala": 18.4, "Ratnapura": 20.7, "Kegalle": 18.0,
}


def find_file(raw: Path, stem: str) -> Path:
    """Locate a HIES file by its stem, whatever the extension or a '_25%' suffix."""
    hits = sorted(p for p in raw.rglob("*") if p.suffix.lower() in {".csv", ".sav", ".dta"}
                  and p.stem.upper().startswith(stem.upper()))
    if not hits:
        raise FileNotFoundError(f"no file starting with {stem!r} under {raw}")
    return min(hits, key=lambda p: {".csv": 0, ".dta": 1, ".sav": 2}[p.suffix.lower()])


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() == ".dta":
        df = pd.read_stata(path, convert_categoricals=False)
    else:
        import pyreadstat
        df, _ = pyreadstat.read_sav(path, apply_value_formats=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_raw(raw: Path) -> dict:
    return {name: read_table(find_file(raw, stem)) for name, stem in FILES.items()}


def monthly_consumption(food: pd.DataFrame, nonfood: pd.DataFrame, food_factor: float,
                        nonfood_factor: float) -> pd.Series:
    """Household consumption per month: purchased + in-kind food and non-food, scaled to a month."""
    f = food.assign(v=food["value"].fillna(0) + food["inkind_value"].fillna(0)).groupby(KEY)["v"].sum()
    nf = nonfood.assign(v=nonfood["nf_value"].fillna(0) + nonfood["nf_inkind_value"].fillna(0)) \
        .groupby(KEY)["v"].sum()
    total = f.mul(food_factor).add(nf.mul(nonfood_factor), fill_value=0)
    return total.rename("consumption_month")


def build_households(t: dict, poverty_line: float, food_factor: float, nonfood_factor: float) -> pd.DataFrame:
    members = t["demographic"]
    if "result" in members:
        members = members[members["result"].fillna(1) == 1]
    hh = members.groupby(KEY).size().rename("hh_size").reset_index()
    w = t["weight"].groupby("psu")["finalweight_25per"].first()
    hh["weight"] = hh["psu"].map(w)
    cons = monthly_consumption(t["food"], t["nonfood"], food_factor, nonfood_factor)
    hh = hh.merge(cons.reset_index(), on=KEY, how="left")
    hh = hh[hh["consumption_month"] > 0].copy()
    hh["pc_consumption"] = hh["consumption_month"] / hh["hh_size"]
    hh["log_pc_consumption"] = np.log(hh["pc_consumption"])
    hh["poor"] = (hh["pc_consumption"] < poverty_line).astype(int)
    hh = hh.merge(build_features(t, KEY), on=KEY, how="inner")
    hh["district_name"] = hh["district"].map(DISTRICTS)
    hh["sector_name"] = hh["sector"].map(SECTORS)
    return hh


def check_against_dcs(hh: pd.DataFrame) -> pd.DataFrame:
    """Weighted share of poor households by district next to the DCS published figure."""
    w = hh["weight"].fillna(1.0)
    ours = (hh.assign(wp=w * hh["poor"], w=w).groupby("district_name")[["wp", "w"]].sum()
            .pipe(lambda d: 100 * d["wp"] / d["w"]))
    ours.loc["Sri Lanka"] = 100 * (w * hh["poor"]).sum() / w.sum()
    out = pd.DataFrame({"ours_pct": ours.round(1), "dcs_pct": pd.Series(DCS_HH_POVERTY_2019)})
    return out.loc[list(DCS_HH_POVERTY_2019)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/hies2019")
    ap.add_argument("--out", default="data/processed/households.parquet")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    p = yaml.safe_load(open(args.params))["data"]
    hh = build_households(load_raw(Path(args.raw)), p["poverty_line"], p["food_factor"], p["nonfood_factor"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hh.to_parquet(args.out, index=False)
    chk = check_against_dcs(hh)
    Path("reports").mkdir(exist_ok=True)
    chk.to_csv("reports/label_check_vs_dcs.csv")
    corr = chk.drop("Sri Lanka")[["ours_pct", "dcs_pct"]].corr().iloc[0, 1]
    summary = {"households": len(hh), "poor": int(hh["poor"].sum()), "poor_share": round(hh["poor"].mean(), 4),
               "weighted_poor_pct": float(chk.loc["Sri Lanka", "ours_pct"]), "dcs_poor_pct": 11.9,
               "district_rate_correlation": round(float(corr), 3), "features": int(hh.shape[1])}
    json.dump(summary, open("reports/data_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(chk.to_string())


if __name__ == "__main__":
    main()
