"""Proxy features of a household that a welfare officer can observe or ask about on a home visit.

Shared by the data pipeline (`src.hies`), training and the web app. Income and expenditure are never
features: they define the target. Ethnicity and religion are kept only for the fairness audit, never
given to the model.
"""
import numpy as np
import pandas as pd

YES = 1  # HIES yes/no questions: 1 = yes, 2 = no

DURABLES = ["radio", "tv", "vcd", "sewingmechine", "washing_mechine", "fridge", "cookers", "electric_fans",
            "telephone", "telephone_mobile", "computers", "camera", "s6a_aircon", "bicycle", "motor_bicycle",
            "three_wheeler", "motor_car_van", "bus_lorry", "tractor_2_wheel", "tractor_4_wheel", "pesticider",
            "threshers", "waterpumps", "mechine", "boats", "fishing_nets"]
HOUSING_CAT = ["structure", "walls", "floor", "roof", "ownership", "drinking_water", "toilet_type",
               "tioilet_use", "lite_source", "cooking_fuel", "garbage_dumping"]
HOUSING_NUM = ["bed_rooms", "area"]
HOUSING_YES = ["is_collect_firewood", "natural_calamity", "flooding", "drought", "land_slides", "wildanimals"]
FACILITY_TIME = ["bus_halt_time", "primery_school_time", "sec_school_time", "hospital_time",
                 "gov_dispensary_time", "ds_office_time", "gn_office_time", "post_office_time", "bank_time"]
FACILITY_YES = ["is_power_lines_near", "is_tel_lines_near", "is_water_service_near"]
LIVESTOCK = {"s9_cattle_buffaloes": "cows_count", "goats_sheeps": "goat_count", "pigs": "pigs_count",
             "chickens": "chicken_count"}

# Nominal codes: one-hot encoded for the model.
CATEGORICAL = ["district", "sector", "head_marital", "head_employment"] + HOUSING_CAT
AUDIT = ["head_ethnicity", "head_religion"]


def _yes(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") == YES).astype(int)


def member_features(m: pd.DataFrame, key: list) -> pd.DataFrame:
    m = m.copy()
    age = pd.to_numeric(m["age"], errors="coerce")
    edu = pd.to_numeric(m["education"], errors="coerce")
    m["child"] = (age < 15).astype(int)
    m["elderly"] = (age >= 60).astype(int)
    m["working_age"] = age.between(15, 59).astype(int)
    m["female"] = (pd.to_numeric(m["sex"], errors="coerce") == 2).astype(int)
    m["employed"] = pd.to_numeric(m["employment_status"], errors="coerce").between(1, 6).astype(int)
    m["edu"] = edu.where(age >= 18)
    g = m.groupby(key)
    out = pd.DataFrame({
        "n_members": g.size(), "n_children": g["child"].sum(), "n_elderly": g["elderly"].sum(),
        "n_working_age": g["working_age"].sum(), "share_female": g["female"].mean(),
        "n_employed": g["employed"].sum(), "max_adult_education": g["edu"].max(),
        "mean_adult_education": g["edu"].mean(),
    })
    out["dependency_ratio"] = (out["n_children"] + out["n_elderly"]) / out["n_working_age"].clip(lower=1)
    out["employed_share"] = out["n_employed"] / out["n_members"]
    head = m[pd.to_numeric(m["relationship"], errors="coerce") == 1].groupby(key).first()
    out["head_female"] = head["female"]
    out["head_age"] = pd.to_numeric(head["age"], errors="coerce")
    out["head_education"] = pd.to_numeric(head["education"], errors="coerce")
    out["head_marital"] = pd.to_numeric(head["marital_status"], errors="coerce")
    out["head_employment"] = pd.to_numeric(head["employment_status"], errors="coerce").fillna(0)
    out["head_ethnicity"] = pd.to_numeric(head["ethnicity"], errors="coerce")
    out["head_religion"] = pd.to_numeric(head["religion"], errors="coerce")
    return out


def build_features(t: dict, key: list) -> pd.DataFrame:
    """One row per household from the raw HIES tables (dict of DataFrames, lower-case columns)."""
    f = member_features(t["demographic"], key)
    first = t["demographic"].groupby(key)[["district", "sector"]].first()
    f = f.join(first)

    h = t["housing"].groupby(key).first()
    for c in HOUSING_CAT:
        f[c] = pd.to_numeric(h[c], errors="coerce")
    for c in HOUSING_NUM:
        f[c] = pd.to_numeric(h[c], errors="coerce")
    for c in HOUSING_YES:
        f[c] = _yes(h[c])

    fac = t["facilities"].groupby(key).first()
    for c in FACILITY_TIME:
        f[c] = pd.to_numeric(fac[c], errors="coerce")
    for c in FACILITY_YES:
        f[c] = _yes(fac[c])

    d = t["durables"].groupby(key).first()
    for c in DURABLES:
        f["own_" + c] = _yes(d[c]) if c in d else 0
    f["n_durables"] = f[["own_" + c for c in DURABLES]].sum(axis=1)

    f["owns_agri_land"] = _yes(t["land_flag"].groupby(key)["is_agriland_owner"].first())
    f["land_perches"] = pd.to_numeric(t["land"]["land_own_perch"], errors="coerce") \
        .groupby([t["land"][k] for k in key]).sum()
    ls = t["livestock"].groupby(key).first()
    for flag, count in LIVESTOCK.items():
        f["has_" + flag] = _yes(ls[flag])
        f[count] = pd.to_numeric(ls[count], errors="coerce")

    num = [c for c in f.columns if c not in CATEGORICAL + AUDIT]
    f[num] = f[num].fillna(0)  # missing land/livestock/durable rows mean "none"
    return f.reset_index()


def model_columns(df: pd.DataFrame) -> list:
    """Raw feature columns used by the model (everything except ids, target and audit columns)."""
    skip = {"psu", "snumber", "hhno", "weight", "consumption_month", "pc_consumption", "log_pc_consumption",
            "poor", "district_name", "sector_name", "hh_size"} | set(AUDIT)
    return [c for c in df.columns if c not in skip]


def to_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """One-hot encode nominal codes. With `columns` (from training) the output is aligned to them."""
    X = df[[c for c in model_columns(df)]].copy()
    cats = [c for c in CATEGORICAL if c in X]
    X[cats] = X[cats].fillna(-1).astype(int).astype(str)
    X = pd.get_dummies(X, columns=cats, prefix_sep="=", dtype=float).astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.fillna(0.0)
