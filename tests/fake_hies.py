"""Write small synthetic files with the HIES 2019 public-use layout, for tests only.

The real microdata cannot be committed (DCS terms), so tests and CI run the pipeline on this stand-in.
Poorer fake households get worse housing and fewer durables, so models have a signal to learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import DURABLES, FACILITY_TIME, FACILITY_YES, HOUSING_CAT, HOUSING_YES
from src.hies import DISTRICTS


def write_fake_hies(out: Path, n_households: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_psu = max(n_households // 10, 4)
    psu = np.arange(1, n_psu + 1)
    psu_district = rng.choice(list(DISTRICTS), size=n_psu)
    psu_sector = rng.choice([1, 2, 3], size=n_psu, p=[0.2, 0.7, 0.1])
    hh_psu = rng.choice(psu, size=n_households)
    hh = pd.DataFrame({"psu": hh_psu, "snumber": np.arange(n_households) % 10 + 1, "hhno": 1})
    hh = hh.drop_duplicates(["psu", "snumber", "hhno"]).reset_index(drop=True)
    n = len(hh)
    hh["district"] = psu_district[hh["psu"] - 1]
    hh["sector"] = psu_sector[hh["psu"] - 1]
    wealth = rng.normal(0, 1, n) - 0.6 * (hh["sector"] == 3)
    size = np.clip(rng.poisson(3.5, n) + (wealth < -1), 1, 10)
    base = hh[["district", "sector", "psu", "snumber", "hhno"]].assign(month=1, nhh=1, result=1)

    rows = []
    for i, s in enumerate(size):
        for j in range(s):
            age = rng.integers(30, 70) if j == 0 else rng.integers(0, 80)
            rows.append({**base.iloc[i].to_dict(), "person_serial_no": j + 1, "relationship": 1 if j == 0 else 3,
                         "sex": rng.integers(1, 3), "age": age, "ethnicity": rng.integers(1, 5),
                         "religion": rng.integers(1, 5), "marital_status": rng.integers(1, 4),
                         "education": int(np.clip(8 + 3 * wealth[i] + rng.normal(0, 2), 0, 19)) if age >= 6 else None,
                         "employment_status": rng.integers(1, 7) if 18 <= age < 60 else None})
    pd.DataFrame(rows).to_csv(out / "SEC_1_DEMOGRAPHIC_25%.csv", index=False)

    housing = base.copy()
    for c in HOUSING_CAT:
        housing[c] = np.where(wealth > 0, 1, rng.integers(2, 5, n))
    housing["bed_rooms"] = np.clip(np.round(2 + wealth + rng.normal(0, 0.5, n)), 1, 6)
    housing["area"] = np.clip(np.round(3 + wealth), 1, 7)
    for c in HOUSING_YES:
        housing[c] = rng.integers(1, 3, n)
    housing.to_csv(out / "SEC_8_HOUSING_25%.csv", index=False)

    fac = base.copy()
    for c in FACILITY_TIME:
        fac[c] = np.clip(rng.normal(20 - 5 * wealth, 5), 1, 120)
    for c in FACILITY_YES:
        fac[c] = rng.integers(1, 3, n)
    fac.to_csv(out / "SEC_7_BASIC_FACILITIES_25%.csv", index=False)

    dur = base.copy()
    for k, c in enumerate(DURABLES):
        dur[c] = np.where(rng.random(n) < 1 / (1 + np.exp(-(wealth - k / 10))), 1, 2)
    dur.to_csv(out / "SEC_6_A_DURABLE_GOODS_25%.csv", index=False)

    base.assign(is_agriland_owner=rng.integers(1, 3, n)).to_csv(out / "SEC_9_IS_A_LAND_25%.csv", index=False)
    base.assign(landnumber=1, land_own_perch=rng.integers(5, 80, n)).to_csv(out / "SEC_9_LAND_ANIMAL_25%.csv",
                                                                            index=False)
    base.assign(s9_cattle_buffaloes=2, cows_count=0, goats_sheeps=2, goat_count=0, pigs=2, pigs_count=0,
                chickens=rng.integers(1, 3, n), chicken_count=rng.integers(0, 10, n)) \
        .to_csv(out / "SEC_9_2_OWNED_LIVESTOCKS_25%.csv", index=False)

    pc = np.exp(9.4 + 0.5 * wealth + rng.normal(0, 0.3, n))  # monthly Rs per person, ~15% below 6,966
    food = base.loc[base.index.repeat(3)].assign(code=1, quantity=1,
                                                  value=np.repeat(0.5 * pc * size / 3, 3), inkind_value=0)
    food.to_csv(out / "SEC_4_1_FOOD_EXP_25%.csv", index=False)
    nonfood = base.assign(nf_code=1, nf_quantity=1, nf_value=0.5 * pc * size, nf_inkind_value=0)
    nonfood.to_csv(out / "SEC_4_2_NONFOOD_25%.csv", index=False)
    pd.DataFrame({"district": psu_district, "sector": psu_sector, "month": 1, "psu": psu,
                  "finalweight_25per": rng.uniform(100, 400, n_psu)}).to_csv(out / "weight_2019_25%.csv", index=False)
    return out
