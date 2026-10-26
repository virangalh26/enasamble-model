import numpy as np
import pandas as pd

from src.features import AUDIT, CATEGORICAL, model_columns, to_matrix
from src.hies import check_against_dcs, find_file, monthly_consumption


def test_one_row_per_household_with_label(pipeline):
    hh = pipeline["households"]
    assert not hh.duplicated(["psu", "snumber", "hhno"]).any()
    assert set(hh["poor"].unique()) <= {0, 1}
    assert ((hh["pc_consumption"] < 6966) == (hh["poor"] == 1)).all()


def test_target_and_sensitive_columns_are_never_features(pipeline):
    cols = model_columns(pipeline["households"])
    for c in ["poor", "pc_consumption", "log_pc_consumption", "consumption_month", "weight", *AUDIT]:
        assert c not in cols


def test_matrix_is_numeric_and_aligns_to_training_columns(pipeline):
    hh = pipeline["households"]
    X = to_matrix(hh)
    assert np.isfinite(X.to_numpy()).all()
    assert any(c.startswith("district=") for c in X.columns)
    one = to_matrix(hh.head(1), list(X.columns))
    assert list(one.columns) == list(X.columns) and len(one) == 1


def test_unseen_category_gives_all_zero_dummies(pipeline):
    hh = pipeline["households"]
    cols = list(to_matrix(hh).columns)
    odd = hh.head(1).copy()
    odd["walls"] = 99
    X = to_matrix(odd, cols)
    assert X[[c for c in cols if c.startswith("walls=")]].to_numpy().sum() == 0


def test_monthly_consumption_adds_food_and_nonfood():
    key = {"psu": [1, 1], "snumber": [1, 2], "hhno": [1, 1]}
    food = pd.DataFrame({**key, "value": [100, 50], "inkind_value": [10, None]})
    nonfood = pd.DataFrame({"psu": [1], "snumber": [1], "hhno": [1], "nf_value": [200], "nf_inkind_value": [0]})
    c = monthly_consumption(food, nonfood, food_factor=2.0, nonfood_factor=1.0)
    assert c.loc[(1, 1, 1)] == 2 * 110 + 200 and c.loc[(1, 2, 1)] == 100


def test_find_file_ignores_suffix_and_prefers_csv(tmp_path):
    (tmp_path / "SEC_8_HOUSING_25%.sav").write_bytes(b"")
    (tmp_path / "SEC_8_HOUSING_25%.csv").write_text("a\n1\n")
    assert find_file(tmp_path, "SEC_8_HOUSING").suffix == ".csv"


def test_label_check_table_has_every_district(pipeline):
    chk = check_against_dcs(pipeline["households"])
    assert len(chk) == 26 and chk.loc["Sri Lanka", "dcs_pct"] == 11.9
