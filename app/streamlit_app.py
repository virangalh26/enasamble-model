"""Rooftop solar detector — Streamlit app.

Run locally:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import readings_to_days  # noqa: E402
from src.features import HOURS, make_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "solar_detector.joblib"
META_PATH = ROOT / "models" / "metadata.json"
SAMPLE_CSV = ROOT / "app" / "sample_meter_readings.csv"

# Median hourly import (kWh) per class in the RECON-SL training households, used as starting points for the
# single-day form. Medians, not means: on a sunny day a solar home imports exactly zero at midday, which an
# average across many days blurs into a small positive value.
PRESETS = {
    "Typical day, home with solar": [0.429, 0.385, 0.361, 0.349, 0.359, 0.421, 0.312, 0.059, 0.001, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.047, 0.220, 0.424, 0.479, 0.497, 0.528, 0.540, 0.484],
    "Typical day, home without solar": [0.116, 0.111, 0.108, 0.107, 0.110, 0.120, 0.120, 0.100, 0.088, 0.087, 0.088,
                                        0.091, 0.096, 0.097, 0.099, 0.099, 0.099, 0.105, 0.132, 0.148, 0.152, 0.150,
                                        0.139, 0.123],
    "Empty house (fridge only)": [0.05] * 24,
}

st.set_page_config(page_title="Rooftop Solar Detector — Sri Lanka", page_icon="☀️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


@st.cache_resource
def tree_explainer(_detector):
    """Tree SHAP on the gradient-boosting base learner with the largest stacking weight."""
    stack = _detector.day_model
    names = [n for n, _ in stack.estimators]
    coef = pd.Series(stack.final_estimator_.coef_.ravel()[: len(names)], index=names)
    gb = [n for n in ("lightgbm", "xgboost", "catboost") if n in names]
    name = coef[gb].idxmax()
    return name, shap.TreeExplainer(stack.estimators_[names.index(name)])


def shap_waterfall(detector, X_row):
    name, explainer = tree_explainer(detector)
    sv = explainer(X_row)
    if sv.values.ndim == 3:
        sv = sv[..., 1]
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    plt.title(f"Why? SHAP contributions ({name} base learner, log-odds)")
    return plt.gcf(), name


def lime_explanation(detector, X_row):
    from lime.lime_tabular import LimeTabularExplainer
    rng = np.random.default_rng(0)
    # Background: perturbations of the presets give LIME a sensible feature range without shipping training data.
    base = pd.DataFrame([v for v in PRESETS.values()], columns=HOURS)
    noise = pd.concat([base] * 200, ignore_index=True) * rng.lognormal(0, 0.5, (600, 24))
    bg = make_features(noise)[detector.feature_names]
    exp = LimeTabularExplainer(bg.values, feature_names=list(bg.columns), class_names=["no solar", "solar"],
                               discretize_continuous=True, random_state=0)
    e = exp.explain_instance(X_row.iloc[0].values, lambda x: detector.day_model.predict_proba(
        pd.DataFrame(x, columns=X_row.columns)), num_features=8, num_samples=2000)
    return pd.DataFrame(e.as_list(label=1), columns=["condition", "weight (towards solar)"])


def profile_chart(values, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(24), values, color=["#E8A33D" if 9 <= h <= 15 else "#4C72B0" for h in range(24)])
    ax.set(xticks=range(0, 24, 2), xlabel="hour of day", ylabel="kWh imported", title=title)
    ax.axvspan(8.5, 15.5, color="#E8A33D", alpha=0.08)
    fig.tight_layout()
    return fig


detector, meta = load_model()
test = meta["test_households"]

st.title("☀️ Rooftop Solar Detector")
st.markdown(
    "Detects whether a Sri Lankan household has **rooftop solar** using only its **grid import** smart-meter "
    "readings. Trained on LIRNEasia's RECON-SL data (LECO customers, Oct 2023 – Dec 2024) with a stacking "
    "ensemble of LightGBM, XGBoost, CatBoost and Extra Trees.")

with st.sidebar:
    st.header("Model")
    st.markdown(f"**{meta['model']}**")
    st.metric("Household accuracy (held-out test)", f"{test['hh_accuracy']:.2%}")
    st.metric("Errors on test households", f"{test['hh_errors']} / {test['hh_n']}")
    st.metric("Household AUC", f"{test['hh_auc']:.4f}")
    st.caption(f"Decision threshold {meta['threshold']:.2f}, chosen on training households only. "
               "Export readings are never used as inputs.")

tab1, tab2 = st.tabs(["Check one day", "Check a household (upload readings)"])

with tab1:
    st.subheader("Enter one day of grid import (kWh per hour)")
    preset = st.selectbox("Start from", list(PRESETS))
    cols = st.columns(6)
    values = []
    for h in range(24):
        with cols[h % 6]:
            values.append(st.number_input(f"{h:02d}:00", min_value=0.0, max_value=50.0, value=float(PRESETS[preset][h]),
                                          step=0.01, format="%.3f", key=f"{preset}-{h}"))
    day = pd.DataFrame([values], columns=HOURS)
    X = detector.features(day)
    p = float(detector.day_model.predict_proba(X)[:, 1][0])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Prediction")
        if p >= meta["threshold"]:
            st.success(f"**Solar installed** — P(solar) = {p:.1%}")
        else:
            st.info(f"**No solar** — P(solar) = {p:.1%}")
        st.caption("One day can be ambiguous (cloudy day, empty house). Upload several days in the next tab for a "
                   "household-level decision.")
    with c2:
        st.pyplot(profile_chart(values, "Your day (shaded = solar hours)"))

    e1, e2 = st.columns([3, 2])
    with e1:
        fig, name = shap_waterfall(detector, X)
        st.pyplot(fig)
    with e2:
        if st.button("Explain with LIME"):
            with st.spinner("Fitting a local surrogate model..."):
                st.dataframe(lime_explanation(detector, X), use_container_width=True)
        st.markdown("**Reading the explanation:** low daytime import relative to the night "
                    "(`daytime_min_share`, `midday_to_evening`) pushes towards solar; a flat profile pushes away.")

with tab2:
    st.subheader("Upload 15-minute meter readings for one household")
    st.markdown("CSV with columns **`timestamp`** and **`import_kwh`** (the cumulative import register, as read "
                "from the smart meter). Days with fewer than 90 of 96 readings are skipped.")
    up = st.file_uploader("Meter readings CSV", type="csv")
    if up is None and SAMPLE_CSV.exists() and st.checkbox("Use the bundled sample household"):
        up = SAMPLE_CSV
    if up is not None:
        readings = pd.read_csv(up)
        missing = {"timestamp", "import_kwh"} - set(readings.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        else:
            days = readings_to_days(readings)
            if days.empty:
                st.warning("No complete days found in the file.")
            else:
                days["p_solar"] = detector.predict_day_proba(days)
                hh_p = days["p_solar"].mean()
                verdict = "Solar installed" if hh_p >= meta["threshold"] else "No solar"
                (st.success if hh_p >= meta["threshold"] else st.info)(
                    f"**{verdict}** — mean P(solar) over {len(days)} complete days = {hh_p:.1%}")
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.plot(days["day"], days["p_solar"], marker=".", lw=1)
                ax.axhline(meta["threshold"], color="red", ls="--", lw=1)
                ax.set(ylim=(-0.02, 1.02), ylabel="P(solar)", title="Day-by-day prediction")
                fig.autofmt_xdate(); fig.tight_layout()
                st.pyplot(fig)
                st.pyplot(profile_chart(days[HOURS].mean().values, "Average day for this household"))
                st.dataframe(days[["day", "p_solar"] + HOURS].round(3), use_container_width=True)
