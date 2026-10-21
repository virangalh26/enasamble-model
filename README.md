# ☀️ Rooftop Solar Detection from Smart-Meter Data — Sri Lanka

**AML Assignment 2 · Ensemble learning on a real, local dataset**

Detect whether a Sri Lankan household has **rooftop solar** using only its **grid import** smart-meter readings,
with a stacking ensemble (LightGBM, XGBoost, CatBoost, Extra Trees → logistic regression), explained with SHAP and
LIME, served through a Streamlit app, and packaged for deployment on Azure.

| Held-out test (592 unseen households, 608 household-periods) | Result |
|---|---|
| **Household accuracy** | **99.18%** (5 errors / 608) |
| Household AUC | 0.9973 |
| Precision / recall (solar) | 1.000 / 0.951 |
| Day-level accuracy (90,559 test days) | 98.35% |
| Majority-class baseline | 83.06% |

---

## 1. The problem

After the 2022 power crisis and the tariff increases that followed, rooftop solar spread rapidly across Sri Lanka.
Distribution utilities (CEB, LECO) need to know **which homes have solar** to plan feeders, manage reverse power flow
and keep records current, including systems that were never registered. Smart meters record how much energy a
home *imports* every 15 minutes; a solar home's import collapses during the day because the panels cover the load.
The task: **classify a household as solar / non-solar from its import profile alone.**

## 2. Data

**RECON-SL** — *Residential Electricity Consumption Dataset for Sri Lanka*, LIRNEasia (2025).
C. Algama et al., IEEE DataPort, doi:[10.21227/n1dk-q860](https://dx.doi.org/10.21227/n1dk-q860);
[Kaggle mirror](https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption).

* 15-minute smart-meter readings from LECO customers (Moratuwa area), **Oct 2023 – Dec 2024** (~45.6 M readings).
* After cleaning: **452,811 complete household-days** from **2,925 households**, 492 of which export solar energy.
* The raw data is not committed (`data/` is git-ignored); `app/sample_meter_readings.csv` is 4 weeks of import
  readings from one held-out household, used as the app's demo input.

**Label.** Defined from the meter's *export register* (energy sent to the grid) and fixed before modelling:
a household has solar if it exported > 1 kWh; a day counts as "solar installed" from the first day with > 0.1 kWh
exported (55 households installed part-way through). **Export readings are never used as features.**

## 3. Method

| Step | What was done | Where |
|---|---|---|
| EDA | coverage, hourly profiles, distributions, household heat-maps, before/after installation, survey vs meter label agreement | `notebooks/01_eda.ipynb` |
| Preprocessing | cumulative registers → 15-min energy; glitch/reset removal; near-complete days only (≥ 90/96 readings, so no imputation); 24 hourly values per day | `src/data.py` |
| Features | 64 features: hourly kWh, hourly shares (scale-free shape), midday/night/evening ratios, daytime minimum, "solar bowl" depth, peak/trough hour | `src/features.py` |
| Models compared | Dummy, Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost | `src/train.py` |
| Tuning | Optuna (TPE) on household-grouped CV, minimising day-level log-loss | `src/train.py` |
| Ensemble | soft voting and **stacking** (out-of-fold, household-grouped meta-features → logistic regression) | `src/train.py` |
| Household decision | mean of day probabilities; threshold (0.45) chosen on training households only | `src/model.py` |
| Explainability | SHAP (tree SHAP on a base learner + permutation SHAP on the full ensemble), LIME | `src/explain.py`, `notebooks/02_…` |
| App | Streamlit: one-day form with SHAP/LIME, or upload raw readings for a household verdict | `app/streamlit_app.py` |

### Keeping the score honest
* **Grouped splits** — a household's days are entirely in train or test (days of one home are near-duplicates).
* **20% of households held out** and scored once at the end; tuning and the threshold use training households only.
* **Leakage removed** — export energy/power, reactive power and power factor are never read; the 6-hour files' `EXPORT`
  flag and meter type are not used. A first run ranked *calendar month* #2 in SHAP; the solar share per month
  (7–53%) reflects which meters reported, not sunlight, so month was **removed** and the model retrained.
* **No single feature is a perfect proxy** — the best single feature has AUC 0.885 (EDA §5).

## 4. Results

| Model | CV household acc. (train, 5-fold grouped) | Test errors / 608 |
|---|---|---|
| Dummy (majority) | 0.8375 | 103 |
| Logistic Regression | 0.9727 | 14 |
| KNN (k=25) | 0.9794 | 10 |
| Decision Tree | 0.9895 | 8 |
| Random Forest / Extra Trees | 0.9882 / 0.9861 | 7 / 7 |
| HistGB / XGBoost / LightGBM / CatBoost | 0.9895 / 0.9903 / 0.9882 / 0.9899 | 8 / 7 / 7 / 7 |
| Tuned LightGBM / XGBoost / CatBoost / Extra Trees | 0.9895 / 0.9882 / 0.9870 / 0.9887 | 7 each |
| Soft-voting ensemble | 0.9887 | 7 |
| **Stacking ensemble** (threshold 0.45) | **0.9912** | **5** |

* Tree ensembles clearly beat linear, distance-based and baseline models. Among the top models, CV differences are
  only a few households (within noise); the stacking ensemble is at least as good in CV and made the fewest test errors.
* **Tuning did not improve the boosting models** (searching on a 25% household subsample chose slower learners that
  under-fit at full scale) — discussed in notebook 02 §2.
* **Error analysis** (notebook 02 §5): all 5 test errors are missed solar homes (no false alarms). They are a home with
  only 3 days of data, a very small exporter (93 kWh in 88 days), short post-installation periods, and one home whose
  import shape barely changes despite exporting (likely battery/hybrid system). Households with > 30 days of data:
  **99.66%** (2 errors / 582).
* **SHAP / LIME**: predictions are driven by *how low daytime import gets relative to the rest of the day*
  (`daytime_min_to_night_min`, `daytime_min_share`) — the physical solar signature — consistently for the base learner
  and the full ensemble.

### Why not 99.97%?
On the test set, 99.97% would mean fewer than one error in 608 household-periods. The remaining errors are genuinely
hard cases (a few days of data, tiny or battery-backed systems), and the label itself has physical limits (an off-grid
system never exports). We report the honest held-out score rather than tuning to the test set.

## 5. Project layout

```
src/        data.py (pipeline) · features.py · model.py (household detector) · train.py · explain.py
notebooks/  01_eda.ipynb · 02_modelling_and_explainability.ipynb   (executed, with outputs)
app/        streamlit_app.py · sample_meter_readings.csv
models/     solar_detector.joblib (stacking ensemble + threshold) · metadata.json
reports/    model_comparison.csv · best_params.json · test predictions · SHAP/LIME outputs · figures/
tests/      feature pipeline + saved-model behaviour tests
.github/    ci.yml (tests + Docker build/health check) · deploy-azure.yml (ACR → Azure Web App)
```

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the RECON-SL zip from Kaggle into data/raw/recon_sl.zip, then:
python -m src.data            # ~1 min  → data/processed/household_days.parquet
python -m src.train           # ~50 min → models/, reports/
python -m src.explain         # ~10 min → SHAP/LIME figures
pytest -q
streamlit run app/streamlit_app.py
```

## 7. MLOps

* **Versioned pipeline** — data build, training and explanation are scripts with fixed seeds; results, parameters and
  splits are saved under `reports/` and `models/metadata.json`.
* **CI** (`.github/workflows/ci.yml`) — on every push: install pinned dependencies, run tests (including behaviour
  tests on the saved model), build the Docker image and check the container's health endpoint.
* **CD** (`.github/workflows/deploy-azure.yml`) — after CI passes on `main`: build and push the image to Azure
  Container Registry, deploy to an Azure Web App for Containers, and verify the live health endpoint.
  Needs the `AZURE_CREDENTIALS` secret and `ACR_NAME` / `AZURE_WEBAPP_NAME` variables.
* **Container** — `Dockerfile` serves the Streamlit app on port 8501 (`WEBSITES_PORT=8501` on Azure).

## 8. Limitations
One utility area and period; solar systems that never export (off-grid, full battery) are labelled non-solar by
definition; households with very few days of data are less reliable; the model detects *presence*, not system size.
