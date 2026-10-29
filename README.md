# ☀️ Rooftop Solar Detection from Smart-Meter Data — Sri Lanka

**AML Assignment 2 · A new ensemble algorithm (Rotation Forest) on a real, local dataset**

Detect whether a Sri Lankan household has **rooftop solar** using only its **grid import** smart-meter readings,
with a **Rotation Forest** (an ensemble of PCA-rotated decision trees, not covered in the lectures), explained with
SHAP, LIME and partial dependence, served through a web app (Python server + HTML page), and deployed on Azure.

| Held-out test (592 unseen households, 608 household-periods) | Result |
|---|---|
| **Household accuracy** | **98.85%** (7 errors / 608) |
| Household AUC | 0.9985 |
| Precision / recall (solar) | 1.000 / 0.932 |
| Day-level accuracy (90,559 test days) | 98.20% |
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
| **Main algorithm** | **Rotation Forest** (Rodríguez et al., 2006): every tree is trained on its own PCA rotation of random feature subsets; the forest averages the trees' probabilities | `src/rotation_forest.py` |
| Tuning | grid search (subset size K, features per split, leaf size) on household-grouped 3-fold CV, minimising day-level log loss | `src/train.py` |
| Baselines | algorithms taught in lectures: Dummy, Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, HistGB, XGBoost, LightGBM, CatBoost, soft voting, stacking | `src/train.py` |
| Ablation | the same forest with the rotation switched off, to show what the rotation adds | `src/train.py` |
| Household decision | mean of day probabilities; threshold (0.46) chosen on training households only | `src/model.py` |
| Explainability | permutation SHAP on the whole forest, LIME, partial dependence + ICE | `src/explain.py`, `notebooks/02_…` |
| App | Python standard-library server with a JSON API + one HTML page: one-day form with SHAP/LIME, or upload raw readings for a household verdict | `app/server.py`, `app/index.html` |

### Keeping the score honest
* **Grouped splits** — a household's days are entirely in train or test (days of one home are near-duplicates).
* **20% of households held out** and scored once at the end; tuning and the threshold use training households only.
* **Same training data for every model** — at most 40 random days per training household (92,563 days).
* **Leakage removed** — export energy/power, reactive power and power factor are never read; the 6-hour files' `EXPORT`
  flag and meter type are not used. A first run ranked *calendar month* #2 in SHAP; the solar share per month
  (7–53%) reflects which meters reported, not sunlight, so month was **removed** and the model retrained.
* **No single feature is a perfect proxy** — the best single feature has AUC 0.885 (EDA §5).

## 4. Results

Household metrics at threshold 0.5 for every model (5-fold grouped CV on training households, then the held-out test):

| Model | CV acc. | CV AUC | CV day log loss | Test errors / 608 |
|---|---|---|---|---|
| Dummy (majority) | 0.8372 | 0.5000 | 2.176 | 103 |
| Logistic Regression | 0.9785 | 0.9923 | 0.160 | 13 |
| KNN (k=25) | 0.9798 | 0.9972 | 0.154 | 8 |
| Decision Tree | 0.9874 | 0.9966 | 0.136 | 6 |
| Random Forest / Extra Trees | 0.9861 / 0.9857 | 0.9987 / 0.9983 | 0.074 / 0.079 | 7 / 7 |
| HistGB / XGBoost / LightGBM / CatBoost | 0.9891 / 0.9912 / 0.9895 / 0.9903 | 0.9982 / 0.9981 / 0.9981 / 0.9985 | 0.069 / 0.069 / 0.069 / 0.067 | 6 / 6 / 6 / 5 |
| Soft voting / Stacking (lecture ensembles) | 0.9903 / 0.9886 | 0.9986 / 0.9979 | 0.066 / 0.076 | 6 / 6 |
| Rotation Forest **without** rotation (ablation) | 0.9870 | 0.9953 | 0.165 | 7 |
| **Rotation Forest** | 0.9870 | **0.9987** | 0.074 | 7 |

With its own threshold (0.46, chosen on training households) the Rotation Forest scores **98.95%** in CV
(25 errors / 2,377) and **98.85%** on the test set (7 / 608, precision 1.000).

* **The rotation is what makes it work**: switching it off raises day log loss from 0.074 to 0.165 and lowers AUC.
* Rotation Forest **ranks households best** (highest CV AUC, tied with Random Forest). Gradient boosting is 2–5
  households more accurate at a fixed threshold; on 608 test households that is within noise, and is reported as is.
* **Error analysis**: all 7 test errors are missed solar homes (no false alarms): a home with 3 days of data, two
  short post-installation periods, a tiny exporter (93 kWh in 88 days) and homes whose import barely changes despite
  exporting (likely battery/hybrid). Households with > 30 days of data: **99.31%** (4 errors / 582).
* **SHAP / LIME / PDP**: predictions are driven by *how low daytime import gets relative to the rest of the day* —
  the physical solar signature.

## 5. Project layout

```
src/        download.py · data.py (pipeline) · features.py · rotation_forest.py · model.py · train.py · explain.py
            drift.py (PSI monitoring) · quality_gate.py · track.py (MLflow/Azure ML) · mlflow_model.py
notebooks/  01_eda.ipynb · 02_modelling_and_explainability.ipynb   (executed, with outputs)
app/        server.py (JSON API) · index.html (the page) · sample_meter_readings.csv
models/     solar_detector.joblib (Rotation Forest + threshold) · metadata.json · reference_stats.json
reports/    metrics.json · model_comparison.csv · best_params.json · test predictions · explain/ · figures/
dvc.yaml    pipeline stages (download → build → train → explain) · params.yaml · dvc.lock
infra/      azure_setup.sh (creates all Azure resources) · azure.env (resource names/IDs, no secrets)
tests/      features, Rotation Forest, saved-model behaviour, app, drift and quality-gate tests
.github/    ci.yml · deploy.yml · retrain.yml · monitor.yml
MAINTENANCE.md  runbook: retrain, roll back, monitor, costs, clean-up
```

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-mlops.txt
dvc repro            # download → build (~1 min) → train (~1 h) → explain (~10 min); reruns only what changed
pytest -q
python app/server.py      # opens http://127.0.0.1:8000
```

## 7. MLOps

The full lifecycle runs on GitHub Actions + Azure (East Asia), with **no stored passwords or keys** — GitHub logs in
to Azure through OIDC as a managed identity limited to one resource group. Day-to-day operation is described in
[MAINTENANCE.md](MAINTENANCE.md).

| Practice | Implementation |
|---|---|
| Data versioning & lineage | **DVC** pipeline (`dvc.yaml`, `params.yaml`, `dvc.lock`) with an **Azure Blob Storage** remote |
| Experiment tracking | **MLflow on Azure Machine Learning** — one run per training, a child run per model compared, explanations and the packaged model as artifacts |
| Model registry | Azure ML registry `solar-detector` (v1, v2, …) tagged `candidate` / `production` / `archived` with image tag, git SHA and data hash |
| Continuous integration | `ci.yml` — pinned dependencies, 25 tests, Docker build + container health check |
| Continuous delivery | `deploy.yml` — image to **Azure Container Registry** (tagged by commit), roll out to **Azure Container Apps**, live health check, registry update; one-click rollback to any earlier tag |
| Continuous training | `retrain.yml` — `dvc repro`, drift report, **quality gate** (new model must not be worse), candidate registered, result pushed to a review branch |
| Monitoring | app logs one JSON line per prediction (no personal data) to **Log Analytics**; `monitor.yml` checks uptime and summarises usage daily; PSI drift report on retraining |
| Infrastructure as code | `infra/azure_setup.sh` recreates every Azure resource; `az group delete -n rg-solar-detector` removes them |

## 8. Limitations
One utility area and period; solar systems that never export (off-grid, full battery) are labelled non-solar by
definition; households with very few days of data are less reliable; the model detects *presence*, not system size.
