# 🏠 Aswesuma-style Poverty Targeting with NGBoost — Sri Lanka

**AML Assignment 2 · A new algorithm (not taught in lectures) on a real, local dataset**

Predict whether a Sri Lankan household is **below the official poverty line** from what a welfare officer can
observe on a home visit (house, facilities, belongings, land, household members), using **NGBoost** (Natural
Gradient Boosting). Each household gets a probability *and* an uncertainty range, explained with SHAP, LIME and
partial dependence, and served through a web app (Python server + HTML page) packaged for Azure.

> **Status:** the code is complete and tested on synthetic HIES-shaped files. Results below are filled in once the
> HIES 2019 microdata (free Lanka Datta registration) is in `data/raw/hies2019/` and `dvc repro` has run.

---

## 1. The problem

In 2023 Sri Lanka replaced Samurdhi with **Aswesuma**, a welfare benefit targeted with a household questionnaire
scored by a formula. Its first lists were widely criticised: poor families left out (*exclusion error*) and better-off
families let in (*inclusion error*). Asking about income does not work well because income is hard to verify, so
programmes like this use a **proxy means test (PMT)**: predict a household's consumption from things that are easy to
observe and hard to hide, such as wall and floor material, a fridge or a motorbike, land, and the number of children.

The task: **from observable household characteristics, predict whether monthly consumption per person is below the
official poverty line**, with an honest measure of how sure the model is.

## 2. Data

**Household Income and Expenditure Survey 2019 (HIES 2019)**, Department of Census and Statistics (DCS),
public-use 25% sample, [Lanka Datta catalog 476](https://nada.statistics.gov.lk/index.php/catalog/476).
The microdata needs a free account and is **not redistributed** (`data/raw/` is git-ignored).

* About **4,982 households** across all 25 districts and the urban, rural and estate sectors.
* 29 files are joined on the household key (`psu`, `snumber`, `hhno`); see `src/hies.py`.
* **Target:** monthly consumption expenditure (food + non-food, purchased and in-kind) per person, and
  `poor` = below **Rs 6,966 per person per month**, the DCS official (updated) poverty line for 2019
  ([Poverty indicators 2019](https://www.statistics.gov.lk/Resource/en/Poverty/PovertyIndicators-2019.pdf)).
  The label is checked against the published DCS poverty rates by district (`reports/label_check_vs_dcs.csv`).
* **Features (~100 raw, ~150 after one-hot):** household members (size, children, elderly, dependency ratio, head's
  sex/age/education, employment), housing (walls, floor, roof, rooms, toilet, water, cooking fuel, lighting),
  26 durable goods, land and livestock, distance to services, district and sector.
* **Never features:** income and expenditure (they define the target), and ethnicity and religion (kept only to
  audit fairness).

## 3. Method

| Step | What is done | Where |
|---|---|---|
| Data | join 29 HIES files, consumption aggregate, poverty label, check against DCS district rates | `src/hies.py` |
| Features | household-level proxies; nominal codes one-hot encoded | `src/features.py` |
| **Model** | **NGBoost** with a Normal distribution on log consumption per person: P(poor) = Φ((log z − μ)/σ) | `src/model.py` |
| Tuning | grid over tree depth, learning rate and mini-batch fraction, with early stopping, on PSU-grouped CV | `src/train.py` |
| Baselines | Linear Regression PMT (standard practice), Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, LightGBM, and NGBoost with the ordinary gradient (ablation) | `src/train.py` |
| Evaluation | PR-AUC, ROC-AUC, F1, Brier; exclusion and inclusion error at a fixed budget; RMSE, NLL and 80% interval coverage of the predicted distribution; fairness by sector, head's sex, ethnicity and district | `src/train.py` |
| Explainability | SHAP (TreeExplainer) on μ and on σ, LIME, PDP + ICE, permutation importance | `src/explain.py` |
| App | officer's visit form → chance of being poor, spending range vs the poverty line, SHAP or LIME reasons | `app/` |

### Why NGBoost
* Not covered in lectures, and not deep learning. NGBoost (Duan et al., ICML 2020) is gradient boosting that learns
  **all parameters of a probability distribution** (here μ and σ of log consumption), using the **natural gradient**,
  which accounts for the geometry of the distribution space and makes learning μ and σ together stable.
* **How it differs:** logistic regression and a PMT regression give one number and a single, shared error band;
  decision trees, Random Forest and XGBoost give one point estimate; k-NN averages neighbours. NGBoost gives each
  household its own spread, so the app can say "Rs 6,200 (80% range Rs 4,900–7,800)" and flag uncertain cases for a
  closer check.
* It predicts consumption, the quantity the poverty line is defined on, instead of only a yes/no label, so the
  same model also serves different poverty lines or budgets.

### Keeping the score honest
* **Grouped splits:** households of one survey cluster (PSU) share prices and infrastructure, so a PSU is entirely
  in training or test. 20% of PSUs are held out and scored once at the end.
* Hyperparameters, the number of boosting rounds and the decision threshold are chosen on training households only.

## 4. Results

*Filled in after training on the real data: `reports/model_comparison.csv`, `reports/metrics.json`,
`reports/fairness.csv`, `reports/explain/`.*

## 5. Project layout

```
src/        hies.py (data + label) · features.py · model.py · train.py · explain.py
            drift.py (PSI monitoring) · quality_gate.py · track.py (MLflow/Azure ML) · mlflow_model.py
app/        server.py (JSON API) · index.html (the page) · code_labels.json (names of HIES answer codes)
models/     poverty_targeter.joblib · metadata.json · form_defaults.json · reference_stats.json
reports/    metrics.json · model_comparison.csv · fairness.csv · label_check_vs_dcs.csv · explain/
dvc.yaml    pipeline stages (build → train → explain) · params.yaml · dvc.lock
infra/      azure_setup.sh (creates all Azure resources) · azure.env (resource names/IDs, no secrets)
tests/      pipeline, model, app, drift and quality-gate tests on synthetic HIES-shaped data
.github/    ci.yml · deploy.yml · retrain.yml · monitor.yml
```

## 6. Reproduce

1. Register at [Lanka Datta](https://nada.statistics.gov.lk), download the HIES 2019 files (catalog 476) and put them
   in `data/raw/hies2019/` (CSV, SPSS or Stata all work).
2. Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-mlops.txt
dvc repro            # build → train → explain; reruns only what changed
pytest -q
python app/server.py      # opens http://127.0.0.1:8000
```

## 7. MLOps

The lifecycle runs on GitHub Actions + Azure (East Asia) with **no stored passwords or keys**: GitHub logs in to
Azure through OIDC as a managed identity limited to one resource group. The Azure resource names keep their original
`solar` prefix from the first version of this project. Day-to-day operation is described in
[MAINTENANCE.md](MAINTENANCE.md).

| Practice | Implementation |
|---|---|
| Data versioning & lineage | **DVC** pipeline (`dvc.yaml`, `params.yaml`, `dvc.lock`) with a private **Azure Blob Storage** remote for the microdata |
| Experiment tracking | **MLflow on Azure Machine Learning**: one run per training, a child run per model compared, explanations and the packaged model as artifacts |
| Model registry | Azure ML registry `poverty-targeter` tagged `candidate` / `production` / `archived` with image tag, git SHA and data hash |
| Continuous integration | `ci.yml`: pinned dependencies, tests on synthetic data, Docker build + container health check |
| Continuous delivery | `deploy.yml`: image to **Azure Container Registry**, roll out to **Azure Container Apps**, live health check; one-click rollback |
| Continuous training | `retrain.yml`: `dvc repro`, drift report, **quality gate** (new model must not be worse), candidate registered, result pushed to a review branch |
| Monitoring | the app logs one JSON line per prediction (score and decision only, never a household's answers) to **Log Analytics**; `monitor.yml` checks uptime daily |

## 8. Limitations and ethics (summary)
The public file is a 25% sample of one year (2019), before the 2022 economic crisis roughly doubled poverty, so the
model would need re-training on a newer survey before real use. Consumption is self-reported over a short period.
A model like this should support an officer's decision, never replace it: wrongly excluding a poor family causes
real harm, so flagged and borderline households need verification and an appeal route.
