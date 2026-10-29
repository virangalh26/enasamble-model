"""Write notebooks/02_modelling_and_explainability.ipynb from the outputs of src.train and src.explain.

Usage (from the repo root):
    python notebooks/make_02.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelling_and_explainability.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 02 — Rotation Forest: training, evaluation and explainability

Training is run by `python -m src.train` and explanations by `python -m src.explain` (`dvc repro` runs both).
This notebook loads their saved outputs, so it runs in seconds and every number here is reproducible.

**Evaluation protocol**
* Grouped splits: all days of a household are on the same side of every split (days of one home are near-duplicates).
* 20% of households are held out as the test set and scored once, at the end.
* The grid search and the decision threshold use training households only.
* Every model trains on the same sample of at most 40 days per training household; the test set keeps every day."""),
    code("""import json
import pandas as pd
from IPython.display import Image, display
pd.set_option("display.precision", 4)
R = "../reports"
comp = pd.read_csv(f"{R}/model_comparison.csv")
meta = json.load(open("../models/metadata.json"))
best = json.load(open(f"{R}/best_params.json"))"""),
    md("""## 1. The algorithm: Rotation Forest

**Rotation Forest** (Rodríguez, Kuncheva & Alonso, *IEEE TPAMI* 2006) is a tree ensemble that was not covered in
the lectures. For every tree it

1. splits the 64 features at random into small disjoint subsets (K features each),
2. fits PCA on each subset, using a random sub-sample of rows, and keeps *all* components,
3. places these rotations in one block matrix R and trains a decision tree on X·R,

and the forest averages the trees' probabilities.

**How it differs from the standard models.** A decision tree, random forest or gradient-boosting model only splits
on the original axes ("h12_share < 0.02"). The solar signature is a *relation* between features (midday import low
compared with night and evening import), so axis-aligned trees approximate it with a staircase of splits. Each
rotated tree splits on linear combinations of features instead, and because every tree gets a different rotation,
the trees stay accurate *and* diverse, which is what makes an ensemble work. Unlike bagging/random forest, diversity
comes from the rotation, not from throwing data away: keeping all principal components loses no information.

**Why it was chosen.** Tabular, engineered, correlated features (hourly kWh and hourly shares move together), a
non-linear but smooth boundary, and the need for a model that is explainable with SHAP/LIME/PDP. Deep learning is
not allowed, and boosting/stacking were covered in the lectures. Implementation: `src/rotation_forest.py`
(scikit-learn compatible, unit-tested in `tests/test_rotation_forest.py`)."""),
    md("## 2. Hyperparameter search (grid, grouped 3-fold CV on 50% of training households, log loss)"),
    code("""grid = pd.read_csv(f"{R}/rotation_forest_grid_search.csv")
print("chosen:", best["rotation_forest"])
grid"""),
    md("""Log loss on days is used as the objective because it is smooth; household accuracy changes by whole
households and is too coarse to rank settings. The number of trees is not tuned: more trees never hurt a forest,
they only cost time. The curve below shows where adding trees stops helping."""),
    code("""display(Image(f"{R}/figures/trees_curve.png"))
pd.read_csv(f"{R}/trees_curve.csv").iloc[[0, 4, 9, 24, 49, -1]]"""),
    md("## 3. Rotation Forest against the lecture algorithms"),
    code("""cols = ["model", "stage", "cv_hh_accuracy", "cv_hh_auc", "cv_day_logloss", "test_hh_accuracy", "test_hh_auc",
        "test_hh_errors", "test_hh_f1", "seconds"]
comp[cols].sort_values("cv_day_logloss")"""),
    code("""display(Image(f"{R}/figures/model_comparison.png"))"""),
    md("""* **Rotation vs. no rotation (ablation).** The same forest with the rotation switched off is clearly worse,
  so the gain comes from the rotation itself, not just from averaging many trees.
* Grouped CV on 2,300+ training households is the more reliable comparison; the test set has only 608
  household-periods, so a difference of one or two errors there is within noise.
* The household decision averages the day probabilities and compares the mean with a threshold chosen on the
  training households (out-of-fold)."""),
    md("## 4. Final result on the held-out test households"),
    code("""print(meta["model"], "| threshold", meta["threshold"])
pd.DataFrame({"CV (train households)": meta["cv_train_households"], "Test (held out)": meta["test_households"]})"""),
    md("## 5. Error analysis"),
    code("""hh = pd.read_csv(f"{R}/test_household_predictions.csv")
errors = hh[hh.pred != hh.solar_installed].sort_values("p_solar")
errors"""),
    code("""hh["days"] = pd.cut(hh.n_days, [0, 7, 30, 10_000], labels=["1-7", "8-30", ">30"])
hh.groupby("days", observed=True).apply(lambda d: pd.Series({"households": len(d),
    "accuracy": (d.pred == d.solar_installed).mean()}))"""),
    md("""The model and threshold were frozen before this analysis. Errors concentrate in households with few days of
data, very small exporters and homes whose import barely changes (likely battery/hybrid systems): one day can be
ambiguous (cloudy weather, an empty house), and averaging over weeks removes most of that noise."""),
    md("""## 6. Explainability — SHAP (permutation, whole forest)

The trees split on rotated axes, so tree-SHAP would explain mixtures of features. Permutation SHAP explains the
whole forest in terms of the original, readable features; values are changes in P(solar)."""),
    code("""for f in ["shap_beeswarm", "shap_bar", "shap_dependence_daytime_min_share", "shap_dependence_midday_to_evening"]:
    display(Image(f"{R}/explain/{f}.png"))
pd.read_csv(f"{R}/explain/shap_importance.csv", index_col=0).head(10)"""),
    md("""The leading features describe **how low daytime import gets relative to the rest of the day**: on a sunny
day the panels cover the home's load, so the meter imports almost nothing between 09:00 and 15:00 and imports
normally in the evening. This is the physical mechanism of rooftop solar, so the model's behaviour matches domain
knowledge; it does not rely on calendar or meter artefacts (calendar month was removed for that reason, see README)."""),
    md("## 7. Partial dependence (PDP + ICE)"),
    code("""display(Image(f"{R}/explain/pdp.png"))"""),
    md("## 8. Local explanations — SHAP waterfalls and LIME"),
    code("""for case in ["solar_day", "non_solar_day", "uncertain_day"]:
    display(Image(f"{R}/explain/shap_waterfall_{case}.png"))
    display(Image(f"{R}/explain/lime_{case}.png"))
pd.read_csv(f"{R}/explain/lime_explanations.csv")"""),
    md("""SHAP and LIME agree on the confident cases. The most uncertain day (P = 0.50) comes from a home *without*
solar whose import was near zero for several daytime hours (nobody home, or a supply interruption), which looks
exactly like a sunny solar day. A single day cannot tell these apart, which is why the household verdict averages
many days."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"}})
nbf.write(nb, "notebooks/02_modelling_and_explainability.ipynb")
