"""Draws the stacking-ensemble architecture as reports/figures/ensemble_architecture.png.

All numbers are read from the saved training outputs, so the figure stays correct after retraining:
    python reports/make_architecture_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
meta = json.load(open(ROOT / "models/metadata.json"))
params = json.load(open(ROOT / "reports/best_params.json"))
weights = pd.read_csv(ROOT / "reports/explain/stacking_weights.csv", index_col=0).iloc[:, 0]
comp = pd.read_csv(ROOT / "reports/model_comparison.csv").set_index("model")
test = meta["test_households"]

INK, MUTED, LINE = "#1f2933", "#52606d", "#9aa5b1"
BAG, BOOST, META, IO = "#e3f4ea", "#e6efff", "#fff1db", "#f2f4f7"
BAG_E, BOOST_E, META_E, IO_E = "#2f9e5b", "#3b6fd8", "#d9822b", "#7b8794"

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, body, face, edge, title_size=12.5, body_size=10, body_gap=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                facecolor=face, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - body_gap, body, ha="center", va="top", fontsize=body_size, color=MUTED, linespacing=1.45)


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=LINE, linewidth=1.6, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, ha="center", va="bottom", fontsize=8.5, color=MUTED)


ax.text(80, 87, "Stacking ensemble — rooftop solar detection from smart-meter data (Sri Lanka)",
        ha="center", va="top", fontsize=17, weight="bold", color=INK)
ax.text(80, 82.6, "RECON-SL (LIRNEasia) · 452,811 household-days · 2,925 households · grid import only (export readings never used as inputs)",
        ha="center", va="top", fontsize=10.5, color=MUTED)

# Input and features
box(2, 50, 24, 20, "Input: one day", "24 hourly values of\ngrid import (kWh)\nfrom 15-minute\nsmart-meter readings", IO, IO_E)
box(2, 22, 24, 22, "64 features", "hourly kWh + hourly share\nmidday / night / evening\nratios, daytime minimum,\n\"solar bowl\" depth,\npeak & trough hour", IO, IO_E)
arrow(14, 50, 14, 44)

# Level-0 base learners
base = [
    ("extra_trees", "Extra Trees", "Bagging · random splits", f"{params['Extra Trees']['n_estimators']} trees, max_features {params['Extra Trees']['max_features']:.2f}", BAG, BAG_E),
    ("lightgbm", "LightGBM", "Boosting · leaf-wise trees", f"{params['LightGBM']['n_estimators']} trees, {params['LightGBM']['num_leaves']} leaves, lr {params['LightGBM']['learning_rate']:.3f}", BOOST, BOOST_E),
    ("xgboost", "XGBoost", "Boosting · level-wise trees", f"{params['XGBoost']['n_estimators']} trees, depth {params['XGBoost']['max_depth']}, lr {params['XGBoost']['learning_rate']:.3f}", BOOST, BOOST_E),
    ("catboost", "CatBoost", "Boosting · symmetric trees", f"{params['CatBoost']['iterations']} trees, depth {params['CatBoost']['depth']}, lr {params['CatBoost']['learning_rate']:.3f}", BOOST, BOOST_E),
]
ax.text(59, 76.5, "Level 0 — 4 tuned base learners", ha="center", fontsize=12, weight="bold", color=INK)
ys = [60.5, 44.5, 28.5, 12.5]
targets = [49, 44, 39, 34]  # where each arrow meets the meta-learner
for (key, name, kind, hp, face, edge), y, ty in zip(base, ys, targets):
    box(42, y, 34, 14.5, name, f"{kind}\n{hp}", face, edge, title_size=12, body_size=9.2, body_gap=6.2)
    ax.text(59, y + 2.6, f"stacking weight w = {weights[key]:.2f}", ha="center", va="top", fontsize=9.5, weight="bold", color=edge)
    arrow(26.4, 33, 41.6, y + 7.25)
    arrow(76.4, y + 7.25, 92.6, ty)

# Level-1 meta learner
ax.text(105, 76.5, "Level 1 — meta-learner", ha="center", fontsize=12, weight="bold", color=INK)
box(93, 31, 24, 22, "Logistic Regression", "combines the 4 P(solar)\nvalues; learned weight w\nshown in each model box", META, META_E, title_size=12, body_size=9.2)
ax.text(105, 28.2, "trained on out-of-fold predictions\n(5-fold CV grouped by household)", ha="center", va="top", fontsize=8.5, color=MUTED)

# Day -> household decision
box(123, 52, 34, 12, "Day prediction", "P(solar installed | this day)", IO, IO_E)
arrow(117.4, 46, 122.6, 57)
box(123, 24, 34, 20, "Household decision", f"average P(solar) over all days\n≥ {meta['threshold']:.2f} → Solar installed\n<  {meta['threshold']:.2f} → No solar\n(threshold chosen on training data)", META, META_E)
arrow(140, 52, 140, 44.4)

# Result banner
ax.add_patch(FancyBboxPatch((123, 5), 34, 13, boxstyle="round,pad=0.4,rounding_size=1.6",
                            facecolor="#123b6e", edgecolor="#123b6e"))
ax.text(140, 15.8, "Held-out test (592 unseen homes)", ha="center", va="top", fontsize=10, color="#dbe7f7")
ax.text(140, 11.4, f"{test['hh_accuracy']:.2%} accuracy", ha="center", va="top", fontsize=16, weight="bold", color="white")
ax.text(140, 6.9, f"{test['hh_errors']} errors / {test['hh_n']} · AUC {test['hh_auc']:.4f}", ha="center", va="top", fontsize=10, color="#dbe7f7")

# Models compared but not in the ensemble
others = ["Dummy (majority)", "Logistic Regression", "KNN (k=25)", "Decision Tree", "Random Forest", "HistGradientBoosting"]
txt = "   ·   ".join(f"{m.replace(' (majority)', '')} {int(comp.loc[m, 'test_hh_errors'])}" for m in others)
ax.text(2, 8.6, "Also compared, not in the ensemble (test errors / 608):", fontsize=9.5, weight="bold", color=INK)
ax.text(2, 4.9, txt, fontsize=9.2, color=MUTED)
ax.text(2, 1.2, f"Every single tuned base model: {int(comp.loc['LightGBM (tuned)', 'test_hh_errors'])} errors  →  stacking ensemble: {test['hh_errors']} errors",
        fontsize=9.2, color=MUTED)

out = ROOT / "reports/figures/ensemble_architecture.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
