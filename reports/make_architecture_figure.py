"""Draw reports/figures/rotation_forest_architecture.png (the Rotation Forest diagram used in the report)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREY, INK = "#5170FF", "#E8A33D", "#E4DCCD", "#1F2933"


def box(ax, x, y, w, h, text, color, fs=9, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=INK, lw=1))


fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.0, 1.85, 1.75, 1.0, "One household-day\n64 features\n(signed log,\nz-score)", GREY, 8.5)
for i, (y, label) in enumerate([(3.5, "tree 1"), (2.0, "tree 2"), (0.3, "tree T")]):
    box(ax, 2.2, y, 2.4, 0.8, "split features into random\nsubsets of K; PCA per subset\n(rows sub-sampled)", "#DCE3FF", 8)
    box(ax, 5.0, y, 1.5, 0.8, f"rotate:\nX · R{['$_1$', '$_2$', '$_T$'][i]}", "#DCE3FF", 9)
    box(ax, 6.9, y, 1.2, 0.8, f"decision\n{label}", BLUE, 9, "white")
    arrow(ax, 1.7, 2.35, 2.2, y + 0.4); arrow(ax, 4.6, y + 0.4, 5.0, y + 0.4); arrow(ax, 6.5, y + 0.4, 6.9, y + 0.4)
    arrow(ax, 8.1, y + 0.4, 8.5, 2.35)
ax.text(5.75, 1.45, "⋮", fontsize=18, ha="center")
box(ax, 8.5, 1.9, 1.4, 0.9, "average\nP(solar | day)", ORANGE, 9)
ax.text(5, 4.45, "Rotation Forest: every tree sees the data on its own PCA-rotated axes", ha="center", fontsize=11,
        color=INK, weight="bold")
ax.text(9.2, 1.55, "household verdict =\nmean over its days\n≥ threshold", ha="center", va="top", fontsize=8, color=INK)
fig.savefig("reports/figures/rotation_forest_architecture.png", dpi=150, bbox_inches="tight")
