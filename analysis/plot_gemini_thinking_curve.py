"""Accuracy vs thinking tokens — Gemini 3.5 Flash on streams/shells (3-class), 3 seeds.
Bold points = mean +/- std across seeds; dim dots = individual seed runs. Same 30 galaxies."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/gemini_thinking_curve.png"

VIOLET, LIGHT = "#4a3aa7", "#c4bce6"
PRIMARY, MUTED, GRID = "#0b0b0b", "#7c7a74", "#e1e0d9"

# per level: (think_tok, balanced_acc) for each of the 3 seeds
DATA = {
    "off":     [(0, 53.3), (0, 50.0), (0, 46.7)],
    "minimal": [(0, 53.3), (0, 50.0), (0, 50.0)],
    "low":     [(275.5, 53.3), (260.6, 60.0), (271.7, 56.7)],
    "medium":  [(777.4, 63.3), (720.4, 60.0), (861.7, 63.3)],
    "high":    [(1146.2, 60.0), (1020.4, 56.7), (1121.9, 60.0)],
}
order = ["off", "minimal", "low", "medium", "high"]
# per-point label placement (dx, dy) in points, to avoid the two x=0 labels colliding
LAB = {"off": (10, -34, "left"), "minimal": (10, 16, "left"),
       "low": (0, 18, "center"), "medium": (0, 18, "center"), "high": (0, 18, "center")}

fig, ax = plt.subplots(figsize=(13.5, 8.5))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

ax.axhline(33.3, color="#c3c2b7", lw=1.4, ls="--", zorder=1)
ax.text(1190, 34.0, "chance 33%", ha="right", va="bottom", fontsize=13, color=MUTED)

mx, my, myerr = [], [], []
for lvl in order:
    xs = [p[0] for p in DATA[lvl]]; ys = [p[1] for p in DATA[lvl]]
    ax.scatter(xs, ys, s=90, color=LIGHT, alpha=0.55, edgecolor="none", zorder=2)   # dim seed dots
    mx.append(np.mean(xs)); my.append(np.mean(ys)); myerr.append(np.std(ys, ddof=1))

ax.plot(mx, my, "-", color=VIOLET, lw=2.6, zorder=3)
ax.errorbar(mx, my, yerr=myerr, fmt="o", color=VIOLET, ms=16, mec=PRIMARY, mew=1.7,
            ecolor=VIOLET, elinewidth=2, capsize=7, zorder=4)

for lvl, x, y, err in zip(order, mx, my, myerr):
    dx, dy, ha = LAB[lvl]
    ax.annotate(f"{lvl}\n{y:.1f}±{err:.1f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=14, fontweight="bold", color=PRIMARY)

ax.set_xlabel("Mean thinking tokens used", fontsize=21, color=PRIMARY, labelpad=12)
ax.set_ylabel("Balanced accuracy (%)", fontsize=21, color=PRIMARY, labelpad=12)
ax.set_title("Does more thinking help?  Gemini 3.5 Flash, streams / shells / neither",
             fontsize=22, fontweight="bold", color=PRIMARY, pad=20)

ax.set_ylim(30, 72); ax.set_xlim(-55, 1230)
ax.grid(color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=15)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
