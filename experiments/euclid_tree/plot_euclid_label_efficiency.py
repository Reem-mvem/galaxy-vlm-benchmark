"""Euclid label-efficiency plot (path accuracy vs #training images), rings style."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

HERE = Path("/home/reem1/scratch/rings_runs/analysis/euclid")
res = pd.read_csv(HERE / "euclid_path_accuracy.csv")

# which training-set sizes to show (drop redundant mid/high points; edit freely)
KEEP = {99, 499, 1998, 9994, 42474, 169899}
res = res[res.n_images.isin(KEEP)]

C = "#c1272d"        # crimson curve
GEM = "#2a9d8f"      # teal Gemini baseline
GEMINI = 0.750       # Gemini structured zero-shot, path accuracy (all galaxies)

xs = sorted(res.n_images.unique())
means = [res[res.n_images == x].path_acc.mean() for x in xs]
stds  = [res[res.n_images == x].path_acc.std()  for x in xs]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for x in xs:
    v = res[res.n_images == x].path_acc.values
    jit = x * np.random.uniform(-0.02, 0.02, len(v))
    ax.scatter(np.full(len(v), x) + jit, v, s=22, color=C, alpha=0.35, edgecolors="none", zorder=2)

ax.errorbar(xs, means, yerr=stds, fmt="o-", color=C, ecolor=C, elinewidth=1.5, capsize=4,
            markersize=7, markerfacecolor=C, markeredgecolor="white", markeredgewidth=1.5,
            linewidth=2, zorder=4, label="Fine-tuned CNN (MaxViT-Base)\nmean ± 1 std, dots = individual seeds")

ax.axhline(GEMINI, color=GEM, ls="--", lw=2, zorder=3, label=f"Gemini zero-shot = {GEMINI:.3f}")

ax.set_xscale("log")
ax.set_xlabel("Number of training images (log scale)", fontsize=12)
ax.set_ylabel("Path accuracy (13-question tree)", fontsize=12)
ax.set_title("GZ Euclid — MaxViT-Base", fontsize=13)
ax.set_xticks(xs)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.minorticks_off()
NICE = {99: "100", 499: "500", 1998: "2,000", 9994: "10,000", 42474: "42,000", 169899: "170,000"}
ax.set_xticklabels([NICE.get(v, f"{v:,}") for v in xs], rotation=35, ha="right")
ax.set_ylim(min(GEMINI, min(means)) - 0.03, 1.005)
ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.35)
ax.legend(loc="lower right", fontsize=9, frameon=True)

fig.tight_layout()
fig.savefig(HERE / "labeleff_gz_euclid.png", dpi=150, bbox_inches="tight")
print("saved:", HERE / "labeleff_gz_euclid.png")
