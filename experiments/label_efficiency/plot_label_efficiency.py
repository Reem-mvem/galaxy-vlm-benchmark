"""
Label-efficiency plots in the rings-plot style (balanced accuracy vs #training
images): faint seed dots, mean +/- 1 std line, dashed Gemini baseline, 100% star.
One figure per task, distinct palette each. Reads balacc_classification.csv.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = "/scratch/reem1/rings_runs/analysis"
res = pd.read_csv(f"{OUT}/balacc_classification.csv")

# per task: curve color, Gemini baseline color+value, star color, title
STYLE = {
    "is_lsb":      dict(title="Is Debris (is-lsb) — MaxViT-Tiny",
                        c="#1b9e77", star="#0b5d46", gem="#d1495b", gemini=0.912),
    "which_lsb":   dict(title="Which Debris (which-lsb) — MaxViT-Tiny",
                        c="#7570b3", star="#403c73", gem="#e6a817", gemini=0.630),
    "strong_lens": dict(title="Strong Lenses — SoViT-400m/14",
                        c="#d95f02", star="#8f3f01", gem="#3969ac", gemini=0.73),
}

def draw(ax, task):
    st = STYLE[task]; d = res[res.task == task]   # all points: sweep divisors + 100% (all 6 seeds)
    xs = sorted(d.n_images.unique())
    means = [d[d.n_images == x].bal_acc.mean() for x in xs]
    stds  = [d[d.n_images == x].bal_acc.std()  for x in xs]

    # individual seed points (faint, jittered)
    for x in xs:
        v = d[d.n_images == x].bal_acc.values
        jit = x * np.random.uniform(-0.02, 0.02, len(v))
        ax.scatter(np.full(len(v), x) + jit, v, s=22, color=st["c"], alpha=0.35,
                   edgecolors="none", zorder=2)

    # mean +/- std line through all points, including 100%
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color=st["c"], ecolor=st["c"],
                elinewidth=1.5, capsize=4, markersize=7, markerfacecolor=st["c"],
                markeredgecolor="white", markeredgewidth=1.5, linewidth=2, zorder=4,
                label="Fine-tuned CNN\nmean ± 1 std, dots = individual seeds")

    # Gemini baseline
    ax.axhline(st["gemini"], color=st["gem"], ls="--", lw=2, zorder=3,
               label=f"Gemini zero-shot = {st['gemini']:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of training images (log scale)", fontsize=12)
    ax.set_ylabel("Balanced accuracy", fontsize=12)
    ax.set_title(st["title"], fontsize=13)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticklabels([f"{v:,}" for v in xs], rotation=35, ha="right")
    lo = min(min(means) - max(stds), st["gemini"]) - 0.03
    ax.set_ylim(max(0, lo), 1.005)
    ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

for task in STYLE:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw(ax, task)
    fig.tight_layout()
    fig.savefig(f"{OUT}/labeleff_{task}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(22, 5.6))
for ax, task in zip(axes, STYLE):
    draw(ax, task)
fig.tight_layout()
fig.savefig(f"{OUT}/labeleff_combined.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved:", ", ".join(f"labeleff_{t}.png" for t in STYLE), "+ labeleff_combined.png")
