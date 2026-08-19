"""Master comparison — horizontal dot-strip (Cleveland) plot.
One row per task, one dot per model, x = balanced accuracy. Rows sorted by
(best VLM - CNN) so the story reads top-to-bottom: VLMs win -> CNN wins.
All fine-tuned on the SAME 100 balanced labels (VLMs LoRA r=8, matched recipe);
Gemini zero-shot; CNN = lowest of 2 seeds; VLMs single seed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/analysis/master_comparison_dot.png"

VIOLET, GREEN, L_BLUE, BLUE, ORANGE = "#4a3aa7", "#008300", "#9ec5f4", "#2a78d6", "#eb6834"
PRIMARY, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# (label, color, marker)
MODELS = [
    ("Gemini 3.5 Flash (zero-shot)", VIOLET, "D"),
    ("Fine-tuned CNN",               GREEN,  "s"),
    ("Fine-tuned Qwen2-VL-7B",       L_BLUE, "o"),
    ("Fine-tuned Qwen3.5-4B",        BLUE,   "o"),
    ("Fine-tuned Qwen3.5-9B",        ORANGE, "o"),
]
# task rows, sorted by (best VLM - CNN), descending: VLM wins on top
#            name                      Gemini  CNN   7B    4B    9B
ROWS = [
    ("rings",                         [61.3, 54.2, 66.0, 68.6, 71.9]),
    ("which_lsb  (4-class)",          [63.0, 59.3, 59.5, 63.6, 50.8]),
    ("is_lsb",                        [91.2, 95.7, 92.4, 94.3, 94.1]),
    ("strong_lens",                   [73.0, 75.2, 65.4, 64.1, 70.0]),
]

# label placement: 1 = above dot, -1 = below dot (resolve near-ties per row)
LBL = {
    ("rings", 0): 1, ("rings", 1): 1, ("rings", 2): 1, ("rings", 3): 1, ("rings", 4): 1,
    ("which_lsb  (4-class)", 0): 1,  ("which_lsb  (4-class)", 3): -1,   # 63.0 vs 63.6
    ("which_lsb  (4-class)", 1): 1,  ("which_lsb  (4-class)", 2): -1,   # 59.3 vs 59.5
    ("which_lsb  (4-class)", 4): 1,
    ("is_lsb", 1): 1, ("is_lsb", 3): 1, ("is_lsb", 4): -1,              # 94.3 vs 94.1
    ("is_lsb", 0): 1, ("is_lsb", 2): -1,                                # 91.2 vs 92.4
    ("strong_lens", 0): 1, ("strong_lens", 1): 1, ("strong_lens", 4): 1,
    ("strong_lens", 2): 1, ("strong_lens", 3): -1,                      # 64.1 vs 65.4
}

fig, ax = plt.subplots(figsize=(11.2, 5.6))
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

nR = len(ROWS)
ys = np.arange(nR)[::-1]  # first row on top

for (task, vals), y in zip(ROWS, ys):
    # range strip behind the dots
    ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=3,
            solid_capstyle="round", zorder=1)
    # dots
    for mi, ((mlabel, mcolor, mk), v) in enumerate(zip(MODELS, vals)):
        ax.scatter([v], [y], s=110, marker=mk, color=mcolor,
                   edgecolors=SURFACE, linewidths=1.6, zorder=3)
        side = LBL.get((task, mi), 1)
        ax.text(v, y + side * 0.21, f"{v:.0f}",
                ha="center", va="bottom" if side > 0 else "top",
                fontsize=9, fontweight="bold", color=PRIMARY, zorder=4)
    # right margin: best-VLM vs CNN delta (the story column)
    best_vlm = max(vals[2:]); cnn = vals[1]; d = best_vlm - cnn
    ax.text(101.0, y, f"best VLM vs CNN  {d:+.0f}", ha="left", va="center",
            fontsize=9.3, color=PRIMARY if d > 0 else MUTED,
            fontweight="bold" if d > 0 else "normal")

# legend proxies
for mlabel, mcolor, mk in MODELS:
    ax.scatter([], [], s=90, marker=mk, color=mcolor,
               edgecolors=SURFACE, linewidths=1.4, label=mlabel)

# chance line (binary tasks); which_lsb chance is 25, off-scale — noted in row name
ax.axvline(50, color=BASELINE, lw=1.0, ls="--", zorder=0)
ax.text(50, nR - 0.42, " binary chance", ha="left", va="top", fontsize=8.2, color=MUTED)

ax.set_xlim(46, 100)
ax.set_ylim(-0.6, nR - 0.4)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], fontsize=11.5, color=PRIMARY)
ax.set_xlabel("Balanced accuracy (%)", fontsize=11.5, color=SECONDARY)
ax.set_title("Open VLMs rival a fully fine-tuned CNN from 100 balanced labels",
             fontsize=14.5, fontweight="bold", color=PRIMARY, pad=44)
ax.text(0.5, 1.115,
        "Balanced accuracy, same test set per task  ·  fine-tuned models trained on the identical 100 balanced labels (VLMs: LoRA r=8)\n"
        "rows sorted by VLM advantage  ·  which_lsb is 4-class (chance 25%)  ·  CNN = lowest of 2 seeds",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0, color=MUTED)

ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(colors=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
          frameon=False, fontsize=9.6, columnspacing=1.2, handletextpad=0.4)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
