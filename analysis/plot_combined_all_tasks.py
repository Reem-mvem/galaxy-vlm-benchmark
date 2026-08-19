"""Combined figure: models x 5 tasks, balanced accuracy. Fine-tuned CNN = baseline (far left).
Dashed line on each fine-tuned Qwen-4B bar = its zero-shot (non-fine-tuned) level."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/combined_all_tasks.png"

VIOLET, TEAL, L_BLUE, GREEN, ORANGE = "#4a3aa7", "#2a9d8f", "#9ec5f4", "#008300", "#eb6834"
PRIMARY, MUTED, GRID = "#0b0b0b", "#7c7a74", "#e1e0d9"

TASKS = ["which_lsb", "is_lsb", "strong_lens", "rings", "Morphology"]
#   (label, color, is_finetuned, [which_lsb, is_lsb, strong_lens, rings, Morphology])
SERIES = [
    ("Fine-tuned CNN  (baseline)", GREEN,  True,  [59.3, 95.7, 75.2, 54.2, 76.7]),
    ("Gemini 3.5 Flash",           VIOLET, False, [56.0, 91.2, 73.0, 59.0, 75.0]),
    ("GPT-5.6-sol",                TEAL,   False, [32.0, 78.0, 52.0, 51.0, 65.0]),
    ("Qwen3.5-397B-A17B",          L_BLUE, False, [41.0, 84.0, 49.0, 51.0, 67.0]),
    ("Fine-tuned Qwen3.5-4B",      ORANGE, True,  [63.6, 94.3, 64.1, 68.6, 73.7]),
]
ZS_4B = [35.4, 86.6, 50.5, 51.1, 61.9]   # Qwen-4B zero-shot (before fine-tuning)

nT, nM = len(TASKS), len(SERIES)
group_w = 0.86
bw = group_w / nM
xg = np.arange(nT)

fig, ax = plt.subplots(figsize=(22, 11))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

for mi, (label, color, ft, vals) in enumerate(SERIES):
    xs = xg - group_w / 2 + bw * (mi + 0.5)
    kw = dict(edgecolor=PRIMARY, linewidth=2.2, hatch="//") if ft else dict(linewidth=0)
    for xi, v in zip(xs, vals):
        if v is None:
            continue
        ax.bar(xi, v, width=bw * 0.9, color=color, zorder=3, **kw)
        ax.text(xi, v + 0.9, f"{v:.0f}", ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=PRIMARY, zorder=4)

# dashed line on the fine-tuned Qwen-4B bar (rightmost) = its zero-shot level
xs_q4 = xg - group_w / 2 + bw * (nM - 1 + 0.5)
for xi, zs in zip(xs_q4, ZS_4B):
    ax.hlines(zs, xi - bw * 0.47, xi + bw * 0.47, color=PRIMARY, lw=3.2,
              linestyles=(0, (4, 3)), zorder=5)

# divider separating the CNN baseline (leftmost bar) from the rest, per group
for g in xg:
    ax.axvline(g - group_w / 2 + bw, ymin=0.0, ymax=0.9, color=GRID, lw=1.6, ls=(0, (3, 3)), zorder=1)

ax.set_ylim(0, 106)
ax.set_ylabel("Balanced accuracy (%)", fontsize=30, color=PRIMARY, labelpad=16)
ax.set_xlabel("Classification task", fontsize=30, color=PRIMARY, labelpad=18)
ax.set_xticks(xg); ax.set_xticklabels(TASKS, fontsize=26, color=PRIMARY)
ax.set_title("Zero-shot vs fine-tuned models across galaxy classification tasks",
             fontsize=34, fontweight="bold", color=PRIMARY, pad=30)

ax.grid(axis="y", color=GRID, lw=1.1, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(axis="y", colors=MUTED, labelsize=22)
ax.tick_params(axis="x", length=0)

handles = [
    Patch(facecolor=GREEN, edgecolor=PRIMARY, hatch="//", label="Fine-tuned CNN  (baseline) (100 labels)"),
    Patch(facecolor=VIOLET, label="Gemini 3.5 Flash"),
    Patch(facecolor=TEAL,   label="GPT-5.6-sol"),
    Patch(facecolor=L_BLUE, label="Qwen3.5-397B-A17B"),
    Patch(facecolor=ORANGE, edgecolor=PRIMARY, hatch="//", label="Fine-tuned Qwen3.5-4B (100 labels)"),
    Line2D([0], [0], color=PRIMARY, lw=3.2, linestyle=(0, (4, 3)),
           label="Qwen-4B zero-shot (before fine-tuning)"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
          frameon=False, fontsize=20, columnspacing=1.8, handlelength=1.8)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
