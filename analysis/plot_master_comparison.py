"""Master comparison — grouped bar chart, balanced accuracy across 4 galaxy tasks x 5 models.
Fine-tuned models (CNN, Qwen2-VL-7B, Qwen3.5-4B, Qwen3.5-9B) trained on the SAME 100
balanced images per task (LoRA r=8, matched per-task recipe); Gemini is zero-shot.
Same test set per task. VLMs = single seed; CNN = lowest of 2 seeds (conservative)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/analysis/master_comparison.png"

# palette (validated dataviz set)
VIOLET, GREEN, L_BLUE, BLUE, ORANGE = "#4a3aa7", "#008300", "#9ec5f4", "#2a78d6", "#eb6834"
PRIMARY, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

TASKS = ["which_lsb\n(4-class)", "is_lsb", "strong_lens", "rings"]
# (label, color, [which_lsb, is_lsb, strong_lens, rings])   balanced accuracy %
SERIES = [
    ("Gemini 3.5 Flash (zero-shot)", VIOLET, [56.0, 91.2, 73.0, 59.0]),
    ("Fine-tuned CNN",               GREEN,  [59.3, 95.7, 75.2, 54.2]),
    ("Fine-tuned Qwen2-VL-7B",       L_BLUE, [59.5, 92.4, 65.4, 66.0]),
    ("Fine-tuned Qwen3.5-4B",        BLUE,   [63.6, 94.3, 64.1, 68.6]),
    ("Fine-tuned Qwen3.5-9B",        ORANGE, [50.8, 94.1, 70.0, 71.9]),
]

nT, nM = len(TASKS), len(SERIES)
group_w = 0.84
bw = group_w / nM
xg = np.arange(nT)

fig, ax = plt.subplots(figsize=(12.2, 6.4))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

for mi, (label, color, vals) in enumerate(SERIES):
    xs = xg - group_w / 2 + bw * (mi + 0.5)
    ax.bar(xs, vals, width=bw * 0.9, color=color, zorder=3, label=label)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.7, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.3, fontweight="bold", color=PRIMARY)

# binary-chance reference (which_lsb is 4-class, chance 25 — noted in axis label/subtitle)
ax.axhline(50, color=BASELINE, lw=1.0, ls="--", zorder=1)
ax.text(nT - 0.52, 50.6, "binary chance 50%", ha="right", va="bottom", fontsize=8, color=MUTED)

ax.set_ylim(0, 104)
ax.set_ylabel("Balanced accuracy (%)", fontsize=12, color=SECONDARY)
ax.set_xticks(xg); ax.set_xticklabels(TASKS, fontsize=11.5, color=PRIMARY)
ax.set_title("Fine-tuned CNN vs VLMs — 100 balanced training labels per task",
             fontsize=14.5, fontweight="bold", color=PRIMARY, pad=34)
ax.text(0.5, 1.05,
        "Balanced accuracy, same test set per task  ·  fine-tuned models trained on the identical 100 balanced labels (VLMs: LoRA r=8)  ·  Gemini zero-shot",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.2, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5,
          frameon=False, fontsize=9.6, columnspacing=1.2, handlelength=1.3)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
