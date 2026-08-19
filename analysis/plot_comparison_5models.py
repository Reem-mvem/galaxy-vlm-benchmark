"""Third comparison figure: 5 models x 4 tasks (balanced accuracy).
Zero-shot: Gemini 3.5 Flash, GPT-5.6-sol, Qwen3.5-397B-A17B.  Fine-tuned: CNN, Qwen3.5-9B.
Numbers read from docs/figures/benchmark_all_models.png (zero-shot) + analysis/master_comparison.png
(fine-tuned). Nothing recomputed. New file — does not overwrite the other two plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/comparison_5models.png"

VIOLET, TEAL, L_BLUE, GREEN, ORANGE = "#4a3aa7", "#2a9d8f", "#9ec5f4", "#008300", "#eb6834"
PRIMARY, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

MODELS = [
    ("Gemini 3.5 Flash  (zero-shot)",      VIOLET),
    ("GPT-5.6-sol  (zero-shot)",           TEAL),
    ("Qwen3.5-397B-A17B  (zero-shot)",     L_BLUE),
    ("Fine-tuned CNN",                     GREEN),
    ("Fine-tuned Qwen3.5-9B",              ORANGE),
]
TASKS = ["which_lsb\n(4-class)", "is_lsb", "strong_lens", "rings"]
#          Gemini  GPT  Qwen397B  CNN  Qwen9B
DATA = {
    "which_lsb":   [56, 32, 41, 59, 51],
    "is_lsb":      [91, 78, 84, 96, 94],
    "strong_lens": [73, 52, 49, 75, 70],
    "rings":       [58, 51, 51, 54, 72],
}
keys = ["which_lsb", "is_lsb", "strong_lens", "rings"]

nT, nM = len(keys), len(MODELS)
gw = 0.84; bw = gw / nM
xg = np.arange(nT)

fig, ax = plt.subplots(figsize=(13.5, 7.2))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

for mi, (label, color) in enumerate(MODELS):
    xs = xg - gw / 2 + bw * (mi + 0.5)
    vals = [DATA[k][mi] for k in keys]
    ax.bar(xs, vals, width=bw * 0.9, color=color, zorder=3, label=label)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=PRIMARY)

ax.axhline(50, color=BASELINE, lw=1.0, ls="--", zorder=1)
ax.text(nT - 0.52, 50.7, "binary chance 50%", ha="right", va="bottom", fontsize=9, color=MUTED)

ax.set_ylim(0, 104)
ax.set_ylabel("Balanced accuracy (%)", fontsize=13, color=SECONDARY)
ax.set_xticks(xg); ax.set_xticklabels(TASKS, fontsize=12.5, color=PRIMARY)
ax.set_title("Zero-shot APIs vs fine-tuned CNN vs fine-tuned VLM",
             fontsize=16, fontweight="bold", color=PRIMARY, pad=34)
ax.text(0.5, 1.05,
        "Balanced accuracy, same test set per task  ·  fine-tuned on 100 balanced labels  ·  zero-shot = no training",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=11)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
          frameon=False, fontsize=10.5, columnspacing=1.6, handlelength=1.4)

fig.tight_layout()
fig.savefig(OUT, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
