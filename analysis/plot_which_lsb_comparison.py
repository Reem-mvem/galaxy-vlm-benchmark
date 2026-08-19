"""which_lsb model comparison — balanced accuracy, all models on the same
218-image test set. Tells the fine-tuning story: Qwen2-VL-7B zero-shot -> fine-tuned,
against zero-shot APIs (OpenAI, Gemini) and the specialist CNN target.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/results/which_lsb_model_comparison.png"

# (label, balanced_accuracy_%, color)  — ascending
BLUE_L, ORANGE, BLUE, VIOLET, GREEN = "#9ec5f4", "#eb6834", "#2a78d6", "#4a3aa7", "#008300"
bars = [
    ("Qwen2-VL-7B\n(zero-shot)",        27.0, BLUE_L),
    ("OpenAI\n(zero-shot)",             31.6, ORANGE),
    ("Qwen2-VL-7B\n(fine-tuned, 508)",  48.7, BLUE),
    ("Gemini\n(zero-shot)",             55.8, VIOLET),
    ("CNN\n(fine-tuned)",               64.0, GREEN),
]
labels = [b[0] for b in bars]
vals   = [b[1] for b in bars]
colors = [b[2] for b in bars]

PRIMARY, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CHANCE, CNN = 25.0, 64.0

fig, ax = plt.subplots(figsize=(8.6, 5.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(bars))
ax.bar(x, vals, width=0.64, color=colors, zorder=3)

# value labels
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.1, f"{v:.0f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=PRIMARY)

# reference lines: random chance (4-class) and CNN target
ax.axhline(CHANCE, ls=(0, (4, 3)), lw=1.2, color=MUTED, zorder=1)
ax.text(len(bars) - 0.5, CHANCE + 0.6, "random chance (25%)", ha="right", va="bottom",
        fontsize=9, color=MUTED, style="italic")
ax.axhline(CNN, ls=(0, (4, 3)), lw=1.2, color=GREEN, alpha=0.55, zorder=1)
ax.text(-0.45, CNN + 0.6, "CNN specialist target", ha="left", va="bottom",
        fontsize=9, color=GREEN)

# fine-tuning jump arrow: zero-shot (bar 0) -> fine-tuned (bar 2)
arr = FancyArrowPatch((0, 27.0 + 5.5), (2, 48.7 + 3.0),
                      connectionstyle="arc3,rad=-0.32",
                      arrowstyle="-|>", mutation_scale=16,
                      lw=2, color=BLUE, zorder=5)
ax.add_patch(arr)
ax.text(1.0, 52.5, "+21.7 pts\nfine-tuning", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=BLUE)

ax.set_ylim(0, 72)
ax.set_ylabel("Balanced accuracy (%)", fontsize=11.5, color=SECONDARY)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10, color=PRIMARY)
ax.set_title("which_lsb (4-class LSB morphology): fine-tuning an open 7B VLM",
             fontsize=13.5, fontweight="bold", color=PRIMARY, pad=32)
ax.text(0.5, 1.045, "Balanced accuracy on the same 218-image test set  ·  Qwen2-VL-7B fine-tuned on 508 labeled images",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
