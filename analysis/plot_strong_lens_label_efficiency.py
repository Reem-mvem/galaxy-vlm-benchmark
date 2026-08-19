"""strong_lens single fact: at a MATCHED ~800-label budget, the fine-tuned open
VLM beats the specialist CNN. The CNN only reaches 73 with its full, larger set.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/strong_lens/results/strong_lens_label_efficiency.png"

BLUE, GREEN, MUTED = "#2a78d6", "#008300", "#898781"
PRIMARY, SECONDARY = "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

bars = [
    ("Qwen2-VL-7B\nfine-tuned\n(788 images)", 69.3, BLUE),
    ("CNN\n(839 images)",                     66.0, GREEN),
]
labels = [b[0] for b in bars]; vals = [b[1] for b in bars]; colors = [b[2] for b in bars]
CHANCE, CNN_FULL = 50.0, 73.0

fig, ax = plt.subplots(figsize=(7.0, 5.8))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = [0, 1]
ax.bar(x, vals, width=0.52, color=colors, zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.0, f"{v:.0f}", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=PRIMARY)

# CNN with full data — reference line above (the CNN needs MORE labels to get here)
ax.axhline(CNN_FULL, ls=(0, (4, 3)), lw=1.4, color=GREEN, alpha=0.6, zorder=1)
ax.text(1.46, CNN_FULL + 0.5, "CNN with full training set (73)",
        ha="right", va="bottom", fontsize=9.5, color=GREEN)



# "same budget" bracket + winner callout (both in clear space above the bars)
ax.annotate("", xy=(0, 84.0), xytext=(1, 84.0),
            arrowprops=dict(arrowstyle="-", color=SECONDARY, lw=1.2))
for xi in x:
    ax.plot([xi, xi], [84.0, 82.7], color=SECONDARY, lw=1.2)
ax.text(0.5, 84.7, "same ~800 labeled images", ha="center", va="bottom",
        fontsize=10.5, color=SECONDARY)
ax.text(0.5, 77.3, "VLM wins at equal budget  (+3 pts)", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=BLUE)

ax.set_ylim(0, 90)
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Balanced accuracy (%)", fontsize=11.5, color=SECONDARY)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, color=PRIMARY)
ax.set_title("strong_lens: at the same label budget,\nthe fine-tuned open VLM beats the CNN",
             fontsize=13.5, fontweight="bold", color=PRIMARY, pad=30)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
