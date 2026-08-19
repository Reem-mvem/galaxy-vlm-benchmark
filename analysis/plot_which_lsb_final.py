"""which_lsb model comparison (final) — balanced accuracy, same 218-image test set.
Zero-shot APIs vs fine-tuned CNN & VLM (both fine-tuned on the SAME 100 balanced images)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/results/which_lsb_final_comparison.png"

# palette (validated dataviz set)
L_BLUE, ORANGE, VIOLET, GREEN, BLUE = "#9ec5f4", "#eb6834", "#4a3aa7", "#008300", "#2a78d6"
PRIMARY, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

# (label, value, color, is_finetuned) — ascending
bars = [
    ("Qwen2-VL-7B\n(zero-shot)",      27, L_BLUE, False),
    ("GPT-5.6-sol\n(zero-shot)",          32, ORANGE, False),
    ("Gemini 3.5 Flash\n(zero-shot)", 56, VIOLET, False),
    ("Fine-tuned CNN",                59, GREEN,  True),
    ("Fine-tuned Qwen2-VL-7B",                60, BLUE,   True),
]
labels = [b[0] for b in bars]; vals = [b[1] for b in bars]; colors = [b[2] for b in bars]
CHANCE = 25

fig, ax = plt.subplots(figsize=(9.0, 5.8))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(bars))
ax.bar(x, vals, width=0.62, color=colors, zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.0, f"{v}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=PRIMARY)




ax.set_ylim(0, 74)
ax.set_ylabel("Balanced accuracy (%)", fontsize=11.5, color=SECONDARY)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10, color=PRIMARY)
ax.set_title("which-LSB (4-class low-surface-brightness morphology)",
             fontsize=14, fontweight="bold", color=PRIMARY, pad=30)
ax.text(0.5, 1.045,
        "Balanced accuracy on the same test set  ·  fine-tuned models trained on the same 100 balanced labels",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
