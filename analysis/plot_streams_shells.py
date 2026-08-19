"""Streams / Shells / Neither — classifier comparison (for the data provider).
Balanced accuracy on the 30-image balanced test (10/10/10). Fine-tuned models trained on
the 72 training labels; zero-shot models get no training. CNN = mean of 2 seeds (error bar
= seed range). DESI Legacy Survey (astronolan/streams_shells_neither)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/analysis/streams_shells_comparison.png"

L_BLUE, ORANGE, VIOLET, GREEN, BLUE = "#9ec5f4", "#eb6834", "#4a3aa7", "#008300", "#2a78d6"
PRIMARY, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# (label, balanced-acc %, color, yerr or None)  — ascending
bars = [
    ("Qwen3.5-4B\n(zero-shot)",            33.3, L_BLUE, None),
    ("Gemini 3.5 Flash\n(zero-shot)",      43.3, VIOLET, None),
    ("MaxViT-Tiny\n(fine-tuned, 72 labels)", 70.0, GREEN, None),  # 2-seed mean
    ("Qwen3.5-4B\n(fine-tuned, 72 labels)",83.3, BLUE,   None),
]
labels = [b[0] for b in bars]; vals = [b[1] for b in bars]
colors = [b[2] for b in bars]; errs = [b[3] for b in bars]
CHANCE = 33.3

fig, ax = plt.subplots(figsize=(9.2, 6.0))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(bars))
ax.bar(x, vals, width=0.62, color=colors, zorder=3)
# CNN seed-range error bar
for xi, v, e in zip(x, vals, errs):
    if e:
        ax.errorbar(xi, v, yerr=e, fmt="none", ecolor=SECONDARY, elinewidth=1.5,
                    capsize=5, capthick=1.5, zorder=4)
for xi, v, e in zip(x, vals, errs):
    off = (e or 0) + 1.4
    ax.text(xi, v + off, f"{v:.0f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=PRIMARY)

# chance line
ax.axhline(CHANCE, color=BASELINE, lw=1.2, ls="--", zorder=1)
ax.text(len(bars) - 0.52, CHANCE + 1.0, "chance (3-class) = 33%", ha="right", va="bottom",
        fontsize=9, color=MUTED)

ax.set_ylim(0, 96)
ax.set_ylabel("Balanced accuracy (%)", fontsize=12, color=SECONDARY)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10.5, color=PRIMARY)
ax.set_title("Classifying tidal features:  streams / shells / neither",
             fontsize=15, fontweight="bold", color=PRIMARY, pad=30)
ax.text(0.5, 1.045,
        "Balanced accuracy on 30 held-out galaxies (10/10/10)  ·  fine-tuned on 72 labels  ·  DESI Legacy Survey",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
