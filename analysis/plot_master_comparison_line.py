"""Master comparison — balanced accuracy across 4 galaxy tasks, one line per model.
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

TASKS  = ["which_lsb\n(4-class)", "is_lsb", "strong_lens", "rings"]
# (label, color, marker, [which_lsb, is_lsb, strong_lens, rings])
SERIES = [
    ("Gemini 3.5 Flash (zero-shot)", VIOLET, "D", [56.0, 91.2, 73.0, 59.0]),
    ("Fine-tuned CNN",               GREEN,  "s", [59.3, 95.7, 75.2, 54.2]),
    ("Fine-tuned Qwen2-VL-7B",       L_BLUE, "o", [59.5, 92.4, 65.4, 66.0]),
    ("Fine-tuned Qwen3.5-4B",        BLUE,   "o", [63.6, 94.3, 64.1, 68.6]),
    ("Fine-tuned Qwen3.5-9B",        ORANGE, "o", [50.8, 94.1, 70.0, 71.9]),
]

x = np.arange(len(TASKS))
fig, ax = plt.subplots(figsize=(10.5, 6.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

for label, color, marker, vals in SERIES:
    ax.plot(x, vals, color=color, lw=2.2, marker=marker, ms=7,
            markerfacecolor=color, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
            zorder=3, label=label, solid_capstyle="round")

# value labels, nudged to avoid collisions at crowded points
NUDGE = {  # (series_idx, task_idx): (dx, dy)   series: 0=Gemini 1=CNN 2=7B 3=4B 4=9B
    # which_lsb: Gemini 63.0 vs 4B 63.6 close; CNN 59.3 vs 7B 59.5 nearly identical
    (0, 0): (-0.06, -3.4),
    (3, 0): (0.05, 1.6),
    (1, 0): (0, -3.4),
    (2, 0): (-0.07, 1.6),
    # is_lsb: 91-96 stack
    (1, 1): (0, 1.6),
    (3, 1): (-0.07, 1.6),
    (4, 1): (0.07, -3.4),
    (2, 1): (-0.08, -3.4),
    (0, 1): (0.07, -3.4),
    # strong_lens: Gemini 73 sits under CNN 75 marker
    (0, 2): (-0.07, -3.4),
    (4, 2): (0, -3.4),
    (2, 2): (0, 1.6),
    (3, 2): (0, -3.4),
    # rings: 4B 68.6 vs 7B 66.0 near the crossing lines
    (3, 3): (0.05, 1.6),
    (2, 3): (0.05, -3.4),
}
for si, (label, color, marker, vals) in enumerate(SERIES):
    for ti, v in enumerate(vals):
        dx, dy = NUDGE.get((si, ti), (0, 1.6))
        ax.text(x[ti] + dx, v + dy, f"{v:.0f}", ha="center", va="bottom" if dy > 0 else "top",
                fontsize=9, fontweight="bold", color=color)



ax.set_xlim(-0.35, len(TASKS) - 0.65)
ax.set_ylim(45, 100)
ax.set_ylabel("Balanced accuracy (%)", fontsize=12, color=SECONDARY)
ax.set_xticks(x); ax.set_xticklabels(TASKS, fontsize=11.5, color=PRIMARY)
ax.set_title("Fine-tuned CNN vs VLMs — 100 balanced training labels per task",
             fontsize=14.5, fontweight="bold", color=PRIMARY, pad=34)
ax.text(0.5, 1.055,
        "Balanced accuracy, same test set per task  ·  fine-tuned models trained on the identical 100 balanced labels (VLMs: LoRA r=8)  ·  Gemini zero-shot",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.2, color=MUTED)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
ax.tick_params(colors=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3,
          frameon=False, fontsize=9.8, columnspacing=1.4, handlelength=1.8)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
