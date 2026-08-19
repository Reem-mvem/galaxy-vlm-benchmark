"""Full benchmark: 3 VLMs across 5 galaxy tasks, grouped bars.

Four tasks are scored by balanced accuracy; euclid (flagged *) by per-galaxy
path accuracy. Bars are the validated categorical slots (Gemini blue, OpenAI
orange, Qwen aqua) with rounded data-ends, direct value labels, a recessive
grid, and a light 0.5 chance line for the binary tasks.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

# palette (validated reference instance, light mode)
SURFACE  = "#fcfcfb"; INK = "#0b0b0b"; INK_2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"; BAND = "#f2f1ec"
GEMINI_C = "#2a78d6"   # slot 1 blue
OPENAI_C = "#eb6834"   # slot 2 orange
QWEN_C   = "#1baf7a"   # slot 3 aqua

# task, metric-flag, {model: score in %}   (euclid = path accuracy; rest balanced accuracy)
TASKS = [
    ("is_lsb",      "", {"g": 91.2, "o": 78.3, "q": 84.2}),
    ("strong_lens", "", {"g": 73.0, "o": 52.5, "q": 48.7}),
    ("rings",       "", {"g": 57.9, "o": 51.2, "q": 51.2}),
    ("which_lsb",   "", {"g": 55.8, "o": 31.6, "q": 41.3}),   # 4-class: chance = 25
    ("euclid*",     "", {"g": 73.7, "o": 64.7, "q": 67.1}),   # path accuracy
]
SERIES = [("g", GEMINI_C, "Gemini"), ("o", OPENAI_C, "OpenAI"), ("q", QWEN_C, "Qwen")]

BAR_W = 0.24; GAP = 0.02; ROUND_X = 0.02; Y_PER_X = 78.0

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

# shade the euclid group to signal a different metric
ax.axvspan(len(TASKS) - 1 - 0.45, len(TASKS) - 1 + 0.45, color=BAND, zorder=0)

def column(x, value, color):
    ax.add_patch(FancyBboxPatch(
        (x - BAR_W / 2, -ROUND_X * Y_PER_X), BAR_W, value + ROUND_X * Y_PER_X,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=ROUND_X),
        mutation_aspect=Y_PER_X, fc=color, ec="none", zorder=3, clip_on=True))
    ax.text(x, value + 1.6, f"{value:.0f}", ha="center", va="bottom",
            fontsize=8.6, fontweight="semibold", color=INK, zorder=4)

for gi, (task, flag, scores) in enumerate(TASKS):
    for si, (k, color, _) in enumerate(SERIES):
        column(gi + (si - 1) * (BAR_W + GAP), scores[k], color)

# chance line for the binary tasks (0.5); note which_lsb / euclid differ
ax.axhline(50, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)


ax.set_xlim(-0.6, len(TASKS) - 0.4)
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=9, color=MUTED)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([t for t, _, _ in TASKS], fontsize=10.5, color=INK_2)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(BASELINE); ax.tick_params(length=0)

# legend
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for _, c, _ in SERIES]
ax.legend(handles, [n for _, _, n in SERIES], loc="upper right",
          bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=9.5,
          labelcolor=INK_2, ncol=3, handlelength=1.0, handleheight=1.0,
          columnspacing=1.3)

fig.text(0.065, 0.965, "Three models across five galaxy tasks",
         fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.065, 0.908,
         "Balanced accuracy (path accuracy for euclid*). Gemini leads every task; "
         ,
         fontsize=9.2, color=INK_2, ha="left", va="top")

fig.subplots_adjust(top=0.80, bottom=0.11, left=0.065, right=0.975)

out = "benchmark_all_models.png"
fig.savefig(out, facecolor=SURFACE); print(f"saved {out}")
