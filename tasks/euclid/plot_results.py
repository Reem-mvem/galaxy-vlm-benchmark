"""Result chart: CNN vs Gemini on GZ Euclid, two metrics, saved as PNG.

Follows the dataviz mark specs: slim columns with rounded data-ends (square at
the baseline), 2px surface gap between paired bars, hairline gridlines, text
in ink tokens (never the series color), legend + value labels on every cap.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

# palette (validated reference instance, light mode)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
CNN_C     = "#2a78d6"   # categorical slot 1 (blue)
GEMINI_C  = "#1baf7a"   # categorical slot 2 (aqua)

METRICS = ["Path accuracy\n(all path questions, n=1000)",
           "Confident galaxies\n(volunteer majority ≥ 0.8)"]
CNN    = [89.4, 99.8]
GEMINI = [75.0, 88.5]

BAR_W   = 0.11     # slim: band is mostly air
GAP     = 0.012    # ~2px surface gap between the paired bars
ROUND_X = 0.030    # data-end corner radius, x units
Y_PER_X = 86.0     # visual y/x scale ratio, for circular corners
ROUND_Y = ROUND_X * Y_PER_X

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

def column(x, value, color, label_dx=0.0):
    # rounded corners also sit below the baseline and get clipped -> square base
    ax.add_patch(FancyBboxPatch(
        (x - BAR_W / 2, -ROUND_Y), BAR_W, value + ROUND_Y,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=ROUND_X),
        mutation_aspect=Y_PER_X,
        fc=color, ec="none", zorder=3, clip_on=True,
    ))
    ax.text(x + label_dx, value + 2.0, f"{value:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="semibold", color=INK, zorder=4)

for i, _ in enumerate(METRICS):
    column(i - (BAR_W + GAP) / 2, CNN[i], CNN_C, label_dx=-0.012)
    column(i + (BAR_W + GAP) / 2, GEMINI[i], GEMINI_C, label_dx=0.022)

# axes & grid: recessive; headroom above 100 for legend and cap labels
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 114)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                   fontsize=9, color=MUTED)
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(METRICS, fontsize=10, color=INK_2)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(length=0)

# legend in the headroom band, clear of both groups' cap labels
handles = [plt.Rectangle((0, 0), 1, 1, fc=CNN_C, ec="none"),
           plt.Rectangle((0, 0), 1, 1, fc=GEMINI_C, ec="none")]
ax.legend(handles, ["Fine-tuned CNN (maxvit, mean of 6 seeds)",
                    "Gemini zero-shot (structured output)"],
          loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=False,
          fontsize=9, labelcolor=INK_2, handlelength=1.0, handleheight=1.0)

# title + subtitle at figure level, with real breathing room
fig.text(0.06, 0.965, "GZ Euclid morphology: fine-tuned CNN vs zero-shot Gemini",
         fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.06, 0.915, "Accuracy vs volunteer majority answers, 1000-galaxy test sample",
         fontsize=9.5, color=INK_2, ha="left", va="top")
fig.subplots_adjust(top=0.84, bottom=0.12, left=0.09, right=0.97)

out = "gz_euclid_results.png"
fig.savefig(out, facecolor=SURFACE)
print(f"saved {out}")
