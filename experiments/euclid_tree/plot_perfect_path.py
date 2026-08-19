"""Result chart: perfect-path rate, CNN vs Gemini on GZ Euclid, saved as PNG.

A galaxy counts as a "perfect path" when the model answers every question on
its classification path correctly (path = questions routed to >=50% of the
galaxy's volunteers; correct = matches the volunteers' majority answer).
Numbers from path_scores.py on the 1000-galaxy sample.

Same mark specs and palette as plot_results.py: slim columns with rounded
data-ends (square at the baseline), hairline gridlines, text in ink tokens,
value labels on every cap.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

# palette (validated reference instance, light mode; same entities as plot_results.py)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
CNN_C     = "#2a78d6"   # categorical slot 1 (blue)
GEMINI_C  = "#1baf7a"   # categorical slot 2 (aqua)

MODELS = ["Fine-tuned CNN\n(maxvit, mean of 6 seeds)",
          "Gemini zero-shot\n(structured output)"]
VALUES = [65.1, 37.6]           # % of galaxies, from path_scores.py
LABELS = ["65.1%", "37.6%"]     # CNN seed std 0.4 pp, noted in subtitle
COLORS = [CNN_C, GEMINI_C]

BAR_W   = 0.16     # slim: band is mostly air
ROUND_X = 0.030    # data-end corner radius, x units
Y_PER_X = 52.0     # visual y/x scale ratio, for circular corners
ROUND_Y = ROUND_X * Y_PER_X

fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for x, (value, color, label) in enumerate(zip(VALUES, COLORS, LABELS)):
    # rounded corners also sit below the baseline and get clipped -> square base
    ax.add_patch(FancyBboxPatch(
        (x - BAR_W / 2, -ROUND_Y), BAR_W, value + ROUND_Y,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=ROUND_X),
        mutation_aspect=Y_PER_X,
        fc=color, ec="none", zorder=3, clip_on=True,
    ))
    ax.text(x, value + 1.6, label, ha="center", va="bottom",
            fontsize=11, fontweight="semibold", color=INK, zorder=4)

# axes & grid: recessive; headroom above the taller cap label
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=9, color=MUTED)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, fontsize=10, color=INK_2)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(length=0)

# title + subtitle at figure level, with real breathing room
fig.text(0.07, 0.955, "How often is the whole classification right?",
         fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.07, 0.895,
         "Galaxies with every path question answered correctly\n"
         "1000-galaxy GZ Euclid test sample; CNN seed std 0.4 pp",
         fontsize=9.5, color=INK_2, ha="left", va="top")
fig.subplots_adjust(top=0.80, bottom=0.13, left=0.12, right=0.95)

out = "gz_euclid_perfect_path.png"
fig.savefig(out, facecolor=SURFACE)
print(f"saved {out}")
