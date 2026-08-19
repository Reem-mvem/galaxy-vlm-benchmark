"""Voting experiment summary: does majority voting beat the best single model?

Diverging horizontal bars — one per task — of (vote - best single model), in
points. Positive (blue) = the 3-model majority vote beat the best individual
model; negative (red) = it did worse. The best single model is Gemini on every
task. Metric is balanced accuracy for the four classification tasks and
per-galaxy path accuracy for euclid (noted in the caption).

Follows the dataviz mark specs: slim bars with rounded data-ends (square at the
x=0 baseline), diverging blue<->red poles, gray zero line, text in ink tokens.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

# palette (validated reference instance, light mode)
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
ZERO     = "#c3c2b7"
POS_C    = "#2a78d6"   # diverging cool pole  -> voting helped
NEG_C    = "#e34948"   # diverging warm pole  -> voting hurt

# task, best-single (Gemini) score, vote score, metric label   (all in %)
ROWS = [
    ("euclid",       73.7, 75.4, "path acc."),
    ("is_lsb",       91.2, 92.4, "bal. acc."),
    ("which_lsb",    55.8, 51.4, "bal. acc."),
    ("rings",        57.9, 52.4, "bal. acc."),
    ("strong_lens",  73.0, 55.4, "bal. acc."),
]
# sorted most-positive (top) to most-negative (bottom)
ROWS.sort(key=lambda r: r[2] - r[1], reverse=True)

labels = [r[0] for r in ROWS]
deltas = [r[2] - r[1] for r in ROWS]

BAR_H   = 0.42
ROUND_X = 0.55                 # data-end corner radius, in x units (points)
X_PER_Y = None                 # set after we know the aspect

fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

xmin, xmax = -20, 6
# aspect for circular corners: data-units per axis-unit
x_span, y_span = (xmax - xmin), len(ROWS)
Y_PER_X = (y_span / x_span) * (7.8 / 4.4)   # approx

def bar(y, value, color):
    # rounded end at the value; square end clipped at x=0 baseline
    x0 = min(0, value)
    w  = abs(value)
    # extend slightly past 0 so the rounding sits beyond the baseline and clips square
    if value >= 0:
        rect = FancyBboxPatch((0 - ROUND_X, y - BAR_H / 2), w + ROUND_X, BAR_H,
                              boxstyle=BoxStyle("Round", pad=0, rounding_size=ROUND_X),
                              mutation_aspect=Y_PER_X, fc=color, ec="none",
                              zorder=3, clip_on=True)
    else:
        rect = FancyBboxPatch((x0, y - BAR_H / 2), w + ROUND_X, BAR_H,
                              boxstyle=BoxStyle("Round", pad=0, rounding_size=ROUND_X),
                              mutation_aspect=Y_PER_X, fc=color, ec="none",
                              zorder=3, clip_on=True)
    ax.add_patch(rect)

ys = list(range(len(ROWS)))[::-1]   # first row at top
for y, (name, best, vote, metric), d in zip(ys, ROWS, deltas):
    bar(y, d, POS_C if d >= 0 else NEG_C)
    # value label just past the bar end
    if d >= 0:
        ax.text(d + 0.5, y, f"+{d:.1f}", va="center", ha="left",
                fontsize=10, fontweight="semibold", color=INK, zorder=4)
    else:
        ax.text(d - 0.5, y, f"{d:.1f}", va="center", ha="right",
                fontsize=10, fontweight="semibold", color=INK, zorder=4)
    # right-edge context: best vs vote scores
    ax.text(xmax + 0.4, y, f"{best:.0f}→{vote:.0f}%  {metric}",
            va="center", ha="left", fontsize=8.5, color=MUTED, zorder=4)

# task names as y ticks
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=10.5, color=INK_2)

# zero line + gridlines
ax.axvline(0, color=ZERO, linewidth=1.4, zorder=2)
for gx in range(xmin, xmax + 1, 5):
    if gx != 0:
        ax.axvline(gx, color=GRID, linewidth=1, zorder=0)
ax.set_xlim(xmin, xmax + 8.5)   # headroom on right for the context labels
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.set_xticks(range(xmin, xmax + 1, 5))
ax.set_xticklabels([f"{t:+d}" if t else "0" for t in range(xmin, xmax + 1, 5)],
                   fontsize=9, color=MUTED)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(ZERO)
ax.tick_params(length=0)

# small pole cues under the axis
ax.text(-10, -1.15, "← voting did worse", ha="center", va="top",
        fontsize=8.5, color=NEG_C)
ax.text(3, -1.15, "voting helped →", ha="center", va="top",
        fontsize=8.5, color=POS_C)

# title + subtitle
fig.text(0.06, 0.965, "Does 3-model majority voting beat the best single model?",
         fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.06, 0.905,
         "Vote − best single model (Gemini won every task). It helped only where all "
         "three models were strong.",
         fontsize=9.0, color=INK_2, ha="left", va="top")
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.13, right=0.97)

out = "vote_vs_best_model.png"
fig.savefig(out, facecolor=SURFACE)
print(f"saved {out}")
