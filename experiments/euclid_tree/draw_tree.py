"""Draw the GZ Euclid 13-question decision tree as a PNG.

Boxes = questions (with their allowed answers); arrows = the volunteer flow,
labelled with the answer that routes there (unlabelled = always asked next).
Merging is the final question of every branch except problem/artifact.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# branch colours
C_ROOT     = ("#e8eef7", "#3b5b8c")
C_SMOOTH   = ("#e6f2e6", "#3d7a3d")
C_FEATURED = ("#fdf1e2", "#b06818")
C_PROBLEM  = ("#fbe9e9", "#a33636")
C_MERGING  = ("#efeaf6", "#6a4f9e")

NODES = {
    # name: (x, y, question, [answers], (fill, edge))
    "root":       (50, 94, "smooth-or-featured", ["smooth", "featured-or-disk", "problem"], C_ROOT),
    "rounded":    (9, 72, "how-rounded", ["round", "in-between", "cigar-shaped"], C_SMOOTH),
    "edgeon":     (37, 72, "disk-edge-on", ["yes", "no"], C_FEATURED),
    "clumps":     (60, 72, "clumps", ["yes", "no"], C_FEATURED),
    "problem":    (86, 74, "problem", ["star", "artifact", "zoom"], C_PROBLEM),
    "eobulge":    (20, 48, "edge-on-bulge", ["boxy", "none", "rounded"], C_FEATURED),
    "bar":        (40, 48, "bar", ["strong", "weak", "no"], C_FEATURED),
    "spiral":     (58, 48, "has-spiral-arms", ["yes", "no"], C_FEATURED),
    "artifact":   (90, 46, "artifact", ["satellite", "scattered", "diffraction",
                                        "ray", "saturation", "other", "ghost"], C_PROBLEM),
    "winding":    (50, 28, "spiral-winding", ["tight", "medium", "loose"], C_FEATURED),
    "armcount":   (66, 28, "spiral-arm-count", ["1", "2", "3", "4", "more-than-4", "cant-tell"], C_FEATURED),
    "bulgesize":  (82, 20, "bulge-size", ["dominant", "large", "moderate", "small", "none"], C_FEATURED),
    "merging":    (14, 20, "merging", ["none", "minor-disturbance",
                                       "major-disturbance", "merger"], C_MERGING),
}

# (from, src_side, to, dst_side, label, curve, label_t)
EDGES = [
    ("root", "b", "rounded", "t",  "smooth",           0.15, 0.5),
    ("root", "b", "edgeon",  "t",  "featured-or-disk", 0.0,  0.5),
    ("root", "b", "clumps",  "t",  "featured-or-disk", 0.0,  0.5),
    ("root", "b", "problem", "t",  "problem",         -0.15, 0.5),
    ("edgeon", "b", "eobulge", "t", "yes",  0.1, 0.5),
    ("edgeon", "b", "bar",     "t", "no",   0.0, 0.5),
    ("bar", "r", "spiral", "l", "", 0.0, 0.5),
    ("spiral", "b", "winding",   "t", "yes", 0.1,  0.5),
    ("winding", "r", "armcount", "l", "",    0.0,  0.5),
    ("armcount", "r", "bulgesize", "l", "",  0.0,  0.5),
    ("spiral", "b", "bulgesize", "t", "no", -0.25, 0.1),
    ("problem", "b", "artifact", "t", "artifact", 0.0, 0.5),
    # merging: final question of every branch except problem/artifact
    ("rounded", "b", "merging",   "t", "", 0.15, 0.5),
    ("eobulge", "b", "merging",   "t", "", -0.1, 0.5),
    ("bulgesize", "b", "merging", "r", "", 0.2,  0.5),
]

BOX_W_PER_CHAR = 0.62
LINE_H = 3.0


def node_box(name):
    x, y, q, answers, _ = NODES[name]
    w = max(len(q) + 2, max(len(a) for a in answers) + 2) * BOX_W_PER_CHAR
    h = (len(answers) + 1) * LINE_H + 2.5
    return x, y, w, h


def anchor(name, side):
    x, y, w, h = node_box(name)
    return {"t": (x, y + h / 2), "b": (x, y - h / 2),
            "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]


def bezier_point(p0, p1, rad, t):
    """Point at parameter t on matplotlib's arc3 quadratic bezier."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    cx, cy = mx - rad * dy, my + rad * dx  # control point, perpendicular offset
    u = 1 - t
    return (u * u * p0[0] + 2 * u * t * cx + t * t * p1[0],
            u * u * p0[1] + 2 * u * t * cy + t * t * p1[1])


fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 104)
ax.axis("off")

# edges first, under the boxes
for src, s_side, dst, d_side, label, curve, label_t in EDGES:
    p0, p1 = anchor(src, s_side), anchor(dst, d_side)
    color = NODES[dst][4][1]
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=color, alpha=0.85, zorder=1,
    ))
    if label:
        lx, ly = bezier_point(p0, p1, curve, label_t)
        ax.text(lx, ly, label, fontsize=8.5, style="italic", color=color,
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

# boxes
for name, (x, y, q, answers, (fill, edge)) in NODES.items():
    _, _, w, h = node_box(name)
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.6", fc=fill, ec=edge, lw=1.6, zorder=2,
    ))
    ax.text(x, y + h / 2 - LINE_H * 0.7, q, fontsize=10.5, weight="bold",
            color=edge, ha="center", va="center", zorder=3)
    for i, a in enumerate(answers):
        ax.text(x, y + h / 2 - LINE_H * (i + 1.8), a, fontsize=8.5,
                color="#333333", ha="center", va="center", zorder=3)

ax.text(26, 34, "final question of every branch\nexcept problem/artifact",
        fontsize=8, style="italic", color=C_MERGING[1], ha="center")

ax.set_title("Galaxy Zoo Euclid — 13-question decision tree",
             fontsize=15, weight="bold", pad=14)

fig.tight_layout()
out = "gz_euclid_decision_tree.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
