"""Rings (binary) — balanced accuracy vs mean thinking tokens. Gemini 3.5 Flash, n=1000, 1 seed/level."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/rings_thinking_curve.png"
CORAL, INK, MUT, GRID = "#d85a30", "#0b0b0b", "#7c7a74", "#e1e0d9"

# (level, mean thinking tokens, balanced acc %)
R = [("off",0,57.1),("minimal",0,56.8),("low",115,55.3),("medium",221,54.5),("high",320,54.4)]
ERR = 1.3   # binomial SE, n=1000
# per-point label offset (dx,dy in points): off ABOVE, minimal BELOW (they share x=0), rest ABOVE
LAB = {"off":(0,32),"minimal":(0,-46),"low":(0,32),"medium":(0,32),"high":(0,32)}

fig, ax = plt.subplots(figsize=(12.5, 7.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

ax.axhline(50, color="#c3c2b7", lw=1.3, ls="--", zorder=1)
ax.text(345, 50.5, "chance 50%", ha="right", fontsize=15, color=MUT)

x=[v for _,v,_ in R]; y=[a for _,_,a in R]
ax.plot(x, y, "-", color=CORAL, lw=3.0, zorder=3)
ax.errorbar(x, y, yerr=[ERR]*len(R), fmt="o", color=CORAL, ms=22, mec=INK, mew=2.0,
            ecolor=CORAL, elinewidth=2.2, capsize=8, zorder=4)
for lvl,vx,acc in R:
    dx,dy = LAB[lvl]
    va = "bottom" if dy > 0 else "top"
    ax.annotate(f"{lvl}\n{acc:.1f}", (vx,acc), textcoords="offset points",
                xytext=(dx,dy), ha="center", va=va, fontsize=18, fontweight="bold", color=INK)

ax.set_xlabel("Mean thinking tokens", fontsize=25, color=INK, labelpad=12)
ax.set_ylabel("Balanced accuracy (%)", fontsize=25, color=INK, labelpad=12)
ax.set_title("More reasoning doesn't mean higher accuracy",
             fontsize=25, fontweight="bold", color=INK, pad=48)
ax.text(0.5, 1.03, "Rings  ·  Gemini 3.5 Flash, zero-shot",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=16, color=MUT)
ax.set_ylim(46, 63); ax.set_xlim(-28, 368)
ax.grid(color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUT, labelsize=16)

fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
