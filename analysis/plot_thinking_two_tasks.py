"""Does thinking help? It depends on the task.
Left: streams/shells (3-class, real-world) — thinking HELPS (rises to a sweet spot).
Right: rings (binary) — thinking HURTS (monotonic decline). Accuracy vs mean thinking tokens."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/thinking_two_tasks.png"
TEAL, CORAL = "#1baf7a", "#d85a30"
INK, MUT, GRID = "#0b0b0b", "#7c7a74", "#e1e0d9"

# streams: 3-seed mean per level (thinking tokens, balanced acc %, std)
S = [("off",0,50.0,3.3),("minimal",0,51.1,1.9),("low",269,56.7,3.4),
     ("medium",787,62.2,1.9),("high",1096,58.9,1.9)]
S_seeds = {"off":[53.3,50.0,46.7],"minimal":[53.3,50.0,50.0],"low":[53.3,60.0,56.7],
           "medium":[63.3,60.0,63.3],"high":[60.0,56.7,60.0]}
# rings: single seed, n=1000 (thinking tokens, balanced acc %); err ~ binomial SE
R = [("off",0,57.1),("minimal",0,56.8),("low",115,55.3),("medium",221,54.5),("high",320,54.4)]
R_ERR = 1.3

fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 7.4))
fig.patch.set_facecolor("#fcfcfb")

# ---- LEFT: streams (helps) ----
axL.set_facecolor("#fcfcfb")
axL.axhline(33.3, color="#c3c2b7", lw=1.3, ls="--", zorder=1)
axL.text(1150, 34.5, "chance 33%", ha="right", fontsize=13, color=MUT)
for lvl,x,acc,std in S:
    axL.scatter([x]*3, S_seeds[lvl], s=70, color=TEAL, alpha=0.25, edgecolor="none", zorder=2)
mx=[x for _,x,_,_ in S]; my=[a for _,_,a,_ in S]; me=[e for _,_,_,e in S]
axL.plot(mx,my,"-",color=TEAL,lw=2.6,zorder=3)
axL.errorbar(mx,my,yerr=me,fmt="o",color=TEAL,ms=15,mec=INK,mew=1.6,ecolor=TEAL,elinewidth=2,capsize=6,zorder=4)
for lvl,x,acc,_ in S:
    axL.annotate(lvl,(x,acc),textcoords="offset points",xytext=(0,16),ha="center",fontsize=13,color=INK)
axL.set_title("Streams / shells (3-class, real-world)\nthinking HELPS ↑",
              fontsize=19,fontweight="bold",color=TEAL,pad=14)
axL.set_ylim(30,70); axL.set_xlim(-60,1230)

# ---- RIGHT: rings (hurts) ----
axR.set_facecolor("#fcfcfb")
axR.axhline(50, color="#c3c2b7", lw=1.3, ls="--", zorder=1)
axR.text(335, 50.6, "chance 50%", ha="right", fontsize=13, color=MUT)
rx=[x for _,x,_ in R]; ry=[a for _,_,a in R]
axR.plot(rx,ry,"-",color=CORAL,lw=2.6,zorder=3)
axR.errorbar(rx,ry,yerr=[R_ERR]*len(R),fmt="o",color=CORAL,ms=15,mec=INK,mew=1.6,ecolor=CORAL,elinewidth=2,capsize=6,zorder=4)
for lvl,x,acc in R:
    axR.annotate(lvl,(x,acc),textcoords="offset points",xytext=(0,15),ha="center",fontsize=13,color=INK)
axR.set_title("Rings (binary)\nthinking HURTS ↓   (n=1000)",
              fontsize=19,fontweight="bold",color=CORAL,pad=14)
axR.set_ylim(48,62); axR.set_xlim(-20,360)

for ax in (axL,axR):
    ax.set_xlabel("Mean thinking tokens", fontsize=17, color=INK)
    ax.grid(color=GRID,lw=0.9,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT,labelsize=14)
axL.set_ylabel("Balanced accuracy (%)", fontsize=17, color=INK)

fig.suptitle("Does reasoning help?  It depends on the task.",
             fontsize=24, fontweight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved ->", OUT)
