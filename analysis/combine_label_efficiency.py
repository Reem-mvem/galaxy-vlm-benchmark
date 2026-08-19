"""Combine the 5 per-task label-efficiency figures into one poster image (2x3 grid)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIG = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures"
OUT = f"{FIG}/label_efficiency_all.png"

panels = [
    f"{FIG}/label_is_lsb.png",
    f"{FIG}/label_which_lsb.png",
    f"{FIG}/label_strong_lens.png",
    f"{FIG}/rings_label_efficiency.png",
    f"{FIG}/label_gz_euclid.png",
]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.patch.set_facecolor("white")
axes = axes.ravel()
for ax, p in zip(axes, panels):
    ax.imshow(mpimg.imread(p))
    ax.axis("off")
# hide the empty 6th cell
for ax in axes[len(panels):]:
    ax.axis("off")

fig.suptitle("Label efficiency: fine-tuned CNN vs Gemini zero-shot across 5 galaxy tasks",
             fontsize=20, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
