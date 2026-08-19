"""ONE big label-efficiency figure across all 5 tasks. CNN curve = its own color per task;
Gemini baseline = ONE standardized color everywhere. Legend at the bottom. Large & poster-scale.
Rings is cached (rings_curve_cache.csv) so edits render instantly."""
import glob, re, os
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

OUT = "/home/reem1/scratch/galaxy-vlm-benchmark/docs/figures/label_efficiency_all.png"
RINGS_CACHE = "/home/reem1/scratch/galaxy-vlm-benchmark/analysis/rings_curve_cache.csv"

GEM_C = "#2b2b2b"     # ONE Gemini baseline color for every panel
DOT_S = 150           # seed dot size (big)
MK_S  = 22            # mean marker size (big)
LW    = 4.0

NICE = {40:"40",50:"50",73:"73",99:"100",100:"100",127:"130",201:"200",254:"250",
        499:"500",503:"500",508:"500",733:"730",839:"840",1006:"1,000",1469:"1,500",
        1998:"2,000",2013:"2,000",2938:"3,000",4997:"5,000",5876:"6,000",7335:"7,300",
        9994:"10,000",18339:"18,000",24271:"24,000",36678:"37,000",42474:"42,000",
        73356:"73,000",84949:"85,000",169899:"170,000"}
nice = lambda v: NICE.get(int(v), f"{int(v):,}")

bal = pd.read_csv("/scratch/reem1/rings_runs/analysis/balacc_classification.csv")
def clf_curve(task):
    d = bal[bal.task == task]; xs = sorted(d.n_images.unique())
    return xs, {x: d[d.n_images == x].bal_acc.values for x in xs}

def euclid_curve():
    d = pd.read_csv("/home/reem1/scratch/rings_runs/analysis/euclid/euclid_path_accuracy.csv")
    d = d[d.n_images.isin({99, 499, 1998, 9994, 42474, 169899})]
    xs = sorted(d.n_images.unique())
    return xs, {x: d[d.n_images == x].path_acc.values for x in xs}

def rings_curve():
    if os.path.exists(RINGS_CACHE):
        d = pd.read_csv(RINGS_CACHE); xs = sorted(d.n_images.unique())
        return xs, {x: d[d.n_images == x].bal_acc.values for x in xs}
    BASE = "/scratch/reem1/rings_runs"
    SW = f"{BASE}/results/downstream/rings/narval_repro_public_best_6seed_2026-06-30"
    BUN = ("/home/reem1/scratch/geir_files/gz_evo_downstream_predictions_bundle_2026-07-02"
           "/gz_evo_downstream_predictions_bundle_2026-07-02/predictions/gz_rings_convnext_base")
    THR, TOT = 0.5, 73356
    vlm = pd.read_csv(f"{BASE}/rings_results.csv"); ids = vlm["id_str"].tolist()
    y = (vlm["ring_fraction"].values >= THR).astype(int); runs = {}
    for f in glob.glob(f"{SW}/div*/seed*_*/predictions_test.csv"):
        div = int(re.search(r"/div(\d+)/", f).group(1))
        p = pd.read_csv(f).set_index("id_str").reindex(ids)
        runs.setdefault(TOT // div, []).append(balanced_accuracy_score(y, (p["regression"].values >= THR).astype(int)))
    for f in glob.glob(f"{BUN}/seed_*/predictions_test.csv"):
        p = pd.read_csv(f); p["id_str"] = "test_" + p["row_index"].astype(str)
        p = p.set_index("id_str").reindex(ids)
        runs.setdefault(TOT, []).append(balanced_accuracy_score(y, (p["regression"].values >= THR).astype(int)))
    xs = sorted(runs.keys())
    pd.DataFrame([(x, v) for x in xs for v in runs[x]], columns=["n_images", "bal_acc"]).to_csv(RINGS_CACHE, index=False)
    return xs, {x: np.array(runs[x]) for x in xs}

# optional xpos: plot a point at a shifted x (label keeps its true value) to de-crowd.
# is_lsb: 40 & 50 overlap on log; move "50" to the log-midpoint of 40 and 100 (~63).
PANELS = [
    ("Is Debris (is-lsb) — MaxViT-Tiny",       "Balanced accuracy",             0.912, lambda: clf_curve("is_lsb"),      "#1b9e77", {50: 63}),
    ("Which Debris (which-lsb) — MaxViT-Tiny", "Balanced accuracy",             0.630, lambda: clf_curve("which_lsb"),   "#7570b3", {}),
    ("Strong Lenses — SoViT-400m/14",          "Balanced accuracy",             0.730, lambda: clf_curve("strong_lens"), "#d95f02", {}),
    ("Rings — ConvNeXt-Base",                  "Balanced accuracy",             0.613, rings_curve,                       "#2a78d6", {}),
    ("Morphology — MaxViT-Base",               "Path accuracy (13-question tree)", 0.750, euclid_curve,                   "#c1272d", {}),
]

fig, axes = plt.subplots(2, 3, figsize=(30, 17))
fig.patch.set_facecolor("white"); axes = axes.ravel()

def draw(ax, title, ylabel, gemini, getter, c, xpos=None):
    xs, runs = getter()
    xpos = xpos or {}
    px = [xpos.get(x, x) for x in xs]                     # plotted x (may be shifted to de-crowd)
    means = [runs[x].mean() for x in xs]; stds = [runs[x].std() for x in xs]
    for x, p in zip(xs, px):
        v = runs[x]; jit = p * np.random.uniform(-0.02, 0.02, len(v))
        ax.scatter(np.full(len(v), p) + jit, v, s=DOT_S, color=c, alpha=0.30, edgecolors="none", zorder=2)
    ax.errorbar(px, means, yerr=stds, fmt="o-", color=c, ecolor=c, elinewidth=2.5, capsize=7,
                markersize=MK_S, markerfacecolor=c, markeredgecolor="white", markeredgewidth=2.5,
                linewidth=LW, zorder=4)
    ax.axhline(gemini, color=GEM_C, ls="--", lw=3.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of training images (log scale)", fontsize=27)
    ax.set_ylabel(ylabel, fontsize=27)
    ax.set_title(title, fontsize=30, fontweight="bold", pad=12)
    ax.set_xticks(px); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticklabels([nice(v) for v in xs], rotation=32, ha="right", fontsize=25)
    ax.tick_params(axis="y", labelsize=25)
    lo = min(min(means) - max(stds), gemini) - 0.03
    ax.set_ylim(max(0, lo), 1.005)
    ax.grid(True, which="both", ls="-", lw=0.5, alpha=0.35)

for ax, p in zip(axes, PANELS):
    draw(ax, *p)
axes[5].axis("off")   # 6th cell empty

# legend at the BOTTOM, spanning, horizontal
handles = [
    Line2D([0], [0], color="#666666", marker="o", markersize=18, markerfacecolor="#666666",
           markeredgecolor="white", markeredgewidth=2.5, linewidth=LW,
           label="Fine-tuned CNN  (color per task above;  mean ± 1 std across seeds, dots = individual seeds)"),
    Line2D([0], [0], color=GEM_C, ls="--", lw=3.5, label="Gemini 3.5 Flash — zero-shot baseline"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2,
           fontsize=25, frameon=True, handlelength=3.0, columnspacing=5, borderpad=1.2)

fig.suptitle("Label efficiency across 5 galaxy tasks:  fine-tuned CNN vs Gemini zero-shot",
             fontsize=40, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0.11, 1, 0.965], h_pad=4.0)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved ->", OUT)
