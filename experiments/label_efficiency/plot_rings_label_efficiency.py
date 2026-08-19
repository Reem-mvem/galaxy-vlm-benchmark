import glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = "/scratch/reem1/rings_runs"
SWEEP_DIR = f"{BASE}/results/downstream/rings/narval_repro_public_best_6seed_2026-06-30"
BUNDLE = ("/home/reem1/scratch/geir_files/gz_evo_downstream_predictions_bundle_2026-07-02"
          "/gz_evo_downstream_predictions_bundle_2026-07-02/predictions/gz_rings_convnext_base")
THR = 0.5
TRAIN_TOTAL = 73356          # full gz_rings train split
gemini = 0.613               # VLM (Gemini) zero-shot balanced accuracy on the same 1000 galaxies

# --- the fixed 1000 VLM galaxies: id_str "test_<N>", true = ring_fraction >= 0.5 ---
vlm = pd.read_csv(f"{BASE}/rings_results.csv")
sample_ids = vlm["id_str"].tolist()                       # "test_<N>"
y_true = (vlm["ring_fraction"].values >= THR).astype(int)

runs = {}   # n_images -> [balanced_accuracy per seed]

# --- sweep points: predictions_test.csv has id_str (test_N) + regression ---
for f in glob.glob(f"{SWEEP_DIR}/div*/seed*_*/predictions_test.csv"):
    div = int(re.search(r"/div(\d+)/", f).group(1))
    p = pd.read_csv(f).set_index("id_str").reindex(sample_ids)
    y_pred = (p["regression"].values >= THR).astype(int)
    runs.setdefault(TRAIN_TOTAL // div, []).append(balanced_accuracy_score(y_true, y_pred))

# --- 100% data: bundle (6 seeds); row_index -> "test_<row_index>" ---
for f in glob.glob(f"{BUNDLE}/seed_*/predictions_test.csv"):
    p = pd.read_csv(f)
    p["id_str"] = "test_" + p["row_index"].astype(str)
    p = p.set_index("id_str").reindex(sample_ids)
    y_pred = (p["regression"].values >= THR).astype(int)
    runs.setdefault(TRAIN_TOTAL, []).append(balanced_accuracy_score(y_true, y_pred))

xs    = sorted(runs.keys())
means = [np.mean(runs[x]) for x in xs]
stds  = [np.std(runs[x])  for x in xs]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# individual seed points (faint, jittered) — every point incl. 100%
for x in xs:
    seeds = runs[x]
    jitter = x * np.random.uniform(-0.02, 0.02, size=len(seeds))
    ax.scatter(np.full(len(seeds), x) + jitter, seeds,
               s=22, color='#2a78d6', alpha=0.35, zorder=2, edgecolors='none')

# mean +/- std line through all points, including 100%
ax.errorbar(xs, means, yerr=stds, fmt='o-', color='#2a78d6',
            ecolor='#2a78d6', elinewidth=1.5, capsize=4,
            markersize=7, markerfacecolor='#2a78d6',
            markeredgecolor='white', markeredgewidth=1.5, linewidth=2,
            label='Fine-tuned CNN (ConvNeXt-Base)\nmean ± 1 std, dots = individual seeds',
            zorder=4)

# Gemini baseline
ax.axhline(gemini, color='#e34948', linestyle='--', linewidth=2,
           label=f'Gemini zero-shot = {gemini:.3f}', zorder=3)

ax.set_xscale('log')
ax.set_xlabel('Number of training images (log scale)', fontsize=12)
ax.set_ylabel('Balanced accuracy', fontsize=12)
ax.set_title('gz_rings: label efficiency — fine-tuned CNN vs Gemini zero-shot', fontsize=13)
ax.set_xticks(xs)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.minorticks_off()
ax.set_xticklabels([f"{v:,}" for v in xs], rotation=35, ha="right")
ax.set_ylim(0.45, 0.92)
ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.35)
ax.legend(loc='lower right', fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig(f'{BASE}/rings_label_efficiency.png', dpi=150, bbox_inches='tight')

print("n_images :  mean_bal_acc  std   (per-seed)")
for x in xs:
    print(f"{x:7d} :  {np.mean(runs[x]):.4f}  {np.std(runs[x]):.4f}  {[round(v,4) for v in runs[x]]}")
print("saved -> rings_label_efficiency.png")
