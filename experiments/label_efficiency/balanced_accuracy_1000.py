"""
Balanced accuracy per divisor for the ConvNeXt-Base gz_rings runs,
evaluated on the SAME 1000 test galaxies used for the VLM (rings_results.csv),
so the two models are directly comparable.

Key facts this relies on:
  * rings_results.csv `id_str` = "test_<N>", where N is the 0-based row index
    into the gz_rings test split.
  * Each ConvNeXt prediction file rings_test_preds_seed<S>.csv is in that same
    test-split order (shuffle=False), so row N == galaxy test_N.
  * True binary class = (ring_fraction >= 0.5); predicted class = (pred >= 0.5).
"""
import glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

BASE = "/scratch/reem1/rings_runs"
PRED_DIR = f"{BASE}/results/downstream/narval_repro_public_best_6seed_2026-06-30"
THR = 0.5

# --- the 1000 VLM galaxies ---
vlm = pd.read_csv(f"{BASE}/rings_results.csv")
idx = vlm["id_str"].str.replace("test_", "", regex=False).astype(int).values
y_true = (vlm["ring_fraction"].values >= THR).astype(int)   # ground truth on the 1000

# --- ConvNeXt balanced accuracy per (divisor, seed) on those 1000 rows ---
rows = []
for f in glob.glob(f"{PRED_DIR}/predict_gz_rings_div*/rings_test_preds_seed*.csv"):
    div  = int(re.search(r"div(\d+)/", f).group(1))
    seed = int(re.search(r"seed(\d+)\.csv", f).group(1))
    pred = pd.read_csv(f)
    y_pred = (pred["pred"].values[idx] >= THR).astype(int)   # subset to the 1000, then binarize
    rows.append((div, seed, balanced_accuracy_score(y_true, y_pred)))

res = pd.DataFrame(rows, columns=["divisor", "seed", "bal_acc"])

# --- mean balanced accuracy per divisor (averaged over the 6 seeds) ---
summary = res.groupby("divisor")["bal_acc"].agg(["mean", "std"]).sort_index()

print("=== ConvNeXt-Base balanced accuracy on the 1000 VLM galaxies ===")
print(f"(threshold={THR}, {len(idx)} galaxies)\n")
print("Mean balanced accuracy per divisor (avg of 6 seeds):")
print(summary.round(4).to_string())

print("\nPer (divisor, seed):")
print(res.pivot(index="divisor", columns="seed", values="bal_acc")[[0,1,2,3,4,42]].round(4).to_string())
