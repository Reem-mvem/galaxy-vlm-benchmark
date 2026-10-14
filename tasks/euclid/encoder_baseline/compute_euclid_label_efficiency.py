"""
Euclid path accuracy vs number of training images, for the label-efficiency plot.
Reuses the exact truth-path construction and scoring from path_scores.py, so the
metric is identical to the headline number. CNN answers come from OUR divisor-sweep
predictions (concentration columns argmax'd); truth comes from the bundle volunteer
fractions on the fixed 1000-galaxy sample.

Validation: the 100% (bundle maxvit_base) score must reproduce 89.4% path accuracy.
"""
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "euclid_VLMs" / "shared"
REPO = HERE.parents[2]
sys.path.insert(0, str(SHARED))
from euclid_schema import QUESTIONS, BUNDLE_PREDICTIONS, cnn_answers, load_bundle_seed
from path_scores import build_truth_paths, score_model

MODEL = "gz_euclid_maxvit_base"
TRAIN_TOTAL = 169899
SWEEP_GLOB = str(REPO / "finetuning/encoders/results/gz_euclid") + "/*/div{d}/seed{s}_*/predictions_test.csv"

sample_ids = [l.strip() for l in open(SHARED / "euclid_sample_ids.txt") if l.strip()]
truth_df = load_bundle_seed(model=MODEL).set_index("id_str").loc[sample_ids]
truth_paths = build_truth_paths(truth_df)
confident = truth_paths["fully_confident"].values
print(f"truth: {len(truth_paths)} galaxies, {confident.sum()} fully-confident")

def score_file(path):
    df = pd.read_csv(path).set_index("id_str").loc[truth_paths.index]
    preds = {q: cnn_answers(df, q) for q in QUESTIONS}
    _, per_galaxy = score_model(truth_paths, preds)
    return per_galaxy.mean(), per_galaxy[confident].mean()

rows = []
# sweep points (~100-img fill div1699 + low-data fill 340..17 + original 7,4,2)
for div in (1699, 340, 85, 34, 17, 7, 4, 2):
    for seed in (0, 1, 2, 3, 4, 42):
        fs = glob.glob(SWEEP_GLOB.format(d=div, s=seed))
        if not fs:
            print(f"  MISSING div{div} seed{seed}"); continue
        acc_all, acc_conf = score_file(fs[0])
        rows.append(("sweep", TRAIN_TOTAL // div, div, seed, acc_all, acc_conf))

# 100% anchor from bundle (6 seeds) — also the validation of 89.4%
for sd in sorted(p for p in (BUNDLE_PREDICTIONS / MODEL).iterdir() if p.is_dir()):
    seed = int(re.search(r"seed_(\d+)", sd.name).group(1))
    acc_all, acc_conf = score_file(sd / "predictions_test.csv")
    rows.append(("bundle_100pct", TRAIN_TOTAL, 1, seed, acc_all, acc_conf))

res = pd.DataFrame(rows, columns=["source", "n_images", "divisor", "seed", "path_acc", "path_acc_confident"])
res.to_csv(HERE / "euclid_path_accuracy.csv", index=False)

print("\n=== path accuracy (all galaxies) per n_images ===")
print(res.groupby("n_images")["path_acc"].agg(["mean", "std", "count"]).round(4).to_string())
b = res[res.source == "bundle_100pct"]["path_acc"].mean()
print(f"\nVALIDATION: bundle 100% path accuracy = {b:.1%}  (expected 89.4%)")
print(f"saved -> {HERE/'euclid_path_accuracy.csv'}")
