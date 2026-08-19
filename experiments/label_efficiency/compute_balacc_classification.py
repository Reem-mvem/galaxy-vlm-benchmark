"""
Balanced accuracy vs number of training images for the three classification tasks:
is-lsb, which-lsb, strong_lens. Sweep points from our divisor runs + the 100% (div1)
anchor from the bundle (mwalmsley best models).

  * true label: joined from the HF dataset by id_str (authoritative). Bundle rows
    carry their own label column, used directly for the 100% anchor.
  * predicted class: argmax over the class_* columns.
  * balanced accuracy: sklearn balanced_accuracy_score (mean per-class recall),
    the right metric here because all three test sets are class-imbalanced.
  * eval sample: is-lsb / which-lsb use the FULL test set (<1000). strong_lens uses
    the fixed 1000-galaxy id list in strong_lens_pred.csv.
"""
import glob, re, os
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score

DL = "/scratch/reem1/rings_runs/results/downstream"
BUNDLE = "/home/reem1/scratch/geir_files/gz_evo_downstream_predictions_bundle_2026-07-02/gz_evo_downstream_predictions_bundle_2026-07-02/predictions"
OUT = "/scratch/reem1/rings_runs/analysis"
os.makedirs(OUT, exist_ok=True)

TASKS = {
    "is_lsb":      dict(hf=("mwalmsley/is-lsb", None),   train=2013, sweep_glob=f"{DL}/is_lsb/*",
                        bundle="is_debris_maxvit_tiny",  bundle_id="hf_id_str", bundle_label="hf_label",
                        sample=None),
    "which_lsb":   dict(hf=("mwalmsley/which-lsb", None), train=508,  sweep_glob=f"{DL}/which_lsb/*",
                        bundle="which_debris_maxvit_tiny", bundle_id="id_str", bundle_label="label",
                        sample=None),
    "strong_lens": dict(hf=("mwalmsley/euclid_strong_lens_expert_judges", "classification"), train=5876,
                        sweep_glob=f"{DL}/strong_lens/*", bundle="strong_lenses_sovit400",
                        bundle_id="id_str", bundle_label="label",
                        sample="/home/reem1/scratch/rings_runs/strong_lens_pred.csv"),
}

def argmax_class(df):
    cols = sorted([c for c in df.columns if re.fullmatch(r"class_\d+", c)],
                  key=lambda c: int(c.split("_")[1]))
    return df[cols].values.argmax(1)

rows = []
for task, cfg in TASKS.items():
    print(f"\n=== {task} ===", flush=True)
    name, sub = cfg["hf"]
    ds = load_dataset(name, sub) if sub else load_dataset(name)
    truth = ds["test"].remove_columns([c for c in ds["test"].column_names if c == "image"]).to_pandas()
    id2y = dict(zip(truth["id_str"], truth["label"]))

    sample_ids = None
    if cfg["sample"]:
        sample_ids = set(pd.read_csv(cfg["sample"])["id_str"])
    print(f"  truth rows={len(id2y)} | eval sample={'full' if sample_ids is None else len(sample_ids)}", flush=True)

    # --- sweep points ---
    for f in glob.glob(f"{cfg['sweep_glob']}/div*/seed*_*/predictions_test.csv"):
        div  = int(re.search(r"/div(\d+)/", f).group(1))
        seed = int(re.search(r"/seed(\d+)_", f).group(1))
        p = pd.read_csv(f)
        if sample_ids is not None:
            p = p[p["id_str"].isin(sample_ids)]
        y_true = p["id_str"].map(id2y).values
        y_pred = argmax_class(p)
        rows.append((task, "sweep", cfg["train"] // div, div, seed,
                     balanced_accuracy_score(y_true, y_pred)))

    # --- 100% anchor from bundle (uses bundle's own label column) ---
    for f in glob.glob(f"{BUNDLE}/{cfg['bundle']}/seed_*/predictions_test.csv"):
        seed = int(re.search(r"seed_(\d+)/", f).group(1))
        p = pd.read_csv(f)
        if sample_ids is not None:
            p = p[p[cfg["bundle_id"]].isin(sample_ids)]
        y_true = p[cfg["bundle_label"]].values
        y_pred = argmax_class(p)
        rows.append((task, "bundle_100pct", cfg["train"], 1, seed,
                     balanced_accuracy_score(y_true, y_pred)))

res = pd.DataFrame(rows, columns=["task", "source", "n_images", "divisor", "seed", "bal_acc"])
res.to_csv(f"{OUT}/balacc_classification.csv", index=False)

print("\n=== mean balanced accuracy per (task, n_images) ===")
summ = res.groupby(["task", "n_images"])["bal_acc"].agg(["mean", "std", "count"]).round(4)
print(summ.to_string())
print(f"\nsaved -> {OUT}/balacc_classification.csv")
