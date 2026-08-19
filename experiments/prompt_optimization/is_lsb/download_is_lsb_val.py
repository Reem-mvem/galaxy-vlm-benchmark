"""
Sample 300 val images from is-lsb TRAIN split, matching the true test set class ratio.
Saves is_lsb_val_catalog.csv with columns: id_str, file_loc, label, label_name
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset

IMAGES_DIR = os.path.join("images", "is_lsb", "val")
OUT_CSV    = "is_lsb_val_catalog.csv"
N_VAL      = 300
SEED       = 42
CLASS_NAMES = ["not_tidal", "tidal"]   # index 0 / 1

os.makedirs(IMAGES_DIR, exist_ok=True)

# ── get test set ratio ────────────────────────────────────────────────────────
print("Loading test split to compute class ratio...")
test_ds    = load_dataset("mwalmsley/is-lsb", split="test")
test_labels = test_ds["label"]
n_test      = len(test_labels)
n_tidal     = sum(test_labels)
tidal_rate  = n_tidal / n_test
print(f"Test set: {n_test} total  |  {n_tidal} tidal ({tidal_rate:.2%})  |  {n_test-n_tidal} not_tidal ({1-tidal_rate:.2%})")

N_TIDAL     = round(N_VAL * tidal_rate)
N_NOT_TIDAL = N_VAL - N_TIDAL
print(f"\nVal target: {N_TIDAL} tidal + {N_NOT_TIDAL} not_tidal = {N_VAL} total")

# ── sample from train split ───────────────────────────────────────────────────
print("\nLoading train split...")
train_ds = load_dataset("mwalmsley/is-lsb", split="train")
print(f"Train set: {len(train_ds)} images")

train_labels = train_ds["label"]
tidal_idx     = [i for i, l in enumerate(train_labels) if l == 1]
not_tidal_idx = [i for i, l in enumerate(train_labels) if l == 0]
print(f"  tidal: {len(tidal_idx)}  |  not_tidal: {len(not_tidal_idx)}")

rng           = np.random.default_rng(SEED)
chosen_tidal     = rng.choice(tidal_idx,     size=min(N_TIDAL,     len(tidal_idx)),     replace=False)
chosen_not_tidal = rng.choice(not_tidal_idx, size=min(N_NOT_TIDAL, len(not_tidal_idx)), replace=False)
chosen_indices   = np.concatenate([chosen_tidal, chosen_not_tidal])
rng.shuffle(chosen_indices)

# ── download images ───────────────────────────────────────────────────────────
print(f"\nDownloading {len(chosen_indices)} images...")
rows = []
for i, idx in enumerate(chosen_indices):
    example    = train_ds[int(idx)]
    id_str     = example["id_str"]
    label_idx  = int(example["label"])
    label_name = CLASS_NAMES[label_idx]
    img_path   = os.path.join(IMAGES_DIR, f"{id_str}.jpg")

    if not os.path.exists(img_path):
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path)

    rows.append({
        "id_str":     id_str,
        "file_loc":   img_path,
        "label":      label_idx,
        "label_name": label_name,
    })

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(chosen_indices)}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows → {OUT_CSV}")
print(df["label_name"].value_counts().to_string())
