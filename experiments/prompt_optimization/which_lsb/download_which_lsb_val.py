"""
Sample 300 val images from which-lsb TRAIN split, matching the true test set class ratio.
Saves which_lsb_val_catalog.csv with columns: id_str, file_loc, label, label_name
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset

IMAGES_DIR  = os.path.join("images", "which_lsb", "val")
OUT_CSV     = "which_lsb_val_catalog.csv"
N_VAL       = 300
SEED        = 42
CLASS_NAMES = ["arm", "stream", "shell", "diffuse"]   # indices 0-3

os.makedirs(IMAGES_DIR, exist_ok=True)

# ── get test set ratio ────────────────────────────────────────────────────────
print("Loading test split to compute class ratio...")
test_ds     = load_dataset("mwalmsley/which-lsb", split="test")
test_labels = test_ds["label"]
n_test      = len(test_labels)
print(f"Test set: {n_test} total")
for i, name in enumerate(CLASS_NAMES):
    n = sum(1 for l in test_labels if l == i)
    print(f"  {name}: {n} ({n/n_test:.2%})")

# target counts per class
n_per_class = [round(N_VAL * sum(1 for l in test_labels if l == i) / n_test)
               for i in range(len(CLASS_NAMES))]
# fix rounding to ensure exactly N_VAL total
diff = N_VAL - sum(n_per_class)
n_per_class[0] += diff
print(f"\nVal target: {dict(zip(CLASS_NAMES, n_per_class))} = {sum(n_per_class)} total")

# ── sample from train split ───────────────────────────────────────────────────
print("\nLoading train split...")
train_ds     = load_dataset("mwalmsley/which-lsb", split="train")
train_labels = train_ds["label"]
print(f"Train set: {len(train_ds)} images")

rng            = np.random.default_rng(SEED)
chosen_indices = []
for i, name in enumerate(CLASS_NAMES):
    pool = [j for j, l in enumerate(train_labels) if l == i]
    n    = min(n_per_class[i], len(pool))
    chosen = rng.choice(pool, size=n, replace=False)
    chosen_indices.extend(chosen.tolist())
    print(f"  {name}: {len(pool)} available → sampling {n}")

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
