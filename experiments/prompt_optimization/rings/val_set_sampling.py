"""
Sample 300 val images from gz_rings test set, matching the true class ratio.
Saves rings_val_catalog.csv with columns: id_str, file_loc, label, label_name
"""

import os
import numpy as np
import pandas as pd

SCRATCH    = "/geir_data/scr/reem"
CNN_CSV    = f"{SCRATCH}/zoobot_project/gz_rings_cnn_predictions.csv"
IMAGES_DIR = f"{SCRATCH}/gz_rings/images/test"
OUT_CSV    = f"{SCRATCH}/po/rings/rings_val_catalog.csv"

THRESHOLD  = 0.5
N_VAL      = 300
SEED       = 42

df = pd.read_csv(CNN_CSV)
df["file_loc"]   = df["id_str"].apply(lambda s: os.path.join(IMAGES_DIR, f"{s.split('_')[1]}.jpg"))
df["label"]      = (df["ring_fraction"] >= THRESHOLD).astype(int)
df["label_name"] = df["label"].map({1: "ring", 0: "not_ring"})

n_total   = len(df)
n_ring    = df["label"].sum()
ring_rate = n_ring / n_total
print(f"Test set: {n_total} total  |  {n_ring} ring ({ring_rate:.2%})  |  {n_total-n_ring} not_ring ({1-ring_rate:.2%})")

N_RING     = round(N_VAL * ring_rate)
N_NOT_RING = N_VAL - N_RING
print(f"\nVal target: {N_RING} ring + {N_NOT_RING} not_ring = {N_VAL} total")

rng = np.random.default_rng(SEED)

ring_pool    = df[df["label"] == 1]
not_ring_pool = df[df["label"] == 0]

ring_idx     = rng.choice(len(ring_pool),     size=min(N_RING,     len(ring_pool)),     replace=False)
not_ring_idx = rng.choice(len(not_ring_pool), size=min(N_NOT_RING, len(not_ring_pool)), replace=False)

val = pd.concat([ring_pool.iloc[ring_idx], not_ring_pool.iloc[not_ring_idx]])
val = val.sample(frac=1, random_state=SEED).reset_index(drop=True)

val[["id_str", "file_loc", "label", "label_name"]].to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(val)} rows → {OUT_CSV}")
print(val["label_name"].value_counts().to_string())
