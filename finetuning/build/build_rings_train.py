"""Build the rings TRAINING pool from gz_rings (classification config, RAW images
— matching the test catalog + task.py). Downloads only the train parquet shards.

Produces, under tasks/rings/data/:
  - train_catalog_full.csv  : all train rows (id_str,label_name) — labels only, no images
  - train_pool_4000.csv     : balanced 2000 ring / 2000 not_ring, images saved
  - train_{100,250,500,1000,2000}.csv : nested balanced subsets of the pool (sweep points)

Nested = each smaller set is the head of the next, so the label-efficiency curve is clean.
"""
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

R = Path("/home/reem1/scratch/galaxy-vlm-benchmark/tasks/rings")
HF, CONFIG = "mwalmsley/gz_rings", "classification"
TRAIN_IMG = R / "data" / "images" / "train"
TRAIN_IMG.mkdir(parents=True, exist_ok=True)
POOL_PER_CLASS = 2000
SWEEP = [100, 250, 500, 1000, 2000]      # total sizes (balanced -> half per class)
RNG = 42

print("loading gz_rings classification TRAIN shards...", flush=True)
ds = load_dataset(HF, data_files={"train": f"{CONFIG}/train-*.parquet"}, split="train")
names = ds.features["label"].names
print(f"train set: {len(ds)}  classes: {names}", flush=True)

# labels + ids via column access (no image decode)
labels = ds["label"]
ids = ds["id_str"] if "id_str" in ds.column_names else [f"train_{i}" for i in range(len(ds))]
full = pd.DataFrame({"row": range(len(ds)), "id_str": ids, "label": labels})
full["label_name"] = full["label"].map({i: n for i, n in enumerate(names)})
full[["id_str", "label_name"]].to_csv(R / "data" / "train_catalog_full.csv", index=False)
print("full train balance:", dict(full.label_name.value_counts()), flush=True)

# balanced pool: sample POOL_PER_CLASS of each class
parts = []
for cls in names:
    sub = full[full.label_name == cls]
    n = min(POOL_PER_CLASS, len(sub))
    parts.append(sub.sample(n=n, random_state=RNG))
    if n < POOL_PER_CLASS:
        print(f"  WARN: only {n} '{cls}' available (< {POOL_PER_CLASS})", flush=True)
pool = pd.concat(parts).sample(frac=1, random_state=RNG).reset_index(drop=True)

# decode+save ONLY the pool images, one pass
sel = ds.select(pool.row.tolist())
rows = []
for k, (r, ex) in enumerate(zip(pool.itertuples(index=False), sel)):
    p = TRAIN_IMG / f"{r.id_str}.jpg"
    if not p.exists():
        img = ex["image"]
        img = img.convert("RGB") if img.mode != "RGB" else img
        img.save(p)
    rows.append({"id_str": r.id_str, "file_loc": str(p), "label_name": r.label_name})
    if (k + 1) % 400 == 0:
        print(f"  saved {k+1}/{len(pool)}", flush=True)
pool_df = pd.DataFrame(rows)
pool_df.to_csv(R / "data" / f"train_pool_{len(pool_df)}.csv", index=False)
print(f"POOL: {len(pool_df)} images ({dict(pool_df.label_name.value_counts())})", flush=True)

# nested balanced sweep subsets (head of the pool, per class -> nested)
pos = pool_df[pool_df.label_name == "ring"].reset_index(drop=True)
neg = pool_df[pool_df.label_name == "not_ring"].reset_index(drop=True)
for total in SWEEP:
    half = total // 2
    if half > len(pos) or half > len(neg):
        print(f"  skip {total}: not enough per class", flush=True); continue
    sub = pd.concat([pos.head(half), neg.head(half)]).sample(frac=1, random_state=RNG)
    sub.to_csv(R / "data" / f"train_{total}.csv", index=False)
    print(f"  train_{total}.csv: {dict(sub.label_name.value_counts())}", flush=True)
print("RINGS_TRAIN_BUILD_DONE", flush=True)
