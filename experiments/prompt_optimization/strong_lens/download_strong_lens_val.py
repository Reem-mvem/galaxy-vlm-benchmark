
import os
import numpy as np
import pandas as pd
from datasets import load_dataset

HF_DATASET  = "mwalmsley/euclid_strong_lens_expert_judges"
HF_CONFIG   = "classification"
IMAGES_DIR  = os.path.join("images", "strong_lenses", "val")
CATALOG_CSV = "strong_lens_val_catalog.csv"
N_VAL       = 300
SEED        = 42

# Match test set ratio exactly: 104 lens / 1476 total = 7.05%
TEST_LENS     = 104
TEST_TOTAL    = 1476
N_LENS    = round(N_VAL * TEST_LENS / TEST_TOTAL)   # 21
N_NOT_LENS = N_VAL - N_LENS                          # 279

os.makedirs(IMAGES_DIR, exist_ok=True)

print(f"Loading {HF_DATASET} train split...")
ds = load_dataset(HF_DATASET, HF_CONFIG, split="train")
print(f"Train set: {len(ds)} images")

label_names = ds.features["label"].names
print(f"Label names: {label_names}")

df = ds.to_pandas()[["id_str", "label"]]
df["label_name"] = df["label"].apply(lambda x: label_names[x])

print(f"\nFull train distribution:")
print(df["label_name"].value_counts())

# Sample to match test ratio
rng = np.random.default_rng(SEED)
lens_pool     = df[df["label_name"] == "grade_a_or_b_lens"]
not_lens_pool = df[df["label_name"] == "not_grade_a_or_b_lens"]

lens_idx     = rng.choice(len(lens_pool),     size=min(N_LENS,     len(lens_pool)),     replace=False)
not_lens_idx = rng.choice(len(not_lens_pool), size=min(N_NOT_LENS, len(not_lens_pool)), replace=False)

val_df = pd.concat([
    lens_pool.iloc[lens_idx],
    not_lens_pool.iloc[not_lens_idx],
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nVal set ({len(val_df)} samples, matching test ratio):")
print(val_df["label_name"].value_counts())
print(f"Positive rate: {val_df['label_name'].eq('grade_a_or_b_lens').mean():.1%}")

# Download images
val_id_set = set(val_df["id_str"])
rows  = []
saved = 0

print(f"\nDownloading {len(val_df)} images...")
for i, example in enumerate(ds):
    if example["id_str"] not in val_id_set:
        continue

    img_path = os.path.join(IMAGES_DIR, f"{example['id_str']}.jpg")
    if not os.path.exists(img_path):
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path)

    label_idx = int(example["label"])
    rows.append({
        "id_str":     example["id_str"],
        "file_loc":   img_path,
        "label":      label_idx,
        "label_name": label_names[label_idx],
    })
    saved += 1
    if saved % 50 == 0:
        print(f"  {saved}/{len(val_df)}")
    if saved == len(val_df):
        break

result = pd.DataFrame(rows)
result.to_csv(CATALOG_CSV, index=False)
print(f"\nSaved {len(result)} samples → {CATALOG_CSV}")
print(f"Images → {IMAGES_DIR}/")
