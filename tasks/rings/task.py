"""rings — binary ring detection on Galaxy Zoo Rings (gz_rings).

Ground truth comes from HuggingFace labels

Run build_catalog() once to download test images and save the catalog CSV.
"""

import os
import re
from pathlib import Path

import pandas as pd

from vlm_eval import TaskSpec
from vlm_eval.images import read_image_bytes
from vlm_eval.sampling import sample_balanced

TASK_DIR   = Path(__file__).resolve().parent
IMAGES_DIR = Path(os.getenv("RINGS_IMAGES_DIR", TASK_DIR / "data" / "images" / "test"))
CATALOG    = Path(os.getenv("RINGS_CATALOG", TASK_DIR / "gz_rings_hf_catalog.csv"))
HF_DATASET = "mwalmsley/gz_rings"
HF_CONFIG  = "classification"

CLASS_NAMES = ["not_ring", "ring"]

PROMPT = (
    "Does this galaxy image show a ring (a circular feature within or around "
    "the galaxy)? Reply with one word: yes or no."
)


def load_catalog():
    if not CATALOG.exists():
        raise FileNotFoundError(CATALOG)
    df = pd.read_csv(CATALOG)
    print(f"Loaded {len(df)} test examples "
          f"(ring: {(df['label_name'] == 'ring').mean():.1%})")
    return df[["id_str", "file_loc", "label_name"]]


def parse_prediction(text):
    text = text.strip().lower()
    if re.search(r"\byes\b", text):
        return "ring"
    if re.search(r"\bno\b", text):
        return "not_ring"
    return None


def build_catalog():
    """Download gz_rings test images from HuggingFace and save catalog CSV.

    Only fetches the two test parquet shards — skips the large train shards.
    Idempotent: existing images are skipped.
    """
    from datasets import load_dataset

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {HF_DATASET} test split (test shards only)...")
    ds = load_dataset(
        HF_DATASET,
        data_files={"test": f"{HF_CONFIG}/test-*.parquet"},
        split="test",
        token=os.getenv("HF_TOKEN"),
    )
    label_names = ds.features["label"].names
    print(f"Test set: {len(ds)} images  |  classes: {label_names}")

    rows = []
    saved = skipped = 0
    for i, example in enumerate(ds):
        id_str   = example.get("id_str", f"test_{i}")
        img_path = IMAGES_DIR / f"{id_str}.jpg"
        if img_path.exists():
            skipped += 1
        else:
            img = example["image"].convert("RGB")
            img.save(img_path)
            saved += 1
        label_idx  = int(example["label"])
        label_name = label_names[label_idx]
        rows.append({"id_str": id_str, "file_loc": str(img_path), "label_name": label_name})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(ds)} (saved {saved}, skipped {skipped})")

    df = pd.DataFrame(rows)
    df.to_csv(CATALOG, index=False)
    print(f"Done: {saved} saved, {skipped} skipped")
    print(f"Catalog → {CATALOG}")
    print(df["label_name"].value_counts().to_string())
    return df


TASK = TaskSpec(
    name="rings",
    label="GZ Rings",
    class_names=CLASS_NAMES,
    prompt=PROMPT,
    dir=TASK_DIR,
    load_catalog=load_catalog,
    build_catalog=build_catalog,
    parse_prediction=parse_prediction,
    get_sample=sample_balanced,
    prepare_image=read_image_bytes,
)
