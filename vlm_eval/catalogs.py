"""Build and load test catalogs from HuggingFace datasets.

A catalog is a CSV with columns: id_str, file_loc, label, label_name.
Images are saved to disk once (under the task's data/ folder, gitignored)
so evaluation runs never re-download.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from datasets import load_dataset


def build_hf_catalog(
    hf_dataset: str,
    images_dir: Path,
    catalog_path: Path,
    class_names: Optional[list] = None,
    hf_config: Optional[str] = None,
    split: str = "test",
    id_fn: Optional[Callable] = None,        # (example, i) -> id string
    filename_fn: Optional[Callable] = None,  # (example, i) -> image filename
) -> pd.DataFrame:
    """Download a HF split, save images to images_dir, write the catalog CSV."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {hf_dataset}{f' ({hf_config})' if hf_config else ''} from HuggingFace...")
    ds = load_dataset(hf_dataset, hf_config, split=split) if hf_config else \
         load_dataset(hf_dataset, split=split)
    print(f"{split} set: {len(ds)} images")

    if class_names is None:
        label_feature = ds.features["label"]
        class_names = label_feature.names if hasattr(label_feature, "names") else None

    if id_fn is None:
        id_fn = lambda ex, i: ex.get("id_str", f"{split}_{i}")
    if filename_fn is None:
        filename_fn = lambda ex, i: f"{id_fn(ex, i)}.jpg"

    rows = []
    for i, example in enumerate(ds):
        img_path = images_dir / filename_fn(example, i)
        if not img_path.exists():
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path)
        label_idx = int(example["label"])
        label_name = class_names[label_idx] if class_names else str(label_idx)
        rows.append({"id_str": id_fn(example, i), "file_loc": str(img_path),
                     "label": label_idx, "label_name": label_name})
        if (i + 1) % 200 == 0:
            print(f"  Downloaded {i + 1}/{len(ds)}")

    df = pd.DataFrame(rows)
    Path(catalog_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(catalog_path, index=False)
    print(f"Catalog saved -> {catalog_path}")
    print(df["label_name"].value_counts().to_string())
    return df


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"{catalog_path} not found — run with --build_catalog_only first "
            f"(or let the eval build it on first run)."
        )
    df = pd.read_csv(catalog_path)
    # Make image paths portable across machines. Catalogs may store absolute paths from
    # wherever they were built (e.g. the cluster). If a stored file_loc isn't found here,
    # rebase its ".../data/..." tail onto THIS catalog's task dir (works locally after a sync;
    # no-op on the machine that built it, where the absolute path already exists).
    if "file_loc" in df.columns:
        task_dir = Path(catalog_path).resolve().parent.parent
        def _resolve(fl):
            s = str(fl).replace("\\", "/")
            if os.path.exists(s):
                return s
            if "/data/" in s:
                cand = task_dir / ("data/" + s.split("/data/", 1)[1])
                if cand.exists():
                    return str(cand)
            return s
        df["file_loc"] = df["file_loc"].map(_resolve)
    print(f"Loaded {len(df)} test examples")
    print(df["label_name"].value_counts().to_string())
    return df
