import os
for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE"): os.environ.pop(k,None)
os.environ["HF_HOME"]="/scratch/reem1/rings_runs/hf_cache"
from datasets import load_dataset
from pathlib import Path
import pandas as pd, shutil

BASE = Path("/home/reem1/scratch/galaxy-vlm-benchmark/tasks/streams_shells_neither")
print("loading astronolan/streams_shells_neither ...", flush=True)
dd = load_dataset("astronolan/streams_shells_neither")
# label may be a ClassLabel (has .names) OR a plain Value (string / int)
_lf = dd["train"].features["label"]
_names = getattr(_lf, "names", None)
def to_name(v):
    return _names[v] if _names is not None else str(v)
print("splits:", {k: len(v) for k,v in dd.items()}, "| label:", _lf, "| names:", _names, flush=True)
print("columns:", dd["train"].column_names, flush=True)

for split in dd:
    imgdir = BASE/"data"/"images"/split
    imgdir.mkdir(parents=True, exist_ok=True)
    ds = dd[split]; rows=[]
    for i in range(len(ds)):
        ex = ds[i]
        idstr = f"{split}_{i}"
        p = imgdir/f"{idstr}.jpg"
        img = ex["image_rgb"]
        img = img.convert("RGB") if img.mode != "RGB" else img
        img.save(p)
        rows.append({"id_str": idstr, "file_loc": str(p),
                     "label_name": to_name(ex["label"]),
                     "ra": ex.get("ra"), "dec": ex.get("dec")})
    df = pd.DataFrame(rows)
    df.to_csv(BASE/"data"/f"{split}_catalog.csv", index=False)
    print(f"{split}: {len(df)} imgs -> {dict(df.label_name.value_counts())}", flush=True)

shutil.copy(BASE/"data"/"train_catalog.csv", BASE/"data"/"train_72.csv")
print("BUILD_DONE", flush=True)
