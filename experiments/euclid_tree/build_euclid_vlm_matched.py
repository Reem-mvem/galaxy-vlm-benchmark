"""Build VLM training data on the EXACT 99 galaxies the CNN (seed0) trained on
(cnn_seed0_ids.txt = its 79 train + 20 val labels). Same 13-line 'question: answer'
targets as the random build. -> train_cnn99.csv (matched to the CNN's label budget)."""
import os, sys
os.environ["HF_HOME"] = "/scratch/reem1/rings_runs/hf_cache"
os.environ.pop("HF_DATASETS_OFFLINE", None); os.environ.pop("HF_HUB_OFFLINE", None)
import numpy as np, pandas as pd
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, "/home/reem1/scratch/rings_runs/analysis/euclid")
from euclid_schema import QUESTIONS, fraction_cols

BASE = Path("/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm")
(BASE / "images/train_cnn99").mkdir(parents=True, exist_ok=True)
WANT = [l.strip() for l in open(BASE / "cnn_seed0_ids.txt") if l.strip()]
want_set = set(WANT)
print(f"{len(WANT)} target ids (CNN seed0 label budget)", flush=True)

def target_text(ex):
    lines = []
    for q, answers in QUESTIONS.items():
        fr = [ex[c] for c in fraction_cols(q)]
        lines.append(f"{q}: {answers[int(np.argmax(fr))]}")
    return "\n".join(lines)

print("loading mwalmsley/gz_euclid ...", flush=True)
dd = load_dataset("mwalmsley/gz_euclid")
tr = dd["train"]
ids = tr["id_str"]                                   # column load, no image decode
pos = {idv: i for i, idv in enumerate(ids)}
missing = [i for i in WANT if i not in pos]
print(f"found {len(WANT)-len(missing)}/{len(WANT)} in train split; missing={len(missing)}", flush=True)

rows = []
for k, idstr in enumerate(WANT):
    if idstr not in pos:
        continue
    ex = tr[pos[idstr]]
    p = BASE / "images/train_cnn99" / f"{idstr}.jpg"
    img = ex["image"]; img = img.convert("RGB") if img.mode != "RGB" else img; img.save(p)
    rows.append({"id_str": idstr, "file_loc": str(p), "label_name": target_text(ex)})
    if (k + 1) % 25 == 0: print(f"  {k+1}/{len(WANT)}", flush=True)
pd.DataFrame(rows).to_csv(BASE / "train_cnn99.csv", index=False)
print(f"train_cnn99.csv written ({len(rows)})", flush=True)
print("BUILD_DONE", flush=True)
