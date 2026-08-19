"""Build euclid VLM fine-tune data.
 train_100.csv : 100 sampled train galaxies -> (id, image, target = 13 'question: answer' lines,
                 answer = argmax of the volunteer vote fractions per question).
 eval_1000.csv : the fixed 1000-galaxy sample (euclid_sample_ids.txt) with images, for scoring
                 path accuracy against the bundle truth (same metric as the CNN/Gemini)."""
import os, sys, random
os.environ["HF_HOME"] = "/scratch/reem1/rings_runs/hf_cache"
os.environ.pop("HF_DATASETS_OFFLINE", None); os.environ.pop("HF_HUB_OFFLINE", None)
import numpy as np, pandas as pd
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, "/home/reem1/scratch/rings_runs/analysis/euclid")
from euclid_schema import QUESTIONS, fraction_cols

BASE = Path("/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm")
(BASE / "images/train").mkdir(parents=True, exist_ok=True)
(BASE / "images/eval").mkdir(parents=True, exist_ok=True)
SAMPLE_IDS = set(l.strip() for l in open("/home/reem1/scratch/rings_runs/analysis/euclid/euclid_sample_ids.txt") if l.strip())
print(f"{len(SAMPLE_IDS)} eval sample ids", flush=True)

def target_text(ex):
    lines = []
    for q, answers in QUESTIONS.items():
        fr = [ex[c] for c in fraction_cols(q)]
        lines.append(f"{q}: {answers[int(np.argmax(fr))]}")
    return "\n".join(lines)

print("loading mwalmsley/gz_euclid ...", flush=True)
dd = load_dataset("mwalmsley/gz_euclid")
tr = dd["train"]
idx = list(range(len(tr))); random.Random(42).shuffle(idx)
rows = []
for k, i in enumerate(idx[:100]):
    ex = tr[i]
    idstr = str(ex.get("id_str", f"train_{i}"))
    p = BASE / "images/train" / f"{idstr}.jpg"
    img = ex["image"]; img = img.convert("RGB") if img.mode != "RGB" else img; img.save(p)
    rows.append({"id_str": idstr, "file_loc": str(p), "label_name": target_text(ex)})
    if (k + 1) % 25 == 0: print(f"  train {k+1}/100", flush=True)
pd.DataFrame(rows).to_csv(BASE / "train_100.csv", index=False)
print(f"train_100.csv written ({len(rows)})", flush=True)

# eval images: the 1000 sample from the test split
te = dd["test"]; erows = []
for i in range(len(te)):
    ex = te[i]
    idstr = str(ex.get("id_str", f"test_{i}"))
    if idstr not in SAMPLE_IDS: continue
    p = BASE / "images/eval" / f"{idstr}.jpg"
    img = ex["image"]; img = img.convert("RGB") if img.mode != "RGB" else img; img.save(p)
    erows.append({"id_str": idstr, "file_loc": str(p)})
    if len(erows) % 200 == 0: print(f"  eval {len(erows)}/{len(SAMPLE_IDS)}", flush=True)
pd.DataFrame(erows).to_csv(BASE / "eval_1000.csv", index=False)
print(f"eval_1000.csv written ({len(erows)} of {len(SAMPLE_IDS)} sample ids found)", flush=True)
print("BUILD_DONE", flush=True)
