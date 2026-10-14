"""Evaluate a fine-tuned (or zero-shot) Qwen3.5 VLM on the GZ Euclid 13-question tree.
Prompts for one 'question: answer' line per question, parses them, and scores PATH ACCURACY
on the fixed 1000-galaxy sample using the SAME path_scores.py as the CNN/Gemini (comparable)."""
import os, sys, csv, re
import numpy as np, pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euclid_VLMs", "shared"))
from euclid_schema import QUESTIONS, load_bundle_seed
from path_scores import build_truth_paths, score_model

MODEL   = os.environ["MODEL_LOCAL_PATH"]
ADAPTER = os.environ.get("QWEN_ADAPTER_PATH")
EVAL_CSV = os.environ.get("EVAL_CSV", "/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm/eval_1000.csv")
OUT     = os.environ["OUT"]

PROMPT = open("/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm/prompt.txt").read()

proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
if ADAPTER:
    model.load_adapter(ADAPTER); print("ADAPTER:", ADAPTER, flush=True)
else:
    print("zero-shot (no adapter)", flush=True)
model.eval()

def parse(text):
    out = {}
    for q, allowed in QUESTIONS.items():
        m = re.search(rf"{re.escape(q)}\s*:\s*([a-zA-Z0-9\-]+)", text, re.I)
        a = m.group(1).strip().lower() if m else None
        out[q] = a if a in allowed else "INVALID"    # missing or not an allowed answer -> scored wrong
    return out

rows = list(csv.DictReader(open(EVAL_CSV)))
per_gal = {}
raw = {}
for i, r in enumerate(rows):
    img = Image.open(r["file_loc"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)
    n = inp.input_ids.shape[1]
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=220, do_sample=False)
    txt = proc.batch_decode(g[:, n:], skip_special_tokens=True)[0]
    per_gal[r["id_str"]] = parse(txt)
    raw[r["id_str"]] = txt
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

truth = load_bundle_seed().set_index("id_str")
sample_ids = [r["id_str"] for r in rows if r["id_str"] in truth.index]
truth = truth.loc[sample_ids]
truth_paths = build_truth_paths(truth)
pred_answers = {q: [per_gal[i][q] for i in truth_paths.index] for q in QUESTIONS}
score, per_galaxy = score_model(truth_paths, pred_answers)

pd.DataFrame([{"id_str": i, "raw_response": raw[i], **per_gal[i]} for i in sample_ids]).to_csv(OUT, index=False)
print(f"\n=== euclid VLM {'fine-tuned' if ADAPTER else 'zero-shot'} ===", flush=True)
print(f"PATH ACCURACY = {np.mean(per_galaxy):.4f}  on {len(sample_ids)} galaxies", flush=True)
n_inv = sum(per_gal[i][q] == "INVALID" for i in sample_ids for q in QUESTIONS)
n_inv_path = sum(per_gal[i][q] == "INVALID" for i in truth_paths.index for q in QUESTIONS if truth_paths.loc[i, q] != "")
print(f"INVALID answers: {n_inv} of {len(sample_ids) * len(QUESTIONS)} total; "
      f"{n_inv_path} on the volunteers' path", flush=True)
print(f"saved -> {OUT}", flush=True)
