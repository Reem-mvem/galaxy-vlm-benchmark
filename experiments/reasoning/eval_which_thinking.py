"""Does 'thinking' help a pure-vision task? Qwen3.5-4B on which_lsb (4-class), thinking
ON vs OFF, zero-shot or with a fine-tuned adapter. Saves the FULL reasoning trace per image.

Writes INCREMENTALLY (one row per image, flushed) and RESUMES: on restart it skips ids
already in OUT. So a wall-clock timeout loses nothing and a re-run continues.

Parse: split on the last '</think>' -> answer = text AFTER it -> take the LAST class word
there (final decision). If </think> never closed (looped/hit cap): think_closed=False,
fall back to the last class word in the whole trace (parse_source='fallback').

Env: MODEL_LOCAL_PATH, QWEN_ADAPTER_PATH(optional), ENABLE_THINKING(1/0), QWEN_MAX_NEW_TOKENS, OUT
"""
import os, csv, re, torch
from collections import defaultdict, Counter
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL   = os.environ["MODEL_LOCAL_PATH"]
CATALOG = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/data/test_catalog.csv"
THINK   = os.environ.get("ENABLE_THINKING", "1") == "1"
MAXTOK  = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "2048" if THINK else "10"))
OUT     = os.environ["OUT"]
CLASSES = ["arm", "stream", "shell", "diffuse"]
PROMPT  = ("This galaxy image shows faint debris. Classify the debris morphology. "
           "Reply with one word: arm, stream, shell, or diffuse.")
FIELDS  = ["id_str", "true_name", "pred_name", "parse_source", "conclusion",
           "think_closed", "n_gen_tokens", "answer_segment", "full_output"]
# budget forcing: think up to THINK_BUDGET; if </think> not emitted, inject it and force a
# 1-word answer. Guarantees a conclusion + captures the answer where the reasoning landed.
BUDGET      = int(os.environ.get("THINK_BUDGET", str(MAXTOK)))   # soft thinking cap
FORCE       = THINK and os.environ.get("FORCE_CONCLUDE", "1") == "1"
FORCE_NUDGE = "</think>\n\nBased on the reasoning above, the final answer (one word) is: "

print(f"MODE: {'THINK' if THINK else 'NO-THINK'}  max_new_tokens={MAXTOK}", flush=True)
proc  = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
adapter = os.environ.get("QWEN_ADAPTER_PATH")
if adapter:
    model.load_adapter(adapter); print(f"ADAPTER LOADED (fine-tuned): {adapter}", flush=True)
else:
    print("no adapter -> zero-shot base model", flush=True)
model.eval()

def last_class(text):
    t = text.lower()
    hits = [(m.start(), c) for c in CLASSES for m in re.finditer(rf"\b{c}\b", t)]
    return sorted(hits)[-1][1] if hits else None

# --- resume: skip ids already written ---
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r["id_str"] for r in csv.DictReader(f)}
    print(f"RESUME: {len(done)} rows already in {OUT}", flush=True)
new_file = not os.path.exists(OUT)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fout = open(OUT, "a", newline="")
writer = csv.DictWriter(fout, fieldnames=FIELDS)
if new_file:
    writer.writeheader(); fout.flush()

rows = list(csv.DictReader(open(CATALOG)))
todo = [r for r in rows if r["id_str"] not in done]
print(f"{len(todo)} images to do ({len(done)} already done)", flush=True)

for i, r in enumerate(todo):
    img = Image.open(r["file_loc"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    inp = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)
    n_in = inp.input_ids.shape[1]
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=BUDGET, do_sample=False)
    gen = g[:, n_in:]
    full = proc.batch_decode(gen, skip_special_tokens=True)[0]
    ntok = int(gen.shape[1])
    closed = "</think>" in full

    if closed:                                   # concluded on its own
        conclusion = "natural"
        ans_seg = full.split("</think>")[-1].strip()
    elif FORCE:                                  # budget forcing: inject </think> + make it answer
        conclusion = "forced"
        nudge_ids = proc.tokenizer(FORCE_NUDGE, add_special_tokens=False,
                                   return_tensors="pt").input_ids.to(model.device)
        forced_ids = torch.cat([g, nudge_ids], dim=1)
        img_kw = {k: inp[k] for k in ("pixel_values", "image_grid_thw") if k in inp}
        with torch.no_grad():
            g2 = model.generate(input_ids=forced_ids,
                                 attention_mask=torch.ones_like(forced_ids),
                                 max_new_tokens=12, do_sample=False, **img_kw)
        ans_seg = proc.batch_decode(g2[:, forced_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        full = full + "\n[BUDGET-FORCED]" + FORCE_NUDGE + ans_seg
    else:
        conclusion = "none"; ans_seg = ""

    pred, src = last_class(ans_seg), "answer"
    if pred is None:
        pred = last_class(full); src = "fallback" if pred else "none"
    writer.writerow({"id_str": r["id_str"], "true_name": r["label_name"], "pred_name": pred or "",
                     "parse_source": src, "conclusion": conclusion, "think_closed": closed,
                     "n_gen_tokens": ntok, "answer_segment": ans_seg[:400], "full_output": full})
    fout.flush()  # <- persist every row; timeout-safe
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(todo)}  (pred={pred} closed={closed} tok={ntok})", flush=True)
fout.close()

# report over the full file
allrows = list(csv.DictReader(open(OUT)))
parsed = [x for x in allrows if x["pred_name"]]
t = defaultdict(int); c = defaultdict(int)
for x in parsed:
    t[x["true_name"]] += 1
    if x["pred_name"] == x["true_name"]: c[x["true_name"]] += 1
ba = sum((c[k]/t[k] if t[k] else 0) for k in CLASSES) / len(CLASSES)
print(f"\n=== {'THINK' if THINK else 'NO-THINK'}{' + adapter' if adapter else ''} | which_lsb ===", flush=True)
print(f"  balanced_acc = {ba:.3f}  on {len(parsed)}/{len(allrows)} parsed ({len(allrows)}/218 evaluated)", flush=True)
print(f"  conclusion: {dict(Counter(x.get('conclusion','') for x in allrows))}   (natural=closed itself, forced=budget-forced)", flush=True)
print(f"  think_closed(natural): {dict(Counter(str(x['think_closed']) for x in allrows))}", flush=True)
print(f"  parse_source: {dict(Counter(x['parse_source'] for x in allrows))}", flush=True)
toks = sorted(int(x['n_gen_tokens']) for x in allrows)
print(f"  median gen tokens: {toks[len(toks)//2]}", flush=True)
print(f"  saved -> {OUT}", flush=True)
