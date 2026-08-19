"""Gemini thinking-LEVEL sweep on streams/shells (30 imgs, 3-class).
Does MORE thinking = better accuracy? Sweeps off -> minimal -> low -> medium -> high
(the knob that actually scales thinking), records ACTUAL thinking tokens + cost per call,
and reports balanced accuracy per level. Resumable.
Setup: pip install google-genai   |   run: python gemini_level_sweep.py
"""
import os, sys, csv, re, time
from pathlib import Path
from google import genai
from google.genai import types

REPO_ROOT = Path(r"C:\Users\reem2\Desktop\geir-local\galaxy-vlm-benchmark")
ENV_PATH  = REPO_ROOT / ".env"
CATALOG   = REPO_ROOT / "tasks" / "streams_shells_neither" / "data" / "test_catalog.csv"
OUT       = REPO_ROOT / "gemini_level_sweep_results.csv"
SUMMARY   = REPO_ROOT / "gemini_level_sweep_summary.csv"
MODEL     = "gemini-3.5-flash"
PROMPT    = ("This is a galaxy image from an astronomical survey. Does it show a stellar stream, "
             "a shell, or neither? Reply with one word: stream, shell, or neither.")
CLASSES   = ["stream", "shell", "neither"]
PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.50, 9.00     # $1.50 in, $9.00 out (incl. thinking)

# ========== the sweep: name -> ThinkingConfig (this knob actually varies the amount) ==========
CONFIGS = [
    ("off",     types.ThinkingConfig(thinking_budget=0)),
    ("minimal", types.ThinkingConfig(thinking_level="MINIMAL")),
    ("low",     types.ThinkingConfig(thinking_level="LOW")),
    ("medium",  types.ThinkingConfig(thinking_level="MEDIUM")),
    ("high",    types.ThinkingConfig(thinking_level="HIGH")),
]
# ==============================================================================================

def read_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    for line in ENV_PATH.read_text().splitlines():
        if line.strip().startswith("GEMINI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"GEMINI_API_KEY not found in {ENV_PATH}")

def local_img(file_loc):
    tail = file_loc.split("galaxy-vlm-benchmark/")[-1].replace("/", os.sep)
    return REPO_ROOT / tail

def parse(text):
    t = (text or "").strip().lower()
    hits = [(m.start(), c) for c in CLASSES for m in re.finditer(rf"\b{c}\b", t)]
    if hits: return sorted(hits)[-1][1]
    if re.search(r"\b(none|no|normal|nothing|absent)\b", t): return "neither"
    return None

client = genai.Client(api_key=read_key())
rows = list(csv.DictReader(open(CATALOG)))

done = set()
if OUT.exists():
    for r in csv.DictReader(open(OUT)):
        done.add((r["level"], r["id_str"]))
new_file = not OUT.exists()
fout = open(OUT, "a", newline="", encoding="utf-8")
W = csv.DictWriter(fout, fieldnames=["level","id_str","true_name","pred_name","think_tok","cost_usd"])
if new_file: W.writeheader()

total_cost = 0.0
for name, tcfg in CONFIGS:
    for i, r in enumerate(rows):
        if (name, r["id_str"]) in done:
            continue
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=local_img(r["file_loc"]).read_bytes(),
                                                 mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(thinking_config=tcfg),
            )
        except Exception as e:
            print(f"!! level '{name}' failed ({type(e).__name__}: {e}) -> skipping this level")
            break                                    # invalid level enum? skip it, keep the rest
        pred = parse(resp.text)
        u = resp.usage_metadata
        it, ot, tt = u.prompt_token_count or 0, u.candidates_token_count or 0, u.thoughts_token_count or 0
        cost = (it*PRICE_IN_PER_M + (ot+tt)*PRICE_OUT_PER_M)/1_000_000
        total_cost += cost
        W.writerow({"level": name, "id_str": r["id_str"], "true_name": r["label_name"],
                    "pred_name": pred or "", "think_tok": tt, "cost_usd": round(cost,6)})
        fout.flush()
        print(f"{name:8} {r['id_str']:8} true={r['label_name']:8} pred={pred}  think_tok={tt}  ${cost:.4f}")
        time.sleep(0.4)
fout.close()

# ---- per-level summary = the curve points (ordered by actual thinking) ----
allrows = list(csv.DictReader(open(OUT)))
order = {n: i for i, (n, _) in enumerate(CONFIGS)}
levels = sorted(set(x["level"] for x in allrows), key=lambda n: order.get(n, 99))
sf = open(SUMMARY, "w", newline="", encoding="utf-8")
SW = csv.DictWriter(sf, fieldnames=["level","n","mean_think_tok","balanced_acc","total_cost"]); SW.writeheader()
print("\n=== THINKING-LEVEL SWEEP SUMMARY (streams/shells, Gemini) ===")
for name in levels:
    grp = [x for x in allrows if x["level"] == name]
    t = {c:0 for c in CLASSES}; ok = {c:0 for c in CLASSES}
    for x in grp:
        t[x["true_name"]] += 1
        if x["pred_name"] == x["true_name"]: ok[x["true_name"]] += 1
    ba  = sum(ok[c]/t[c] if t[c] else 0 for c in CLASSES)/len(CLASSES)
    mtt = sum(int(x["think_tok"]) for x in grp)/len(grp)
    tc  = sum(float(x["cost_usd"]) for x in grp)
    SW.writerow({"level":name,"n":len(grp),"mean_think_tok":round(mtt,1),"balanced_acc":round(ba,4),"total_cost":round(tc,4)})
    print(f"  {name:8} mean_think_tok={mtt:7.1f}  balanced_acc={ba:.3f}  (n={len(grp)}, ${tc:.3f})")
sf.close()
print(f"\nTOTAL COST: ${total_cost:.3f}   saved -> {OUT.name} + {SUMMARY.name}")