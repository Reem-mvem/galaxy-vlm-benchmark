"""Gemini 3.5 Flash thinking-level sweep on RINGS (binary, same 1000 test set as the other models).
Self-contained: needs only google-genai + stdlib. No vlm_eval / runners.
Levels: off, minimal, low, medium, high  ·  1 seed  ·  same yes/no prompt as the other rings runs.
Prints running cost + a confusion matrix + cost per level. Resumable (skips done (level,id)).
Retries transient API errors (503 / timeouts) per image; skips only that image if it never succeeds.
Run:  python gemini_rings_thinking.py
"""
import os, sys, csv, re, time
from pathlib import Path
from google import genai
from google.genai import types

# ===================== geir paths =====================
IMAGES_DIR = Path("/geir_data/scr/reem/gz_rings/images/test")          # full 18k test folder
CATALOG    = Path("rings_1000_geir.csv")                                # id_str,filename,label_name
ENV_PATH   = Path("/geir_data/scr/reem/zoobot_project/.env")            # holds GEMINI_API_KEY=...
OUT        = Path("gemini_rings_thinking_results.csv")
SUMMARY    = Path("gemini_rings_thinking_summary.csv")
# ======================================================

MODEL   = "gemini-3.5-flash"
PROMPT  = ("Does this galaxy image show a ring (a circular feature within or around "
           "the galaxy)? Reply with one word: yes or no.")
CLASSES = ["ring", "not_ring"]
PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.50, 9.00      # $1.50 in, $9.00 out (incl. thinking)

CONFIGS = [
    ("off",     dict(thinking_budget=0)),
    ("minimal", dict(thinking_level="MINIMAL")),
    ("low",     dict(thinking_level="LOW")),
    ("medium",  dict(thinking_level="MEDIUM")),
    ("high",    dict(thinking_level="HIGH")),
]

def read_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    for line in ENV_PATH.read_text().splitlines():
        if line.strip().startswith("GEMINI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"GEMINI_API_KEY not found in env or {ENV_PATH}")

def img_path(row):
    fn = row.get("filename") or f"{row['id_str']}.jpg"
    if fn.startswith("test_"):
        fn = fn[5:]                 # geir names are number-only: test_7531.jpg -> 7531.jpg
    return IMAGES_DIR / fn

def parse(text):
    t = (text or "").strip().lower()
    if re.search(r"\byes\b", t): return "ring"
    if re.search(r"\bno\b", t):  return "not_ring"
    if re.search(r"not[_ ]?a? ?ring|no ring|without", t): return "not_ring"
    if re.search(r"\bring\b", t): return "ring"
    return None

client = genai.Client(api_key=read_key())
rows = list(csv.DictReader(open(CATALOG)))
print(f"{len(rows)} images x {len(CONFIGS)} levels = {len(rows)*len(CONFIGS)} calls")

# sanity: first image must exist
p0 = img_path(rows[0])
if not p0.exists():
    sys.exit(f"First image not found: {p0}\nCheck IMAGES_DIR / filename column.")

done = set()
if OUT.exists():
    for r in csv.DictReader(open(OUT)): done.add((r["level"], r["id_str"]))
new_file = not OUT.exists()
fout = open(OUT, "a", newline="", encoding="utf-8")
W = csv.DictWriter(fout, fieldnames=["level","id_str","true_name","pred_name","think_tok","cost_usd"])
if new_file: W.writeheader()

total_cost = 0.0
for name, cfg in CONFIGS:
    tcfg = types.ThinkingConfig(**cfg)
    for i, r in enumerate(rows):
        if (name, r["id_str"]) in done: continue
        # --- retry transient errors (503 / timeout / rate limit); skip only this image if it never lands ---
        resp = None
        for attempt in range(6):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[types.Part.from_bytes(data=img_path(r).read_bytes(),
                                                     mime_type="image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(thinking_config=tcfg))
                break
            except Exception as e:
                wait = min(2 ** attempt, 30)     # backoff: 1, 2, 4, 8, 16, 30 s
                print(f"  retry {name} {r['id_str']} (attempt {attempt+1}/6, {type(e).__name__}) wait {wait}s")
                time.sleep(wait)
        if resp is None:
            print(f"  !! {name} {r['id_str']} skipped after 6 retries")
            continue        # skip only THIS image, keep the level going
        pred = parse(resp.text)
        u = resp.usage_metadata
        it, ot, tt = u.prompt_token_count or 0, u.candidates_token_count or 0, u.thoughts_token_count or 0
        cost = (it*PRICE_IN_PER_M + (ot+tt)*PRICE_OUT_PER_M)/1_000_000
        total_cost += cost
        W.writerow({"level":name,"id_str":r["id_str"],"true_name":r["label_name"],
                    "pred_name":pred or "","think_tok":tt,"cost_usd":round(cost,6)})
        fout.flush()
        if (i+1) % 50 == 0:
            print(f"  {name:8} {i+1}/{len(rows)}  running_cost=${total_cost:.3f}")
        time.sleep(0.3)
fout.close()

# ---- per-level summary: confusion matrix + balanced acc + cost ----
allrows = list(csv.DictReader(open(OUT)))
order = {n:i for i,(n,_) in enumerate(CONFIGS)}
levels = sorted(set(x["level"] for x in allrows), key=lambda n: order.get(n,99))
sf = open(SUMMARY, "w", newline="", encoding="utf-8")
SW = csv.DictWriter(sf, fieldnames=["level","n","mean_think_tok","balanced_acc","accuracy",
                                    "ring_ring","ring_notring","notring_ring","notring_notring","cost"])
SW.writeheader()
for name in levels:
    g = [x for x in allrows if x["level"]==name]
    cm = {(t,p):0 for t in CLASSES for p in CLASSES}
    for x in g:                                    # <-- FIXED: normalize labels, never crash
        tn = x["true_name"] if x["true_name"] in CLASSES else ("not_ring" if "ring" in (x["true_name"] or "") else None)
        pn = x["pred_name"] if x["pred_name"] in CLASSES else None
        if tn and pn: cm[(tn, pn)] += 1
    n_ring = sum(cm[("ring",p)] for p in CLASSES); n_not = sum(cm[("not_ring",p)] for p in CLASSES)
    rec_ring = cm[("ring","ring")]/n_ring if n_ring else 0
    rec_not  = cm[("not_ring","not_ring")]/n_not if n_not else 0
    ba  = (rec_ring + rec_not)/2
    acc = (cm[("ring","ring")]+cm[("not_ring","not_ring")])/max(1,sum(cm.values()))
    mtt = sum(int(x["think_tok"]) for x in g)/len(g)
    tc  = sum(float(x["cost_usd"]) for x in g)
    print(f"\n===== {name.upper()}  (n={len(g)}, mean_think_tok={mtt:.0f}, cost=${tc:.3f}) =====")
    print(f"                 pred:ring   pred:not_ring")
    print(f"  true:ring        {cm[('ring','ring')]:>5}        {cm[('ring','not_ring')]:>5}")
    print(f"  true:not_ring    {cm[('not_ring','ring')]:>5}        {cm[('not_ring','not_ring')]:>5}")
    print(f"  balanced_acc={ba:.3f}   accuracy={acc:.3f}")
    SW.writerow({"level":name,"n":len(g),"mean_think_tok":round(mtt,1),"balanced_acc":round(ba,4),
                 "accuracy":round(acc,4),"ring_ring":cm[("ring","ring")],"ring_notring":cm[("ring","not_ring")],
                 "notring_ring":cm[("not_ring","ring")],"notring_notring":cm[("not_ring","not_ring")],
                 "cost":round(tc,4)})
sf.close()
print(f"\nTOTAL COST: ${total_cost:.3f}   saved -> {OUT.name} + {SUMMARY.name}")