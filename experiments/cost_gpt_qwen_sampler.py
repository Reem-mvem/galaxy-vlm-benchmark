"""Cost sampler for the zero-shot API models, matching vlm_eval/providers.py EXACTLY.
  GPT-5.6-sol (OpenAI): image detail='low', max_completion_tokens=500, reasoning at model default.
  Qwen3.5-397B (OpenRouter): reasoning DISABLED, max_tokens=20, temperature=0.
Reads OPENAI_API_KEY + OPENROUTER_API_KEY from the .env (no export needed).
Runs a small sample, logs REAL token usage, reports avg tokens + $/100 images per model.
No need to run full sets — per-image cost is stable. Run:  python cost_sampler.py
"""
import os, csv, base64, statistics, requests
from pathlib import Path
from openai import OpenAI

# ===================== config =====================
CATALOG    = Path("rings_1000_geir.csv")                       # id_str,filename,label_name
IMAGES_DIR = Path("/geir_data/scr/reem/gz_rings/images/test")  # full test folder
ENV_PATH   = Path("/geir_data/scr/reem/zoobot_project/.env")   # holds OPENAI_API_KEY + OPENROUTER_API_KEY
PROMPT     = ("Does this galaxy image show a ring (a circular feature within or around "
              "the galaxy)? Reply with one word: yes or no.")
N_SAMPLE   = 25            # bump to 50 if GPT's reasoning-token std is large

GPT_MODEL  = "gpt-5.6-sol"                 # <- confirm exact id used in the benchmark
QWEN_MODEL = "qwen/qwen3.5-397b-a17b"      # <- confirm exact OpenRouter slug
GPT_IN,  GPT_OUT  = 5.0,    30.0           # $/1M input, output
QWEN_IN, QWEN_OUT = 0.3025, 1.925
# ==================================================

def env_key(name):
    if os.environ.get(name):
        return os.environ[name].strip()
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

def img_path(row):
    fn = row.get("filename") or f"{row['id_str']}.jpg"
    if fn.startswith("test_"):
        fn = fn[5:]                          # geir names are number-only: test_7531.jpg -> 7531.jpg
    return IMAGES_DIR / fn

def b64(p):
    return base64.b64encode(p.read_bytes()).decode()

def report(name, ins, outs, reas, costs):
    if not costs:
        print(f"{name}: no data collected"); return
    print(f"\n=== {name}  (n={len(costs)}) ===")
    print(f"  avg input tokens : {statistics.mean(ins):.0f}")
    print(f"  avg output tokens: {statistics.mean(outs):.0f}  (of which reasoning: {statistics.mean(reas):.0f})")
    print(f"  avg cost / image : ${statistics.mean(costs):.5f}   (std ${statistics.pstdev(costs):.5f})")
    print(f"  COST / 100 IMAGES: ${statistics.mean(costs)*100:.3f}")

rows = list(csv.DictReader(open(CATALOG)))[:N_SAMPLE]
print(f"sampling {len(rows)} images")

# ---------- GPT via OpenAI (matches OpenAIProvider) ----------
key = env_key("OPENAI_API_KEY")
if key:
    client = OpenAI(api_key=key)
    ins, outs, reas, costs = [], [], [], []
    for r in rows:
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64(img_path(r))}", "detail": "low"}},
                    {"type": "text", "text": PROMPT},
                ]}],
                max_completion_tokens=500)
        except Exception as e:
            print(f"  GPT error: {e}"); continue
        u = resp.usage
        rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        cost = (u.prompt_tokens * GPT_IN + u.completion_tokens * GPT_OUT) / 1_000_000
        ins.append(u.prompt_tokens); outs.append(u.completion_tokens); reas.append(rt); costs.append(cost)
    report("GPT-5.6-sol", ins, outs, reas, costs)
else:
    print("GPT-5.6-sol: OPENAI_API_KEY not found in .env")

# ---------- Qwen-397B via OpenRouter (matches OpenRouterProvider) ----------
key = env_key("OPENROUTER_API_KEY")
if key:
    ins, outs, costs = [], [], []
    for r in rows:
        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64(img_path(r))}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 20,
            "temperature": 0.0,
            "reasoning": {"enabled": False},
        }
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"},
                                 json=payload, timeout=60)
            resp.raise_for_status()
            u = resp.json().get("usage", {})
        except Exception as e:
            print(f"  Qwen error: {e}"); continue
        pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = (pin * QWEN_IN + pout * QWEN_OUT) / 1_000_000
        ins.append(pin); outs.append(pout); costs.append(cost)
    report("Qwen3.5-397B", ins, outs, [0] * len(ins), costs)
else:
    print("Qwen3.5-397B: OPENROUTER_API_KEY not found in .env")