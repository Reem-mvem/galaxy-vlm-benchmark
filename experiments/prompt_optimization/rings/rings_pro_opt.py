"""

Loop:
  Round 0  — Generator writes N diverse cold-start prompts
  Round 1+ — Generator sees top-M past results, writes N improved prompts
  Evaluator scores each prompt on 300 val images (balanced accuracy)
  Results saved incrementally; identical prompts never re-scored (cached)

"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score
from google import genai

load_dotenv()

# ── paths & models ────────────────────────────────────────────────────────────
EVALUATOR_MODEL = "gemini-3.5-flash"
GENERATOR_MODEL = "gemini-3.1-flash-lite"
VAL_CATALOG = "rings_val_catalog.csv"
RESULTS_CSV = "rings_prompt_opt_results.csv"
CACHE_FILE  = "rings_prompt_opt_cache.json"
POS_CLASS   = "ring"
NEG_CLASS   = "not_ring"
IMG_SIZE    = 224

TOKENS_PER_CALL = 1200
PRICE_PER_1M    = 1.50

# ── image preprocessing (same as CNN input) ───────────────────────────────────
def preprocess_image(path: str) -> bytes:
    img = np.array(Image.open(path).convert("RGB")).astype(float)
    for c in range(3):
        lo, hi = np.percentile(img[:, :, c], [1, 99])
        img[:, :, c] = np.clip((img[:, :, c] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    pil = Image.fromarray(img.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

# ── task context ──────────────────────────────────────────────────────────────
TASK_DESCRIPTION = """\
Task: classify galaxy images as containing a galactic ring or not.

Background:
- A galactic ring is a distinct circular or elliptical band of stars or dust \
that clearly encircles the galaxy core — visually separate from the main disk \
or bulge.
- Ring galaxies can appear as a complete ring around the nucleus or as a \
partial arc-shaped band.
- Non-ring galaxies include spirals, ellipticals, mergers, and irregulars that \
may have arc-like shapes but lack the characteristic closed ring morphology.

Output requirement: every prompt you write MUST end with the instruction \
to reply with ONE word: ring or not_ring\
"""

# ── meta-prompts ──────────────────────────────────────────────────────────────
META_ROUND0 = """\
You are an expert prompt engineer for vision-language models.

{task}

Generate {n} diverse candidate prompts for this binary classification task.
Vary the: level of visual detail, reasoning style (direct vs step-by-step), \
emphasis (what to look for vs what to ignore), and framing (scientific vs descriptive).

Rules:
- Each prompt MUST end with: "Reply with ONE word: ring or not_ring"
- Do NOT mention the class ratio or leak any statistics
- Do NOT coach the answer — describe visual features only

Return ONLY a valid JSON array of {n} strings, no other text:
["prompt 1", "prompt 2", ...]\
"""

META_OPRO = """\
You are an expert prompt engineer for vision-language models.

{task}

You previously tested these prompts for galactic ring classification \
(balanced accuracy, higher is better, max=1.0):

{past}

Best score so far: {best:.4f}

Generate {n} NEW prompts likely to score higher. Learn from what worked \
and what did not. Do not repeat any prompt already listed above.

Rules:
- Each prompt MUST end with: "Reply with ONE word: ring or not_ring"
- Do NOT mention the class ratio or leak any statistics
- Do NOT coach the answer — describe visual features only

Return ONLY a valid JSON array of {n} strings, no other text:
["prompt 1", "prompt 2", ...]\
"""

# ── cache helpers ─────────────────────────────────────────────────────────────
def phash(prompt: str) -> str:
    return hashlib.md5(prompt.strip().encode()).hexdigest()

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ── generator ─────────────────────────────────────────────────────────────────
def generate_prompts(client, n: int, round_num: int, top_results: list) -> list:
    if round_num == 0:
        meta = META_ROUND0.format(task=TASK_DESCRIPTION, n=n)
    else:
        past_lines = "\n".join(
            f"[{r['score']:.4f}] {str(r['prompt'])[:180]}..."
            for r in top_results
        )
        best = max(r["score"] for r in top_results)
        meta = META_OPRO.format(task=TASK_DESCRIPTION, past=past_lines, best=best, n=n)

    print(f"\n[Generator] Round {round_num} — requesting {n} prompts...")
    try:
        response = client.models.generate_content(
            model=GENERATOR_MODEL,
            contents=[meta],
            config=genai.types.GenerateContentConfig(
                temperature=1.0,
                response_mime_type="application/json",
            ),
        )
        raw = response.text if hasattr(response, "text") else str(response)
        text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        prompts = json.loads(text)
        if isinstance(prompts, list):
            prompts = [str(p).strip() for p in prompts if p]
            print(f"[Generator] Received {len(prompts)} prompts")
            return prompts
    except Exception as e:
        print(f"[Generator] Error: {e}")
    return []

# ── evaluator ─────────────────────────────────────────────────────────────────
def parse_response(text: str) -> str:
    t = text.strip().lower()
    if "not_ring" in t or ("not" in t and "ring" in t):
        return NEG_CLASS
    if "ring" in t:
        return POS_CLASS
    return None

def score_prompt(client, prompt: str, catalog: pd.DataFrame) -> dict:
    true_bin, pred_bin = [], []
    n_failed = 0
    n_total  = len(catalog)

    for i, (_, row) in enumerate(catalog.iterrows()):
        img_path = Path(row["file_loc"])
        true_label = 1 if row["label_name"] == POS_CLASS else 0

        if not img_path.exists():
            true_bin.append(true_label)
            pred_bin.append(0)
            n_failed += 1
            continue

        pred = None
        for attempt in range(2):
            try:
                print(f"  [{i+1}/{n_total}] querying...", end=" ", flush=True)
                img_bytes = preprocess_image(str(img_path))
                response = client.models.generate_content(
                    model=EVALUATOR_MODEL,
                    contents=[
                        genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.0,
                        thinking_config=genai.types.ThinkingConfig(thinking_level="MINIMAL"),
                    ),
                )
                raw  = response.text if hasattr(response, "text") else str(response)
                pred = parse_response(raw)
                print(f"{'ring' if pred == POS_CLASS else 'not_ring'}", flush=True)
                break
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                if attempt == 0:
                    time.sleep(2)
                else:
                    n_failed += 1

        pred_label = (1 if pred == POS_CLASS else 0) if pred is not None else 0
        if pred is None:
            n_failed += 1
        true_bin.append(true_label)
        pred_bin.append(pred_label)
        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            partial = balanced_accuracy_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else 0.0
            print(f"  [{i+1}/{n_total}] partial_bal_acc={partial:.4f}  n_failed={n_failed}")

    bal_acc    = float(balanced_accuracy_score(true_bin, pred_bin)) if true_bin else 0.0
    n_pos_pred = sum(pred_bin)
    print(f"  → bal_acc={bal_acc:.4f}  pred_pos={n_pos_pred}/{sum(true_bin)}  failed={n_failed}")
    return {"bal_acc": bal_acc, "n_failed": n_failed, "n_pos_pred": n_pos_pred}

# ── main loop ─────────────────────────────────────────────────────────────────
def run_optimization(n_prompts: int, n_rounds: int, top_m: int):
    if not os.path.exists(VAL_CATALOG):
        print(f"ERROR: {VAL_CATALOG} not found. Run prepare_rings_val.py first.")
        sys.exit(1)

    catalog = pd.read_csv(VAL_CATALOG)
    n_val   = len(catalog)
    n_pos   = catalog["label_name"].eq(POS_CLASS).sum()
    print(f"Val set: {n_val} images  ({n_pos} ring / {n_val - n_pos} not_ring)")

    total_eval_calls = n_prompts * n_rounds * n_val
    est_cost = total_eval_calls * TOKENS_PER_CALL * PRICE_PER_1M / 1_000_000
    print(f"\nCost estimate:")
    print(f"  {n_prompts} prompts × {n_rounds} rounds × {n_val} val = {total_eval_calls:,} eval calls")
    print(f"  ≈ ${est_cost:.2f}  (input tokens only — output adds ~20%)")
    print(f"\nPress Enter to start, Ctrl+C to cancel...")
    input()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    cache       = load_cache()
    all_results = []

    if os.path.exists(RESULTS_CSV):
        existing    = pd.read_csv(RESULTS_CSV)
        all_results = existing.to_dict("records")
        print(f"Resuming: {len(all_results)} existing results loaded")

    try:
        for round_num in range(n_rounds):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}  ({len(all_results)} results so far)")
            print(f"{'='*60}")

            top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:top_m]
            candidates  = generate_prompts(client, n_prompts, round_num, top_results)

            if not candidates:
                print(f"[Round {round_num}] No candidates returned — skipping")
                continue

            for p_idx, prompt in enumerate(candidates):
                h = phash(prompt)
                print(f"\n[Round {round_num} | Prompt {p_idx+1}/{len(candidates)}]")
                print(f"  {prompt[:120]}...")

                if h in cache:
                    score = cache[h]
                    print(f"  [CACHED] bal_acc={score:.4f}")
                    result = {"round": round_num, "prompt": prompt,
                              "score": score, "n_failed": 0,
                              "n_pos_pred": -1, "cached": True}
                else:
                    scored = score_prompt(client, prompt, catalog)
                    score  = scored["bal_acc"]
                    cache[h] = score
                    save_cache(cache)
                    result = {"round": round_num, "prompt": prompt,
                              "score": score, "n_failed": scored["n_failed"],
                              "n_pos_pred": scored["n_pos_pred"], "cached": False}

                all_results.append(result)
                pd.DataFrame(all_results).sort_values("score", ascending=False).to_csv(
                    RESULTS_CSV, index=False)

    except KeyboardInterrupt:
        print(f"\n\nStopped early — {len(all_results)} results saved to {RESULTS_CSV}")

    if not all_results:
        return

    df = pd.DataFrame(all_results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\n{'='*60}")
    print("FINAL RANKED RESULTS")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  score={row['score']:.4f}  round={row['round']}")
        print(f"  {str(row['prompt'])[:100]}...")

    best = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"BEST PROMPT  (val bal_acc={best['score']:.4f})")
    print(f"{'='*60}")
    print(best["prompt"])
    print(f"\nAll results saved → {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GZ Rings OPRO prompt optimization")
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--n_rounds",  type=int, default=3)
    parser.add_argument("--top_m",     type=int, default=3)
    args = parser.parse_args()

    run_optimization(args.n_prompts, args.n_rounds, args.top_m)
