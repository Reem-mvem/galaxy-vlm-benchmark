#!/usr/bin/env python
"""
OPRO-style prompt optimization for Gemini on which-lsb (arm/stream/shell/diffuse).

4-class single-label classification.
Score = balanced accuracy across all 4 classes.
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score
from google import genai

load_dotenv()

# ── paths & models ────────────────────────────────────────────────────────────
EVALUATOR_MODEL = "gemini-3.5-flash"
GENERATOR_MODEL = "gemini-3.1-flash-lite"
VAL_CATALOG = "which_lsb_val_catalog.csv"
RESULTS_CSV = "which_lsb_prompt_opt_results.csv"
CACHE_FILE  = "which_lsb_prompt_opt_cache.json"
CLASS_NAMES = ["arm", "stream", "shell", "diffuse"]

TOKENS_PER_CALL = 1200
PRICE_PER_1M    = 1.50

# ── task context ──────────────────────────────────────────────────────────────
TASK_DESCRIPTION = """\
Task: classify a galaxy image as showing exactly ONE of these four tidal feature types:
arm, stream, shell, or diffuse.

Definitions:
- arm: a tidal arm — a broad, curved sweep of stars pulling away from the galaxy body, \
resembling a spiral arm but originating from a tidal interaction.
- stream: a tidal stream — a thin, elongated, linear or gently curving trail of stars \
extending from the galaxy, narrower than an arm.
- shell: tidal shells — concentric arc-like ripples or layered structures surrounding \
the galaxy, often appearing as faint nested arcs.
- diffuse: diffuse tidal debris — faint, irregular, extended emission without a clear \
geometric structure, spread beyond the smooth galaxy halo.

Each image shows exactly one of these feature types.

Output requirement: every prompt you write MUST end with the instruction \
to reply with ONE word: arm, stream, shell, or diffuse\
"""

# ── meta-prompts ──────────────────────────────────────────────────────────────
META_ROUND0 = """\
You are an expert prompt engineer for vision-language models.

{task}

Generate {n} diverse candidate prompts for this 4-class classification task.
Vary the: level of visual detail, reasoning style (direct vs step-by-step), \
emphasis on distinguishing features between classes, and scientific framing.

Rules:
- Each prompt MUST end with: "Reply with ONE word: arm, stream, shell, or diffuse"
- Do NOT mention class frequencies or leak any statistics
- Do NOT coach the answer — describe distinguishing visual features only

Return ONLY a valid JSON array of {n} strings, no other text:
["prompt 1", "prompt 2", ...]\
"""

META_OPRO = """\
You are an expert prompt engineer for vision-language models.

{task}

You previously tested these prompts for tidal feature type classification \
(balanced accuracy across 4 classes, higher is better, max=1.0):

{past}

Best score so far: {best:.4f}

Generate {n} NEW prompts likely to score higher. Learn from what worked \
and what did not. Do not repeat any prompt already listed above.

Rules:
- Each prompt MUST end with: "Reply with ONE word: arm, stream, shell, or diffuse"
- Do NOT mention class frequencies or leak any statistics
- Do NOT coach the answer — describe distinguishing visual features only

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
    t = text.strip().lower().split()[0] if text.strip() else ""
    if t in CLASS_NAMES:
        return t
    # fallback: check if any class name appears in full text
    for name in CLASS_NAMES:
        if name in text.strip().lower():
            return name
    return None

def score_prompt(client, prompt: str, catalog: pd.DataFrame) -> dict:
    true_labels, pred_labels = [], []
    n_failed = 0
    n_total  = len(catalog)
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}

    for i, (_, row) in enumerate(catalog.iterrows()):
        img_path   = Path(row["file_loc"])
        true_label = label_map[row["label_name"]]

        if not img_path.exists():
            true_labels.append(true_label)
            pred_labels.append(0)
            n_failed += 1
            continue

        pred = None
        for attempt in range(2):
            try:
                print(f"  [{i+1}/{n_total}] querying...", end=" ", flush=True)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
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
                print(f"{pred}", flush=True)
                break
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                if attempt == 0:
                    time.sleep(2)
                else:
                    n_failed += 1

        pred_label = label_map.get(pred, 0) if pred is not None else 0
        if pred is None:
            n_failed += 1
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            partial = balanced_accuracy_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else 0.0
            print(f"  [{i+1}/{n_total}] partial_bal_acc={partial:.4f}  n_failed={n_failed}")

    bal_acc = float(balanced_accuracy_score(true_labels, pred_labels)) if true_labels else 0.0
    print(f"  → bal_acc={bal_acc:.4f}  failed={n_failed}")
    return {"bal_acc": bal_acc, "n_failed": n_failed, "n_pos_pred": -1}

# ── main loop ─────────────────────────────────────────────────────────────────
def run_optimization(n_prompts: int, n_rounds: int, top_m: int):
    if not os.path.exists(VAL_CATALOG):
        print(f"ERROR: {VAL_CATALOG} not found. Run download_which_lsb_val.py first.")
        sys.exit(1)

    catalog = pd.read_csv(VAL_CATALOG)
    n_val   = len(catalog)
    print(f"Val set: {n_val} images")
    print(catalog["label_name"].value_counts().to_string())

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
                              "n_pos_pred": -1, "cached": False}

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
    parser = argparse.ArgumentParser(description="which-lsb OPRO prompt optimization")
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--n_rounds",  type=int, default=3)
    parser.add_argument("--top_m",     type=int, default=3)
    args = parser.parse_args()

    run_optimization(args.n_prompts, args.n_rounds, args.top_m)
