#!/usr/bin/env python
"""
Few-shot Gemini evaluation on Which-LSB (arm / stream / shell / diffuse).

Passes 1 example image per class before each test query.
Examples are picked from the test catalog and excluded from the evaluation pool.

Key motivation: zero-shot Gemini gets only 23% recall on stream — it confuses
streams with arms and diffuse. Few-shot aims to show it what a stream looks like.
"""

import os
import re
import sys
import csv
import time
import textwrap
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
    sys.exit(1)

MODEL_NAME   = "gemini-3.5-flash"
CATALOG_PATH = Path(__file__).resolve().parent / "data" / "test_catalog.csv"  # built by eval_gemini.py --build_catalog_only
CLASS_NAMES  = ["arm", "stream", "shell", "diffuse"]

# ── prompts ───────────────────────────────────────────────────────────────────
CAPTIONS = {
    "arm":     "EXAMPLE — ARM: a broad, curved stellar feature sweeping away from the galaxy body.",
    "stream":  "EXAMPLE — STREAM: a narrow, elongated stellar trail, relatively straight or gently curved.",
    "shell":   "EXAMPLE — SHELL: a concentric arc or ripple structure around the galaxy.",
    "diffuse": "EXAMPLE — DIFFUSE: a faint, irregular glow extending beyond the galaxy edges with no clear structure.",
}

PROMPT_QUERY = """This galaxy has tidal debris from a past merger. Study the four class examples above carefully, then classify this galaxy's debris — pick exactly ONE:

arm — broad curved feature sweeping from the galaxy body (like the arm example)
stream — narrow elongated stellar trail, relatively straight (like the stream example)
shell — concentric arc or ripple structure (like the shell example)
diffuse — faint irregular glow with no clear structure (like the diffuse example)

Pay close attention to the stream example: streams are NARROW and ELONGATED, while arms and diffuse features are broader and more irregular,but dont overpredict the stream class. If you are unsure, make your best guess based on the examples above.

Reply with ONE word: arm, stream, shell, or diffuse"""


def load_catalog():
    df = pd.read_csv(CATALOG_PATH)
    print(f"Loaded {len(df)} test examples")
    for name in CLASS_NAMES:
        n = (df["label_name"] == name).sum()
        print(f"  {name}: {n}")
    return df


def pick_examples(df, n_per_class=1, seed=42):
    """Pick n_per_class examples per class from the test set."""
    rng = np.random.default_rng(seed)
    examples = []
    for cls in CLASS_NAMES:
        pool = df[df["label_name"] == cls]
        idx  = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
        examples.append(pool.iloc[idx])
    examples_df = pd.concat(examples).reset_index(drop=True)
    print(f"\nFew-shot examples ({n_per_class} per class):")
    for _, row in examples_df.iterrows():
        print(f"  {row['id_str']}  [{row['label_name']}]")
    return examples_df


def read_image(path):
    with open(path, "rb") as f:
        return f.read()


def build_contents(examples_df, test_img_bytes):
    """Build multi-image content: 4 class examples then the test image."""
    contents = []
    for _, row in examples_df.iterrows():
        img_bytes = read_image(str(row["file_loc"]))
        contents.append(genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        contents.append(CAPTIONS[row["label_name"]])
    contents.append(genai.types.Part.from_bytes(data=test_img_bytes, mime_type="image/jpeg"))
    contents.append(PROMPT_QUERY)
    return contents


def parse_prediction(text):
    text = text.strip().lower()
    for cls in CLASS_NAMES:
        if re.search(rf'\b{cls}\b', text):
            return cls
    return None


def save_example_grid(examples_df, out_path):
    n = len(examples_df)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Few-shot examples passed to Gemini (1 per class)", fontweight="bold")
    colors = {"arm": "#0072B2", "stream": "#009E73", "shell": "#CC79A7", "diffuse": "#E69F00"}
    for ax, (_, row) in zip(axes, examples_df.iterrows()):
        img = Image.open(row["file_loc"])
        ax.imshow(img)
        ax.axis("off")
        cls = row["label_name"]
        ax.set_title(cls.upper(), fontsize=10, color=colors.get(cls, "black"), fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved example grid: {out_path}")


def run_eval(output_csv, thinking_level, n_per_class):
    catalog     = load_catalog()
    examples_df = pick_examples(catalog, n_per_class=n_per_class)
    example_ids = set(examples_df["id_str"])

    test_df = catalog[~catalog["id_str"].isin(example_ids)].reset_index(drop=True)
    print(f"\nEvaluating {len(test_df)} images (excluded {len(example_ids)} used as examples)")

    ex_grid_path = output_csv.replace(".csv", "_examples.png")
    save_example_grid(examples_df, ex_grid_path)

    sample_ids_path = output_csv.replace(".csv", "_sample_ids.txt")
    test_df["id_str"].to_csv(sample_ids_path, index=False, header=False)

    client  = genai.Client(api_key=API_KEY)
    records = []
    n_total = len(test_df)
    total_input_tokens    = 0
    total_output_tokens   = 0
    total_thinking_tokens = 0

    try:
        for i, (_, row) in enumerate(test_df.iterrows()):
            img_path = Path(row["file_loc"])
            if not img_path.exists():
                print(f"[{i+1}/{n_total}] SKIPPED: {img_path}", file=sys.stderr)
                continue

            print(f"[{i+1}/{n_total}] {row['id_str']}", end=" ", flush=True)
            try:
                test_bytes = read_image(str(img_path))
                contents   = build_contents(examples_df, test_bytes)

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_level=thinking_level.upper()
                        ),
                    ),
                )

                raw_text = response.text if hasattr(response, "text") else str(response)
                pred     = parse_prediction(raw_text)

                usage = response.usage_metadata
                input_tok    = usage.prompt_token_count     or 0
                output_tok   = usage.candidates_token_count or 0
                thinking_tok = usage.thoughts_token_count   or 0
                total_input_tokens    += input_tok
                total_output_tokens   += output_tok
                total_thinking_tokens += thinking_tok

                records.append({
                    "id_str":       row["id_str"],
                    "true_name":    row["label_name"],
                    "pred_name":    pred,
                    "raw_response": raw_text.strip(),
                })

                print(f"→ true={row['label_name']:8s} pred={str(pred):8s} [in={input_tok}]")
                time.sleep(0.5)

            except Exception as e:
                print(f"\n  ERROR: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print(f"\n\nStopped early — saving {len(records)} records...")

    if not records:
        print("No results to save.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} predictions to {output_csv}")

    print(f"\n{'='*60}")
    print(f"TOKEN USAGE  (model={MODEL_NAME}, {n_per_class} example/class)")
    print(f"{'='*60}")
    print(f"  Input tokens:    {total_input_tokens:,}")
    print(f"  Output tokens:   {total_output_tokens:,}")
    print(f"  Thinking tokens: {total_thinking_tokens:,}")

    valid     = [r for r in records if r["pred_name"] is not None]
    n_invalid = len(records) - len(valid)
    if not valid:
        print("No valid predictions.")
        return

    true_names = [r["true_name"] for r in valid]
    pred_names = [r["pred_name"] for r in valid]
    true_idx   = [CLASS_NAMES.index(n) for n in true_names]
    pred_idx   = [CLASS_NAMES.index(n) for n in pred_names]

    bal_acc = balanced_accuracy_score(true_idx, pred_idx)
    GEMINI_ZERO_SHOT = 0.61

    print(f"\n{'='*60}")
    print(f"EVALUATION  ({n_per_class} example/class few-shot)")
    print(f"{'='*60}")
    print(f"  Valid predictions:   {len(valid)} / {len(records)}")
    print(f"  Unparseable:         {n_invalid}")
    print(f"  Balanced Accuracy:   {bal_acc:.4f}  ({bal_acc:.2%})")
    print(f"  Zero-shot baseline:  {GEMINI_ZERO_SHOT:.2%}")
    print(f"  Delta:               {bal_acc - GEMINI_ZERO_SHOT:+.2%}")
    print()
    print(classification_report(true_idx, pred_idx, target_names=CLASS_NAMES, zero_division=0))

    # Confusion matrix
    cm      = confusion_matrix(true_idx, pred_idx, labels=list(range(4)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    prompt_wrap = '\n'.join(textwrap.wrap(PROMPT_QUERY.replace('\n', ' '), width=70))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Gemini Few-Shot — Which-LSB  |  Bal. Acc: {bal_acc:.1%}\n"
        f"({n_per_class} example/class, model={MODEL_NAME})",
        fontsize=10
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i,j] > 0.55 else "black")
    plt.colorbar(im, ax=ax)
    fig.text(0.5, 0.01, f"Prompt: {prompt_wrap}",
             ha="center", va="bottom", fontsize=6, style="italic",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    cm_path = output_csv.replace(".csv", "_confusion.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      default="which_lsb_fewshot_results.csv")
    parser.add_argument("--thinking",    default="minimal",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--n_per_class", type=int, default=1,
                        help="Examples per class shown to Gemini")
    args = parser.parse_args()

    run_eval(args.output, args.thinking, args.n_per_class)
