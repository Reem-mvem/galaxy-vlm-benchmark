#!/usr/bin/env python
"""
Few-shot Gemini evaluation on Euclid Strong Lenses.

Shows 1 lens + 1 not-lens example (from the training set)
before each query. All 500 test samples are available for evaluation.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from google import genai

from utils import (
    CATALOG_PATH,
    build_catalog, load_catalog, parse_prediction,
    get_sample, save_results_csv, compute_metrics_and_plot,
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
    sys.exit(1)

MODEL_NAME        = "gemini-2.0-flash"
TRAIN_CATALOG     = "../strong_lens_train_catalog.csv"

# Fixed training examples chosen by the user
EXAMPLE_NOT_LENS_ID = "102044826_NEG540292237276624842"   # class 0
EXAMPLE_LENS_ID     = "102042917_NEG539279061291936993"   # class 1

CAPTION_LENS     = "EXAMPLE — LENS: this Euclid image is a strong gravitational lens candidate, as judged by human experts."
CAPTION_NOT_LENS = "EXAMPLE — NOT LENS: this Euclid image is not a strong gravitational lens, as judged by human experts."

PROMPT_QUERY = """Study the two labeled examples above carefully, then answer for this new Euclid image:
This image is from the Euclid space telescope. Does it show a strong gravitational lens (a confident or probable lens candidate, as an expert would judge)? Reply with one word: yes or no?

Reply with one word: yes or no."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_examples():
    train = pd.read_csv(TRAIN_CATALOG)
    train["file_loc"] = train["file_loc"].apply(
        lambda p: str(PROJECT_ROOT / p)
    )
    not_lens = train[train["id_str"] == EXAMPLE_NOT_LENS_ID]
    lens     = train[train["id_str"] == EXAMPLE_LENS_ID]

    missing = []
    if not_lens.empty:
        missing.append(EXAMPLE_NOT_LENS_ID)
    if lens.empty:
        missing.append(EXAMPLE_LENS_ID)
    if missing:
        print(f"ERROR: IDs not found in training catalog: {missing}", file=sys.stderr)
        sys.exit(1)

    examples_df = pd.concat([not_lens, lens]).reset_index(drop=True)
    print("Few-shot examples (from training set):")
    for _, row in examples_df.iterrows():
        print(f"  {row['id_str']}  [{row['label_name']}]")
    return examples_df


def build_contents(examples_df, test_img_bytes):
    contents = []
    for _, row in examples_df.iterrows():
        img_bytes = open(row["file_loc"], "rb").read()
        contents.append(genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        if row["label_name"] == "grade_a_or_b_lens":
            contents.append(CAPTION_LENS)
        else:
            contents.append(CAPTION_NOT_LENS)
    contents.append(genai.types.Part.from_bytes(data=test_img_bytes, mime_type="image/jpeg"))
    contents.append(PROMPT_QUERY)
    return contents


def save_example_grid(examples_df, out_path):
    n = len(examples_df)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Few-shot examples passed to Gemini (from training set)", fontweight="bold")
    colors = {"grade_a_or_b_lens": "#009E73", "not_grade_a_or_b_lens": "#CC79A7"}
    labels = {"grade_a_or_b_lens": "LENS", "not_grade_a_or_b_lens": "NOT LENS"}
    for ax, (_, row) in zip(axes, examples_df.iterrows()):
        ax.imshow(Image.open(row["file_loc"]))
        ax.axis("off")
        cls = row["label_name"]
        ax.set_title(labels[cls], fontsize=10, color=colors[cls], fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved example grid → {out_path}")


def run_eval(output_csv, n_sample=500, seed=42):
    catalog     = build_catalog() if not os.path.exists(CATALOG_PATH) else load_catalog()
    examples_df = load_examples()
    sample      = get_sample(catalog, n_sample, seed)
    print(f"\nEvaluating {len(sample)} test images")

    save_example_grid(examples_df, output_csv.replace(".csv", "_examples.png"))

    client  = genai.Client(api_key=API_KEY)
    records = []
    n_total = len(sample)
    tot_in = tot_out = tot_think = 0

    try:
        for i, (_, row) in enumerate(sample.iterrows()):
            img_path = Path(row["file_loc"])
            if not img_path.exists():
                print(f"[{i+1}/{n_total}] SKIPPED: {img_path}", file=sys.stderr)
                continue
            print(f"[{i+1}/{n_total}] {row['id_str']} (true={row['label_name']})",
                  end=" ", flush=True)
            try:
                test_bytes = open(str(img_path), "rb").read()
                contents   = build_contents(examples_df, test_bytes)
                response   = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(),
                )
                raw_text = response.text if hasattr(response, "text") else str(response)
                pred     = parse_prediction(raw_text)
                usage    = response.usage_metadata
                in_tok   = usage.prompt_token_count     or 0
                out_tok  = usage.candidates_token_count or 0
                th_tok   = usage.thoughts_token_count   or 0
                tot_in += in_tok; tot_out += out_tok; tot_think += th_tok
                records.append({"id_str": row["id_str"], "true_name": row["label_name"],
                                "pred_name": pred, "raw_response": raw_text.strip()})
                print(f"→ pred={str(pred):25s} [in={in_tok}]")
                time.sleep(0.5)
            except Exception as e:
                print(f"\n  ERROR: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n\nStopped early — saving {len(records)} records...")

    if not records:
        print("No results.")
        return
    save_results_csv(records, output_csv)
    print(f"\nToken usage — in: {tot_in:,}  out: {tot_out:,}  think: {tot_think:,}")
    CLASS_NAMES = ["not_grade_a_or_b_lens", "grade_a_or_b_lens"]
    compute_metrics_and_plot(records, CLASS_NAMES, MODEL_NAME, "Gemini Few-Shot", output_csv, prompt=PROMPT_QUERY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="strong_lens_results_fewshot.csv")
    parser.add_argument("--model",    default=MODEL_NAME)
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    MODEL_NAME = args.model
    run_eval(args.output, args.n_sample, args.seed)
