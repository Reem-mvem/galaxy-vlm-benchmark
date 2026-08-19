#!/usr/bin/env python
"""
Gemini zero-shot on the GZ Euclid 13-question decision tree.

One API call per galaxy: the image plus all 13 questions, answered
unconditionally (relevance is decided by the volunteers' routing at scoring
time, so Gemini must answer everything — unused answers are simply ignored).

Resumable: rows are appended to the output CSV as they arrive; rerunning
skips ids already present. Score afterwards with score_euclid.py.

Run:  python euclid_eval_gemini.py                  (full sample)
      python euclid_eval_gemini.py --limit 10       (smoke test)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai

from euclid_schema import QUESTIONS, ROOT

load_dotenv()
load_dotenv(ROOT / "zoobot_project" / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
    sys.exit(1)

HF_DATASET = "mwalmsley/gz_euclid"
MODEL_NAME = "gemini-3.5-flash"
IMG_SIZE = 424
HERE = Path(__file__).resolve().parent
N_AUDIT_IMAGES = 15

PROMPT = """You are analysing an image of a galaxy from the Euclid space telescope.
Answer ALL of the following questions about its visual morphology. If a
question does not apply to this galaxy (e.g. spiral questions for a smooth
galaxy), still give your best guess.

Q1  smooth-or-featured: Is the galaxy smooth and featureless, does it have
    features or a disk, or is there a problem with the image?
    [smooth | featured-or-disk | problem]
Q2  disk-edge-on: Is the galaxy a disk viewed exactly edge-on?  [yes | no]
Q3  has-spiral-arms: Are there spiral arms?  [yes | no]
Q4  bar: Is there a bar structure through the centre?  [strong | weak | no]
Q5  bulge-size: How large is the central bulge compared to the galaxy?
    [dominant | large | moderate | small | none]
Q6  how-rounded: How rounded is the galaxy?  [round | in-between | cigar-shaped]
Q7  edge-on-bulge: For an edge-on disk, what shape is the bulge?
    [boxy | none | rounded]
Q8  spiral-winding: How tightly wound are the spiral arms?
    [tight | medium | loose]
Q9  spiral-arm-count: How many spiral arms?  [1 | 2 | 3 | 4 | more-than-4 | cant-tell]
Q10 merging: Is the galaxy disturbed or merging?
    [none | minor-disturbance | major-disturbance | merger]
Q11 clumps: Are there bright clumps in the galaxy?  [yes | no]
Q12 problem: If there is a problem, what kind?  [star | artifact | zoom]
Q13 artifact: If an artifact, what kind?
    [satellite | scattered | diffraction | ray | saturation | other | ghost]

Reply with ONLY a JSON object, one key per question (use the exact question
names above as keys), values chosen exactly from the options given, e.g.:
{"smooth-or-featured": "featured-or-disk", "disk-edge-on": "no", ...}"""


def image_to_jpeg_bytes(pil_image):
    img = pil_image.convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def parse_answers(raw_text):
    """Parse Gemini's JSON reply -> {question: answer-or-None}."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {q: None for q in QUESTIONS}
    answers = {}
    for q, allowed in QUESTIONS.items():
        val = obj.get(q)
        val = str(val).strip().lower() if val is not None else None
        answers[q] = val if val in allowed else None
    return answers


def load_done_ids(output_csv):
    if not os.path.exists(output_csv):
        return set()
    with open(output_csv, newline="", encoding="utf-8") as f:
        return {row["id_str"] for row in csv.DictReader(f)}


def run_eval(output_csv, ids_path, thinking_level, limit, sleep_s):
    with open(ids_path, encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    if limit:
        sample_ids = sample_ids[:limit]

    done = load_done_ids(output_csv)
    todo = [i for i in sample_ids if i not in done]
    print(f"{len(sample_ids)} sample ids, {len(done)} already done, {len(todo)} to run")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading {HF_DATASET} test split (from HF cache)...")
    ds = load_dataset(HF_DATASET, split="test", token=os.getenv("HF_TOKEN"))
    row_of_id = {id_str: i for i, id_str in enumerate(ds["id_str"])}
    missing = [i for i in todo if i not in row_of_id]
    if missing:
        print(f"ERROR: {len(missing)} sample ids not in dataset, e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)

    audit_dir = HERE / "sent_images_sample"
    audit_dir.mkdir(exist_ok=True)

    client = genai.Client(api_key=API_KEY)
    fieldnames = ["id_str"] + list(QUESTIONS) + ["raw_response"]
    write_header = not os.path.exists(output_csv)

    total_in = total_out = total_think = 0
    n_ok = n_partial = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, id_str in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {id_str[:16]}...", end=" ", flush=True)
            try:
                img_bytes = image_to_jpeg_bytes(ds[row_of_id[id_str]]["image"])
                if i < N_AUDIT_IMAGES:
                    (audit_dir / f"{id_str[:16]}.jpg").write_bytes(img_bytes)

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        PROMPT,
                    ],
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(
                            thinking_level=thinking_level.upper()
                        ),
                        response_mime_type="application/json",
                    ),
                )
                raw_text = response.text if hasattr(response, "text") else str(response)
                answers = parse_answers(raw_text)

                usage = response.usage_metadata
                total_in    += usage.prompt_token_count or 0
                total_out   += usage.candidates_token_count or 0
                total_think += usage.thoughts_token_count or 0

                n_missing = sum(v is None for v in answers.values())
                if n_missing == 0:
                    n_ok += 1
                else:
                    n_partial += 1

                row = {"id_str": id_str, "raw_response": raw_text.strip().replace("\n", " ")}
                row.update({q: (a if a is not None else "") for q, a in answers.items()})
                writer.writerow(row)
                f.flush()

                print(f"-> {answers['smooth-or-featured'] or 'INVALID'}"
                      + (f"  ({n_missing} invalid answers)" if n_missing else ""))
                time.sleep(sleep_s)

            except KeyboardInterrupt:
                print("\nInterrupted — progress is saved, rerun to resume.")
                break
            except Exception as e:
                print(f"\n  ERROR (skipping, rerun to retry): {e}", file=sys.stderr)
                time.sleep(2.0)

    print(f"\n{'='*60}")
    print(f"TOKEN USAGE  (model={MODEL_NAME}, thinking={thinking_level})")
    print(f"{'='*60}")
    print(f"  Input tokens:    {total_in:,}")
    print(f"  Output tokens:   {total_out:,}")
    print(f"  Thinking tokens: {total_think:,}")
    print(f"\n  Fully parsed galaxies:     {n_ok}")
    print(f"  Galaxies w/ invalid values: {n_partial}")
    print(f"\nNext: python score_euclid.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default=str(HERE / "gemini_euclid_predictions.csv"))
    parser.add_argument("--ids",      default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--thinking", default="minimal",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--limit",    type=int, default=None,
                        help="Only run the first N sample ids (smoke test)")
    parser.add_argument("--sleep",    type=float, default=0.5,
                        help="Seconds to sleep between API calls")
    args = parser.parse_args()
    run_eval(args.output, args.ids, args.thinking, args.limit, args.sleep)
