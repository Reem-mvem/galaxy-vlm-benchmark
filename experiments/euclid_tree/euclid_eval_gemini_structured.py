#!/usr/bin/env python
"""
Gemini zero-shot on the GZ Euclid 13-question decision tree — STRUCTURED OUTPUT.

Same design as euclid_eval_gemini.py (one call per galaxy, all 13 questions
answered unconditionally, resumable CSV) but instead of asking for free-form
JSON and regex-parsing it, a Pydantic model with Literal fields is passed as
the response_schema. The API then constrains decoding to the schema, so every
answer is guaranteed to be one of the allowed options — no invalid/missing
values, no parse failures.

Run:  python euclid_eval_gemini_structured.py                (full sample)
      python euclid_eval_gemini_structured.py --limit 5      (smoke test)
Score: python score_euclid.py --gemini_csv gemini_euclid_predictions_structured.csv
"""

import argparse
import csv
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Literal, get_args

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

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


class EuclidDecisionTree(BaseModel):
    """All 13 GZ Euclid decision-tree questions, answered unconditionally.

    Field order matters: Gemini fills the schema in declaration order, so the
    root question comes first and dependent questions follow their parent.
    """

    smooth_or_featured: Literal["smooth", "featured-or-disk", "problem"] = Field(
        ..., description="Is the galaxy smooth and featureless, does it have features or a disk, or is there a problem with the image?"
    )
    how_rounded: Literal["round", "in-between", "cigar-shaped"] = Field(
        ..., description="How rounded is the galaxy? (asked for smooth galaxies; give best guess regardless)"
    )
    disk_edge_on: Literal["yes", "no"] = Field(
        ..., description="Is the galaxy a disk viewed exactly edge-on?"
    )
    edge_on_bulge: Literal["boxy", "none", "rounded"] = Field(
        ..., description="For an edge-on disk, what shape is the central bulge? (give best guess even if not edge-on)"
    )
    has_spiral_arms: Literal["yes", "no"] = Field(
        ..., description="Are there spiral arms?"
    )
    spiral_winding: Literal["tight", "medium", "loose"] = Field(
        ..., description="How tightly wound are the spiral arms? (give best guess even if no arms)"
    )
    spiral_arm_count: Literal["1", "2", "3", "4", "more-than-4", "cant-tell"] = Field(
        ..., description="How many spiral arms? (give best guess even if no arms)"
    )
    bar: Literal["strong", "weak", "no"] = Field(
        ..., description="Is there a bar structure through the centre of the galaxy?"
    )
    bulge_size: Literal["dominant", "large", "moderate", "small", "none"] = Field(
        ..., description="How large is the central bulge compared to the galaxy?"
    )
    merging: Literal["none", "minor-disturbance", "major-disturbance", "merger"] = Field(
        ..., description="Is the galaxy disturbed, interacting, or merging?"
    )
    clumps: Literal["yes", "no"] = Field(
        ..., description="Are there bright clumps visible in the galaxy?"
    )
    problem: Literal["star", "artifact", "zoom"] = Field(
        ..., description="If there is a problem with the image, what kind? (give best guess even if no problem)"
    )
    artifact: Literal["satellite", "scattered", "diffraction", "ray", "saturation", "other", "ghost"] = Field(
        ..., description="If an artifact, what kind? (give best guess even if no artifact)"
    )


# question name in euclid_schema  ->  pydantic field name
FIELD_OF_QUESTION = {q: q.replace("-", "_") for q in QUESTIONS}

# fail fast if this schema ever drifts out of sync with euclid_schema.QUESTIONS
for _q, _allowed in QUESTIONS.items():
    _lit = EuclidDecisionTree.model_fields[FIELD_OF_QUESTION[_q]].annotation
    assert list(get_args(_lit)) == _allowed, f"schema mismatch for {_q}: {get_args(_lit)} != {_allowed}"


PROMPT = """You are analysing an image of a galaxy from the Euclid space telescope.
Answer ALL of the questions in the response schema about its visual morphology,
exactly as a Galaxy Zoo volunteer would. If a question does not apply to this
galaxy (e.g. spiral questions for a smooth galaxy), still give your best guess."""


def image_to_jpeg_bytes(pil_image):
    img = pil_image.convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


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
    n_ok = 0

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
                        response_schema=EuclidDecisionTree,
                    ),
                )
                tree = response.parsed
                if tree is None:  # rare: fall back to validating the raw text
                    tree = EuclidDecisionTree.model_validate_json(response.text)

                usage = response.usage_metadata
                total_in    += usage.prompt_token_count or 0
                total_out   += usage.candidates_token_count or 0
                total_think += usage.thoughts_token_count or 0
                n_ok += 1

                row = {"id_str": id_str,
                       "raw_response": tree.model_dump_json()}
                row.update({q: getattr(tree, field)
                            for q, field in FIELD_OF_QUESTION.items()})
                writer.writerow(row)
                f.flush()

                print(f"-> {tree.smooth_or_featured}")
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
    print(f"\n  Galaxies answered (all guaranteed valid): {n_ok}")
    print(f"\nNext: python score_euclid.py --gemini_csv {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default=str(HERE / "gemini_euclid_predictions_structured.csv"))
    parser.add_argument("--ids",      default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--thinking", default="minimal",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--limit",    type=int, default=None,
                        help="Only run the first N sample ids (smoke test)")
    parser.add_argument("--sleep",    type=float, default=0.5,
                        help="Seconds to sleep between API calls")
    args = parser.parse_args()
    run_eval(args.output, args.ids, args.thinking, args.limit, args.sleep)
