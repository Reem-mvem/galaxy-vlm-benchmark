#!/usr/bin/env python
"""
Shared machinery for the GZ Euclid 13-question decision-tree eval, so every
model (Gemini, OpenAI, Qwen) runs the *same* loop over the *same* sample with
the *same* prompt and writes an identical CSV.

Each provider script supplies a ``query_one(image_bytes, prompt)`` callable and
calls :func:`run_eval`. ``query_one`` returns ``(tree, usage, raw_text)`` where

  tree      an ``EuclidDecisionTree`` instance, or ``None`` if the model's
            output could not be coerced to the schema (recorded as a done-but-
            blank row so it counts as wrong and is not retried forever).
  usage     a ``Usage`` token tally.
  raw_text  the model's raw JSON / text, stored verbatim for auditing.

Transient failures (network, rate limit) should be raised as exceptions — the
loop catches them, skips the galaxy, and picks it up on the next resume run.
"""

import csv
import os
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, get_args

from datasets import load_dataset
from pydantic import BaseModel, Field

from euclid_schema import QUESTIONS, ROOT

HF_DATASET = "mwalmsley/gz_euclid"
IMG_SIZE = 424
HERE = Path(__file__).resolve().parent
N_AUDIT_IMAGES = 15

# One prompt for all three models (chosen 2026-07-20).
PROMPT = (
    "This is an image of a galaxy from the Euclid space telescope. Answer all "
    "of the questions in the response schema about its visual morphology, as a "
    "Galaxy Zoo volunteer would. If a question does not apply to this galaxy "
    "(e.g. spiral questions for a smooth galaxy), still give your best guess."
)


class EuclidDecisionTree(BaseModel):
    """All 13 GZ Euclid decision-tree questions, answered unconditionally.

    Field order matters: providers that fill the schema in declaration order
    (Gemini) see the root question first and dependent questions after their
    parent.
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


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    thinking: int = 0

    def __iadd__(self, other):
        self.input += other.input
        self.output += other.output
        self.thinking += other.thinking
        return self


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


def run_eval(query_one, provider_label, output_csv, ids_path, limit, sleep_s):
    """Drive ``query_one`` over the sample, resumable, writing one CSV row per id."""
    with open(ids_path, encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    if limit:
        sample_ids = sample_ids[:limit]

    done = load_done_ids(output_csv)
    todo = [i for i in sample_ids if i not in done]
    print(f"[{provider_label}] {len(sample_ids)} sample ids, {len(done)} already done, {len(todo)} to run")
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

    audit_dir = HERE / f"sent_images_sample_{provider_label.lower()}"
    audit_dir.mkdir(exist_ok=True)

    fieldnames = ["id_str"] + list(QUESTIONS) + ["raw_response"]
    write_header = not os.path.exists(output_csv)
    total = Usage()
    n_ok = n_bad = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, id_str in enumerate(todo):
            print(f"[{provider_label} {i+1}/{len(todo)}] {id_str[:16]}...", end=" ", flush=True)
            try:
                img_bytes = image_to_jpeg_bytes(ds[row_of_id[id_str]]["image"])
                if i < N_AUDIT_IMAGES:
                    (audit_dir / f"{id_str[:16]}.jpg").write_bytes(img_bytes)

                tree, usage, raw = query_one(img_bytes, PROMPT)
                total += usage

                row = {"id_str": id_str, "raw_response": raw}
                if tree is not None:
                    row.update({q: getattr(tree, field) for q, field in FIELD_OF_QUESTION.items()})
                    n_ok += 1
                    print(f"-> {tree.smooth_or_featured}")
                else:
                    n_bad += 1
                    print("-> UNPARSEABLE (blank row, counts wrong)")
                writer.writerow(row)
                f.flush()
                time.sleep(sleep_s)

            except KeyboardInterrupt:
                print("\nInterrupted — progress is saved, rerun to resume.")
                break
            except Exception as e:
                print(f"\n  ERROR (skipping, rerun to retry): {e}", file=sys.stderr)
                time.sleep(2.0)

    print(f"\n{'='*60}")
    print(f"TOKEN USAGE  ({provider_label})")
    print(f"{'='*60}")
    print(f"  Input tokens:    {total.input:,}")
    print(f"  Output tokens:   {total.output:,}")
    print(f"  Thinking tokens: {total.thinking:,}")
    print(f"  Galaxies answered: {n_ok} valid, {n_bad} unparseable")
    print(f"\nOutput: {output_csv}")
