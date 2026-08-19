"""streams_shells_neither — 3-class tidal-feature morphology (real-world application task).

Dataset: astronolan/streams_shells_neither (HuggingFace) — DESI Legacy Survey DR10 cutouts.
Labels:  stream / shell / neither  (neither = hard-negative galaxies).
Splits:  train 72 (29 stream / 24 shell / 19 neither), test 30 (10 / 10 / 10, balanced).
Images built by build_data.py (uses the image_rgb column). This is the mentor's real task —
a live test of the label-efficient VLM pipeline on a new survey (domain shift from Galaxy Zoo).
"""

import re
from functools import partial
from pathlib import Path

from vlm_eval import TaskSpec
from vlm_eval.catalogs import load_catalog

TASK_DIR = Path(__file__).resolve().parent
CATALOG_PATH = TASK_DIR / "data" / "test_catalog.csv"

CLASS_NAMES = ["stream", "shell", "neither"]

PROMPT = (
    "This is a galaxy image from an astronomical survey. Does it show a stellar stream, "
    "a shell, or neither? Reply with one word: stream, shell, or neither."
)


def parse_prediction(text):
    text = text.strip().lower()
    for cls in CLASS_NAMES:
        if re.search(rf"\b{cls}\b", text):
            return cls
    # common ways the model expresses the negative class
    if re.search(r"\b(none|no|neither|normal|nothing|absent)\b", text):
        return "neither"
    return None


def _build(*a, **k):
    raise SystemExit("Build this task's data with tasks/streams_shells_neither/build_data.py "
                     "(it uses the image_rgb column + writes train/test catalogs).")


TASK = TaskSpec(
    name="streams_shells_neither",
    label="streams-shells-neither",
    class_names=CLASS_NAMES,
    prompt=PROMPT,
    dir=TASK_DIR,
    load_catalog=partial(load_catalog, CATALOG_PATH),
    build_catalog=_build,
    parse_prediction=parse_prediction,
)
