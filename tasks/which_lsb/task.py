"""which-lsb — 4-class morphology of low-surface-brightness tidal debris.

Dataset: mwalmsley/which-lsb (HuggingFace), test split (218 images).
Labels:  arm / stream / shell / diffuse. The full test set is small enough
to always evaluate completely.
"""

import re
from functools import partial
from pathlib import Path

from vlm_eval import TaskSpec
from vlm_eval.catalogs import build_hf_catalog, load_catalog

TASK_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TASK_DIR / "data" / "images" / "test"
CATALOG_PATH = TASK_DIR / "data" / "test_catalog.csv"

CLASS_NAMES = ["arm", "stream", "shell", "diffuse"]

PROMPT = (
    "This galaxy image shows faint debris. Classify the debris morphology. "
    "Reply with one word: arm, stream, shell, or diffuse."
)


def parse_prediction(text):
    text = text.strip().lower()
    for cls in CLASS_NAMES:
        if re.search(rf"\b{cls}\b", text):
            return cls
    return None


TASK = TaskSpec(
    name="which_lsb",
    label="which-lsb",
    class_names=CLASS_NAMES,
    prompt=PROMPT,
    dir=TASK_DIR,
    load_catalog=partial(load_catalog, CATALOG_PATH),
    build_catalog=partial(
        build_hf_catalog, "mwalmsley/which-lsb", IMAGES_DIR, CATALOG_PATH,
        class_names=CLASS_NAMES,
        id_fn=lambda ex, i: f"test_{i}",
        filename_fn=lambda ex, i: f"{i}.jpg",
    ),
    parse_prediction=parse_prediction,
)
