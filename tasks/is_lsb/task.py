"""is-lsb — binary detection of low-surface-brightness tidal debris.

Dataset: mwalmsley/is-lsb (HuggingFace), test split.
Labels:  0 = not_tidal, 1 = tidal.
"""

from functools import partial
from pathlib import Path

from vlm_eval import TaskSpec
from vlm_eval.catalogs import build_hf_catalog, load_catalog
from vlm_eval.sampling import sample_balanced

TASK_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TASK_DIR / "data" / "images" / "test"
CATALOG_PATH = TASK_DIR / "data" / "test_catalog.csv"

CLASS_NAMES = ["not_tidal", "tidal"]  # index order matches dataset label ints

PROMPT = (
    "Does this galaxy image show faint debris (low-surface-brightness features "
    "around the main galaxy body)? Reply with one word: yes or no."
)


def parse_prediction(text):
    t = text.strip().lower()
    if t.startswith("no"):
        return "not_tidal"
    if t.startswith("yes"):
        return "tidal"
    return None


TASK = TaskSpec(
    name="is_lsb",
    label="is-lsb",
    class_names=CLASS_NAMES,
    prompt=PROMPT,
    dir=TASK_DIR,
    load_catalog=partial(load_catalog, CATALOG_PATH),
    build_catalog=partial(
        build_hf_catalog, "mwalmsley/is-lsb", IMAGES_DIR, CATALOG_PATH,
        class_names=CLASS_NAMES,
        id_fn=lambda ex, i: ex["id_str"],
    ),
    parse_prediction=parse_prediction,
    get_sample=sample_balanced,
)
