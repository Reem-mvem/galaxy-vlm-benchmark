"""strong_lens — binary strong-gravitational-lens detection on Euclid images.

Dataset: mwalmsley/euclid_strong_lens_expert_judges ("classification" config),
test split. Labels: 1 = grade A or B lens candidate (expert-judged).
Positives are rare, so sampling keeps every positive and tops up with
random negatives.
"""

from functools import partial
from pathlib import Path

from vlm_eval import TaskSpec
from vlm_eval.catalogs import build_hf_catalog, load_catalog
from vlm_eval.sampling import sample_minority_topup

TASK_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TASK_DIR / "data" / "images" / "test"
CATALOG_PATH = TASK_DIR / "data" / "test_catalog.csv"

CLASS_NAMES = ["not_grade_a_or_b_lens", "grade_a_or_b_lens"]

PROMPT = (
    "This image is from the Euclid space telescope. Does it show a strong "
    "gravitational lens (a confident or probable lens candidate, as an expert "
    "would judge)? Reply with one word: yes or no."
)


def parse_prediction(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "grade_a_or_b_lens"
    if t.startswith("no"):
        return "not_grade_a_or_b_lens"
    return None


TASK = TaskSpec(
    name="strong_lens",
    label="Euclid Strong Lenses",
    class_names=CLASS_NAMES,
    prompt=PROMPT,
    dir=TASK_DIR,
    load_catalog=partial(load_catalog, CATALOG_PATH),
    build_catalog=partial(
        build_hf_catalog, "mwalmsley/euclid_strong_lens_expert_judges",
        IMAGES_DIR, CATALOG_PATH,
        hf_config="classification",
        class_names=CLASS_NAMES,
        filename_fn=lambda ex, i: f"{i}.jpg",
    ),
    parse_prediction=parse_prediction,
    get_sample=sample_minority_topup,
)
