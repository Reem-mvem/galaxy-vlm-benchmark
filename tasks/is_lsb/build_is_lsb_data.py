"""Build is-lsb test + train catalogs from HF mwalmsley/is-lsb (binary:
not_tidal / tidal). Overwrites the committed test_catalog.csv (which carried
dead Windows file paths) with locally-downloaded images + local file_loc.

TEST : the full 863-image test split (id = the dataset's id_str hash, matching
       the existing Gemini/OpenAI result CSVs so comparisons still line up).
TRAIN: the full 2013-image train split — a big pool to subsample sweep sizes from.
Images are saved RAW (is_lsb uses no percentile stretch; task.py sets no prepare_image).
"""
from pathlib import Path
from vlm_eval.catalogs import build_hf_catalog

T = Path(__file__).resolve().parent
HF = "mwalmsley/is-lsb"
CLASS_NAMES = ["not_tidal", "tidal"]          # index order matches dataset label ints
idf = lambda ex, i: ex["id_str"]
fnf = lambda ex, i: f"{ex['id_str']}.jpg"

print("=== TEST (863) ===", flush=True)
build_hf_catalog(HF, T/"data"/"images"/"test", T/"data"/"test_catalog.csv",
                 class_names=CLASS_NAMES, split="test", id_fn=idf, filename_fn=fnf)

print("=== TRAIN pool (2013) ===", flush=True)
build_hf_catalog(HF, T/"data"/"images"/"train", T/"data"/"train_catalog.csv",
                 class_names=CLASS_NAMES, split="train", id_fn=idf, filename_fn=fnf)
print("IS_LSB_BUILD_DONE", flush=True)
