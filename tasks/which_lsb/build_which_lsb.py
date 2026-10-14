"""Build which-lsb test + train catalogs from HF mwalmsley/which-lsb (4-class).
Test keeps test_<i> ids (row i) so it matches the existing Gemini eval.
Train gets ALL ~508 examples."""
from pathlib import Path
from vlm_eval.catalogs import build_hf_catalog

W = Path(__file__).resolve().parent
CLASS_NAMES = ["arm", "stream", "shell", "diffuse"]
HF = "mwalmsley/which-lsb"

print("=== TEST (218) ===", flush=True)
build_hf_catalog(HF, W/"data"/"images"/"test", W/"data"/"test_catalog.csv",
                 class_names=CLASS_NAMES, split="test",
                 id_fn=lambda ex, i: f"test_{i}", filename_fn=lambda ex, i: f"{i}.jpg")

print("=== TRAIN (all ~508) ===", flush=True)
build_hf_catalog(HF, W/"data"/"images"/"train", W/"data"/"train_catalog.csv",
                 class_names=CLASS_NAMES, split="train",
                 id_fn=lambda ex, i: f"train_{i}", filename_fn=lambda ex, i: f"{i}.jpg")
print("WHICH_LSB_BUILD_DONE", flush=True)
