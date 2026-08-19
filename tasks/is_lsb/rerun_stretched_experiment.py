"""Experiment: does an asinh stretch recover Gemini's is-lsb mistakes?

Take 20 false negatives (true=tidal, Gemini said not_tidal) and 20 false
positives (true=not_tidal, Gemini said tidal) from the stored Gemini results,
apply an asinh stretch (a=0.1, per channel), and re-query Gemini with the exact
production call. "Before" = the stored (wrong) prediction; "after" = the
prediction on the stretched image. Report how many flip to correct.

No change to the core pipeline: is-lsb uses prepare_image=read_image_bytes (raw
bytes, no normalization), so the stretched JPEG bytes are a faithful drop-in.
"""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from task import TASK
from vlm_eval.providers import get_provider

IMG_DIR = Path(r"C:\Users\reem2\Desktop\geir-local\Gz evo\images\is_lsb\test")
RESULTS = Path("results/is_lsb_results_gemini.csv")
STRETCHED_DIR = Path("asinh_rerun_a0.1"); STRETCHED_DIR.mkdir(exist_ok=True)
OUT_CSV = Path("results/is_lsb_stretch_rerun_a0.1.csv")
A = 0.1
MODEL = "gemini-3.5-flash"
N_EACH = 20


def asinh_jpeg_bytes(img_path, a=A, quality=95):
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
    stretched = np.clip(np.arcsinh(arr / a) / np.arcsinh(1.0 / a), 0, 1)
    pil = Image.fromarray((stretched * 255).astype(np.uint8))
    buf = BytesIO(); pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), pil


def main():
    df = pd.read_csv(RESULTS)
    fn = df[(df.true_name == "tidal") & (df.pred_name == "not_tidal")].head(N_EACH)
    fp = df[(df.true_name == "not_tidal") & (df.pred_name == "tidal")].head(N_EACH)
    sel = pd.concat([fn, fp]).reset_index(drop=True)
    print(f"Selected {len(fn)} false negatives + {len(fp)} false positives = {len(sel)}")

    prov = get_provider("gemini", MODEL, thinking=None)
    rows = []
    for i, r in sel.iterrows():
        ids, true, before = r.id_str, r.true_name, r.pred_name
        img_path = IMG_DIR / f"{ids}.jpg"
        rows.append(_run_one(prov, ids, true, before, img_path))
        print(f"[{i+1}/{len(sel)}] {ids[:12]} true={true:9s} before={before:9s} "
              f"after={rows[-1]['after_pred']}  {'FLIP-correct' if rows[-1]['after_correct'] else ''}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    summarize(out)


def _run_one(prov, ids, true, before, img_path):
    img_bytes, pil = asinh_jpeg_bytes(img_path)
    pil.save(STRETCHED_DIR / f"{ids[:16]}.jpg", quality=95)
    text, _ = prov.query(img_bytes, TASK.prompt)
    after = TASK.parse_prediction(text)
    return {"id_str": ids, "true_name": true, "before_pred": before,
            "after_pred": after, "raw_after": (text or "").strip().replace("\n", " ")[:80],
            "after_correct": (after == true), "error_type":
            "false_negative" if true == "tidal" else "false_positive"}


def summarize(out):
    print("\n" + "=" * 60)
    for et, label in [("false_negative", "FN (true tidal, was not_tidal)"),
                      ("false_positive", "FP (true not_tidal, was tidal)")]:
        s = out[out.error_type == et]
        rec = s.after_correct.sum()
        blank = s.after_pred.isna().sum()
        print(f"{label}: {rec}/{len(s)} recovered "
              f"({rec/len(s)*100:.0f}%)   [blank/unparsed after: {blank}]")
    tot = out.after_correct.sum()
    print(f"OVERALL: {tot}/{len(out)} recovered ({tot/len(out)*100:.0f}%)")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
