"""Control for the asinh-stretch experiment: re-query Gemini on the SAME 40
mislabeled images, UNSTRETCHED, to measure the baseline flip-to-correct rate
from run-to-run nondeterminism alone.

Faithful to the original eval: uses TASK.prepare_image (read_image_bytes -> raw
JPEG bytes), same model, same prompt. Compare this recovery rate against the
stretched run (is_lsb_stretch_rerun_a0.1.csv); the difference is the real
stretch effect.
"""

from pathlib import Path

import pandas as pd

from task import TASK
from vlm_eval.providers import get_provider

IMG_DIR = Path(r"C:\Users\reem2\Desktop\geir-local\Gz evo\images\is_lsb\test")
RESULTS = Path("results/is_lsb_results_gemini.csv")
OUT_CSV = Path("results/is_lsb_control_rerun.csv")
MODEL = "gemini-3.5-flash"
N_EACH = 20


def main():
    df = pd.read_csv(RESULTS)
    fn = df[(df.true_name == "tidal") & (df.pred_name == "not_tidal")].head(N_EACH)
    fp = df[(df.true_name == "not_tidal") & (df.pred_name == "tidal")].head(N_EACH)
    sel = pd.concat([fn, fp]).reset_index(drop=True)
    print(f"Control: {len(fn)} false negatives + {len(fp)} false positives = {len(sel)}",
          flush=True)

    prov = get_provider("gemini", MODEL, thinking=None)
    rows = []
    for i, r in sel.iterrows():
        ids, true, before = r.id_str, r.true_name, r.pred_name
        img_bytes = TASK.prepare_image(str(IMG_DIR / f"{ids}.jpg"))  # raw, unstretched
        text, _ = prov.query(img_bytes, TASK.prompt)
        after = TASK.parse_prediction(text)
        rows.append({"id_str": ids, "true_name": true, "before_pred": before,
                     "control_pred": after,
                     "raw_control": (text or "").strip().replace("\n", " ")[:80],
                     "control_correct": (after == true),
                     "error_type": "false_negative" if true == "tidal" else "false_positive"})
        print(f"[{i+1}/{len(sel)}] {ids[:12]} true={true:9s} before={before:9s} "
              f"control={after}  {'FLIP-correct' if rows[-1]['control_correct'] else ''}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("\n" + "=" * 60)
    for et, label in [("false_negative", "FN (true tidal, was not_tidal)"),
                      ("false_positive", "FP (true not_tidal, was tidal)")]:
        s = out[out.error_type == et]
        rec = s.control_correct.sum()
        print(f"{label}: {rec}/{len(s)} flipped correct on plain re-run ({rec/len(s)*100:.0f}%)")
    tot = out.control_correct.sum()
    print(f"OVERALL baseline (nondeterminism): {tot}/{len(out)} ({tot/len(out)*100:.0f}%)")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
