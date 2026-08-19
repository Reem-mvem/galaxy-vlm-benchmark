#!/usr/bin/env python
"""
Score Gemini and the fine-tuned CNN on the same GZ Euclid sample, with
identical rules for both:

  relevant   question routed to >= 50% of the galaxy's volunteers
  correct    predicted answer == volunteers' majority answer (argmax of fractions)
  subsets    all-relevant, and confident (majority fraction >= 0.8)

Gemini answers that are missing/invalid count as WRONG (it had its chance).
CNN accuracy is mean +/- std over the 6 seeds, recomputed on the sample ids.

Outputs:
  gz_euclid_gemini_pernode_accuracy.csv   question,subset,acc,n_galaxies
  gz_euclid_cnn_pernode_accuracy_sample.csv  question,subset,acc_mean,acc_std,n_galaxies
  printed side-by-side table + overall micro/macro accuracies

Run:  python score_euclid.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_schema import (
    QUESTIONS, BUNDLE_PREDICTIONS,
    relevance_mask, true_answers, true_confidence, cnn_answers, load_bundle_seed,
)

HERE = Path(__file__).resolve().parent

SUBSETS = ["all-relevant", "confident(maj>=0.8)"]


def subset_masks(truth_df, question):
    relevant = relevance_mask(truth_df, question)
    confident = relevant & (true_confidence(truth_df, question) >= 0.8)
    return {"all-relevant": relevant, "confident(maj>=0.8)": confident}


def per_question_accuracy(truth_df, pred_answers):
    """pred_answers: {question: Series of predicted answer strings (aligned to truth_df)}.
    Returns rows of (question, subset, acc, n) and per-subset (correct, n) totals."""
    rows = []
    totals = {s: [0, 0] for s in SUBSETS}  # subset -> [n_correct, n]
    for q in QUESTIONS:
        truth = true_answers(truth_df, q)
        correct = (pred_answers[q] == truth)
        for subset, mask in subset_masks(truth_df, q).items():
            n = int(mask.sum())
            n_correct = int(correct[mask].sum())
            rows.append({"question": q, "subset": subset,
                         "acc": n_correct / n if n else np.nan, "n_galaxies": n})
            totals[subset][0] += n_correct
            totals[subset][1] += n
    return rows, totals


def overall(rows, totals):
    out = {}
    for subset in SUBSETS:
        accs = [r["acc"] for r in rows if r["subset"] == subset and r["n_galaxies"] > 0]
        micro = totals[subset][0] / totals[subset][1]
        out[subset] = {"micro": micro, "macro": float(np.mean(accs))}
    return out


def score_gemini(truth_df, gemini_csv):
    g = pd.read_csv(gemini_csv, dtype=str).drop_duplicates("id_str", keep="last")
    g = g.set_index("id_str")
    missing_ids = truth_df.index.difference(g.index)
    if len(missing_ids):
        print(f"WARNING: {len(missing_ids)} sample galaxies have no Gemini row yet "
              f"(rerun euclid_eval_gemini.py) — scoring only the {len(g)} done.")
        truth_df = truth_df.loc[truth_df.index.intersection(g.index)]
    g = g.loc[truth_df.index]
    preds = {q: g[q].fillna("__invalid__") for q in QUESTIONS}
    rows, totals = per_question_accuracy(truth_df, preds)
    return rows, overall(rows, totals), truth_df


def score_cnn(truth_ids, model):
    seed_dirs = sorted(p for p in (BUNDLE_PREDICTIONS / model).iterdir() if p.is_dir())
    per_seed_rows = []
    per_seed_overall = []
    for sd in seed_dirs:
        df = pd.read_csv(sd / "predictions_test.csv").set_index("id_str")
        df = df.loc[truth_ids]
        preds = {q: cnn_answers(df, q) for q in QUESTIONS}
        rows, totals = per_question_accuracy(df, preds)
        per_seed_rows.append(pd.DataFrame(rows))
        per_seed_overall.append(overall(rows, totals))

    stacked = pd.concat(per_seed_rows)
    agg = (stacked.groupby(["question", "subset"], sort=False)
           .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                n_galaxies=("n_galaxies", "first"))
           .reset_index())
    ov = {s: {m: float(np.mean([o[s][m] for o in per_seed_overall]))
              for m in ("micro", "macro")} for s in SUBSETS}
    return agg, ov, len(seed_dirs)


def main(gemini_csv, ids_path, model):
    sample_ids = [l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()]
    print(f"{len(sample_ids)} galaxies in sample")

    # ground truth: fractions/votes from bundle seed_0 (identical across seeds)
    truth_df = load_bundle_seed(model=model).set_index("id_str").loc[sample_ids]

    print(f"\nScoring Gemini: {gemini_csv}")
    g_rows, g_overall, truth_scored = score_gemini(truth_df, gemini_csv)
    g_df = pd.DataFrame(g_rows)
    g_out = HERE / "gz_euclid_gemini_pernode_accuracy.csv"
    g_df.to_csv(g_out, index=False)
    print(f"Saved {g_out}")

    print(f"\nScoring CNN ({model}) on the same {len(truth_scored)} galaxies...")
    c_df, c_overall, n_seeds = score_cnn(truth_scored.index, model)
    c_out = HERE / "gz_euclid_cnn_pernode_accuracy_sample.csv"
    c_df.to_csv(c_out, index=False)
    print(f"Saved {c_out}  ({n_seeds} seeds)")

    # side-by-side
    merged = g_df.merge(c_df, on=["question", "subset"], suffixes=("_gemini", "_cnn"))
    for subset in SUBSETS:
        sub = merged[merged["subset"] == subset]
        print(f"\n{'='*72}\n{subset}\n{'='*72}")
        print(f"{'question':22s} {'n':>5s} {'gemini':>8s} {'cnn':>8s} {'(+/-)':>7s}")
        for _, r in sub.iterrows():
            print(f"{r['question']:22s} {r['n_galaxies_gemini']:5d} "
                  f"{r['acc']:8.1%} {r['acc_mean']:8.1%} {r['acc_std']:7.1%}")
        print(f"{'-'*52}")
        print(f"{'OVERALL micro':22s} {'':5s} {g_overall[subset]['micro']:8.1%} "
              f"{c_overall[subset]['micro']:8.1%}")
        print(f"{'OVERALL macro':22s} {'':5s} {g_overall[subset]['macro']:8.1%} "
              f"{c_overall[subset]['macro']:8.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_csv", default=str(HERE / "gemini_euclid_predictions.csv"))
    parser.add_argument("--ids",        default=str(HERE / "euclid_sample_ids.txt"))
    # maxvit_base is the model behind zoobot_project/gz_euclid_cnn_pernode_accuracy.csv
    # (verified: this script reproduces that CSV to float precision on the full test set)
    parser.add_argument("--model",      default="gz_euclid_maxvit_base",
                        choices=["gz_euclid_convnext_base", "gz_euclid_maxvit_base"])
    args = parser.parse_args()
    main(args.gemini_csv, args.ids, args.model)
