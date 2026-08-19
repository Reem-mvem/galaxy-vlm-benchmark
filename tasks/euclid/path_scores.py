#!/usr/bin/env python
"""
Per-galaxy ground-truth paths + ONE score per model.

Writes euclid_truth_paths.csv: one row per galaxy, with the volunteers'
majority answer for every question on that galaxy's path (a question is on
the path when >=50% of the galaxy's volunteers were routed to it — same rule
as score_euclid.py), blank for questions off the path.

Score: per galaxy, fraction of its path questions answered correctly;
model score = mean over galaxies (each galaxy counts equally, whether its
path has 3 questions or 8). CNN = mean of the 6 seed scores.

Run:  python path_scores.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_schema import (
    QUESTIONS, BUNDLE_PREDICTIONS,
    relevance_mask, true_answers, true_confidence, cnn_answers, load_bundle_seed,
)

CONFIDENT = 0.8  # volunteer majority fraction for a "confident" answer

HERE = Path(__file__).resolve().parent

# tree traversal order, for the readable path column
TREE_ORDER = [
    "smooth-or-featured", "how-rounded", "clumps", "disk-edge-on",
    "edge-on-bulge", "bar", "has-spiral-arms", "spiral-winding",
    "spiral-arm-count", "bulge-size", "merging", "problem", "artifact",
]
assert set(TREE_ORDER) == set(QUESTIONS)


def build_truth_paths(truth_df):
    """DataFrame: id_str index, one column per question (majority answer on
    path, '' off path), plus a readable 'path' column."""
    out = pd.DataFrame(index=truth_df.index)
    for q in QUESTIONS:
        majority = true_answers(truth_df, q)
        on_path = relevance_mask(truth_df, q)
        out[q] = majority.where(on_path, "")
    out["n_path_questions"] = (out[list(QUESTIONS)] != "").sum(axis=1)
    # galaxy is "fully confident" when every path question's majority >= CONFIDENT
    confident_on_path = pd.DataFrame(index=truth_df.index)
    for q in QUESTIONS:
        on_path = out[q] != ""
        confident_on_path[q] = ~on_path | (true_confidence(truth_df, q) >= CONFIDENT)
    out["fully_confident"] = confident_on_path.all(axis=1)
    out["path"] = [
        " > ".join(f"{q}={row[q]}" for q in TREE_ORDER if row[q])
        for _, row in out.iterrows()
    ]
    return out


def score_model(truth_paths, pred_answers):
    """pred_answers: {question: Series aligned to truth_paths.index}.
    Returns mean over galaxies of per-galaxy path accuracy."""
    correct = pd.DataFrame(index=truth_paths.index)
    for q in QUESTIONS:
        on_path = truth_paths[q] != ""
        correct[q] = np.where(on_path, pred_answers[q] == truth_paths[q], np.nan)
    per_galaxy = correct.mean(axis=1, skipna=True)  # fraction of path questions right
    return per_galaxy.mean(), per_galaxy


def main(gemini_csv, ids_path, model):
    sample_ids = [l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()]
    truth_df = load_bundle_seed(model=model).set_index("id_str").loc[sample_ids]

    truth_paths = build_truth_paths(truth_df)
    out_csv = HERE / "euclid_truth_paths.csv"
    truth_paths.reset_index().to_csv(out_csv, index=False)
    print(f"Saved {out_csv}  ({len(truth_paths)} galaxies, "
          f"path length {truth_paths['n_path_questions'].min()}–"
          f"{truth_paths['n_path_questions'].max()}, "
          f"median {int(truth_paths['n_path_questions'].median())})")

    fc = truth_paths[truth_paths["fully_confident"]]
    print(f"\nFully-confident galaxies (every path question majority >= {CONFIDENT}): "
          f"{len(fc)} of {len(truth_paths)} ({len(fc) / len(truth_paths):.1%})")
    print("Path-length distribution of that subset vs all:")
    dist = pd.DataFrame({
        "all": truth_paths["n_path_questions"].value_counts().sort_index(),
        "fully_confident": fc["n_path_questions"].value_counts().sort_index(),
    }).fillna(0).astype(int)
    print(dist.to_string())

    # --- Gemini ---
    g = (pd.read_csv(gemini_csv, dtype=str).drop_duplicates("id_str", keep="last")
         .set_index("id_str").reindex(truth_paths.index))
    g_preds = {q: g[q].fillna("__missing__") for q in QUESTIONS}
    g_score, g_per_galaxy = score_model(truth_paths, g_preds)

    # --- CNN: one score per seed, then mean ---
    seed_scores = []
    seed_perfect = []
    for sd in sorted(p for p in (BUNDLE_PREDICTIONS / model).iterdir() if p.is_dir()):
        df = pd.read_csv(sd / "predictions_test.csv").set_index("id_str").loc[truth_paths.index]
        c_preds = {q: cnn_answers(df, q) for q in QUESTIONS}
        s, per_galaxy = score_model(truth_paths, c_preds)
        seed_scores.append(s)
        seed_perfect.append((per_galaxy == 1).mean())

    print(f"\nONE NUMBER PER MODEL (mean per-galaxy path accuracy, n={len(truth_paths)}):")
    print(f"  CNN ({model}, {len(seed_scores)} seeds): {np.mean(seed_scores):.1%}"
          f"  (seed std {np.std(seed_scores):.2%})")
    print(f"  Gemini (structured, zero-shot):          {g_score:.1%}")

    print(f"\nPERFECT-PATH RATE (every path question correct, n={len(truth_paths)}):")
    print(f"  CNN ({model}, {len(seed_perfect)} seeds): {np.mean(seed_perfect):.1%}"
          f"  (seed std {np.std(seed_perfect):.2%})")
    print(f"  Gemini (structured, zero-shot):          {(g_per_galaxy == 1).mean():.1%}")

    print(f"\nGemini per-galaxy score distribution: "
          f"25%={g_per_galaxy.quantile(.25):.0%}  median={g_per_galaxy.median():.0%}  "
          f"75%={g_per_galaxy.quantile(.75):.0%}  perfect-path galaxies: "
          f"{(g_per_galaxy == 1).mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_csv", default=str(HERE / "gemini_euclid_predictions_structured.csv"))
    parser.add_argument("--ids",        default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--model",      default="gz_euclid_maxvit_base",
                        choices=["gz_euclid_convnext_base", "gz_euclid_maxvit_base"])
    args = parser.parse_args()
    main(args.gemini_csv, args.ids, args.model)
