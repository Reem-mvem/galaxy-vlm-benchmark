#!/usr/bin/env python
"""
Download the GZ Euclid test split, verify it matches the CNN predictions
bundle, and pick the evaluation sample.

Sampling: all galaxies where a rare question (edge-on-bulge, problem,
artifact) is relevant are force-included, then the rest is filled randomly
(seed 42) up to --n_sample. The CNN is re-scored on exactly these ids, so
both contestants see the same galaxies.

Outputs (in this folder):
  euclid_sample_ids.txt    one id_str per line — THE sample definition
  euclid_sample_truth.csv  id_str + true vote fractions + total votes

Run:  python sample_test_set.py
"""

import argparse
import os
import sys

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from euclid_schema import (
    QUESTIONS, RARE_QUESTIONS, ROOT,
    fraction_cols, votes_col, relevance_mask, load_bundle_seed,
)

load_dotenv()
load_dotenv(ROOT / "zoobot_project" / ".env")

HF_DATASET = "mwalmsley/gz_euclid"


def main(n_sample, seed):
    print(f"Loading bundle ground truth (convnext_base seed_0)...")
    df = load_bundle_seed()
    print(f"  {len(df)} test galaxies in bundle")

    print(f"Loading {HF_DATASET} test split from HuggingFace (one-time download)...")
    ds = load_dataset(HF_DATASET, split="test", token=os.getenv("HF_TOKEN"))
    hf_ids = set(ds["id_str"])
    bundle_ids = set(df["id_str"])
    if hf_ids != bundle_ids:
        print(f"ERROR: id mismatch — HF has {len(hf_ids)}, bundle has {len(bundle_ids)}, "
              f"overlap {len(hf_ids & bundle_ids)}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK: HF test split ids match bundle exactly ({len(hf_ids)} galaxies)")

    # force-include all galaxies where any rare question is relevant
    rare_mask = np.zeros(len(df), dtype=bool)
    for q in RARE_QUESTIONS:
        rare_mask |= relevance_mask(df, q).to_numpy()
    forced = df[rare_mask]
    rest = df[~rare_mask]
    n_fill = n_sample - len(forced)
    if n_fill < 0:
        print(f"ERROR: {len(forced)} forced galaxies exceed n_sample={n_sample}", file=sys.stderr)
        sys.exit(1)
    filled = rest.sample(n=min(n_fill, len(rest)), random_state=seed)
    sample = (
        df.loc[sorted(forced.index.tolist() + filled.index.tolist())]
        .reset_index(drop=True)
    )
    print(f"\nSample: {len(forced)} forced (rare branches) + {len(filled)} random = {len(sample)}")

    print("\nRelevant galaxies per question (sample vs full test set):")
    for q in QUESTIONS:
        n_s = int(relevance_mask(sample, q).sum())
        n_f = int(relevance_mask(df, q).sum())
        print(f"  {q:20s} {n_s:5d} / {n_f}")

    ids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "euclid_sample_ids.txt")
    sample["id_str"].to_csv(ids_path, index=False, header=False)
    print(f"\nSaved sample ids to {ids_path}")

    truth_cols = ["id_str"]
    for q in QUESTIONS:
        truth_cols += fraction_cols(q) + [votes_col(q)]
    truth_path = ids_path.replace("euclid_sample_ids.txt", "euclid_sample_truth.csv")
    sample[truth_cols].to_csv(truth_path, index=False)
    print(f"Saved ground truth to {truth_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_sample, args.seed)
