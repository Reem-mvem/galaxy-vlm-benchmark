"""Test-set sampling strategies (all deterministic given a seed)."""

import numpy as np
import pandas as pd


def sample_all(catalog: pd.DataFrame, n_sample=None, seed=42) -> pd.DataFrame:
    """Evaluate the full catalog (n_sample ignored)."""
    print(f"Evaluating all {len(catalog)} test images")
    return catalog.copy()


def sample_balanced(catalog: pd.DataFrame, n_sample=None, seed=42) -> pd.DataFrame:
    """Draw n_sample images split equally across classes (no replacement)."""
    if not n_sample or n_sample >= len(catalog):
        return sample_all(catalog)
    rng = np.random.default_rng(seed)
    class_names = catalog["label_name"].unique()
    n_each = n_sample // len(class_names)
    groups = []
    for cls in class_names:
        pool = catalog[catalog["label_name"] == cls]
        idx = rng.choice(len(pool), size=min(n_each, len(pool)), replace=False)
        groups.append(pool.iloc[idx])
    sample = pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sample)} images (balanced)")
    return sample


def sample_minority_topup(catalog: pd.DataFrame, n_sample=None, seed=42) -> pd.DataFrame:
    """Keep ALL minority-class images, fill the rest with random majority.

    Used for strong lensing, where positives are far too rare to discard.
    """
    if not n_sample or n_sample >= len(catalog):
        return sample_all(catalog)
    rng = np.random.default_rng(seed)
    counts = catalog["label_name"].value_counts()
    minority = catalog[catalog["label_name"] == counts.idxmin()]
    majority = catalog[catalog["label_name"] == counts.idxmax()]
    n_majority = n_sample - len(minority)
    maj_idx = rng.choice(len(majority), size=n_majority, replace=False)
    sample = pd.concat([minority, majority.iloc[maj_idx]]).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sample)}: {len(minority)} {counts.idxmin()} + {n_majority} {counts.idxmax()}")
    return sample
