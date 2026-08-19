"""
Shared GZ Euclid decision-tree schema + ground-truth helpers.

Column naming in the bundle predictions CSVs
(predictions/gz_euclid_*/seed_*/predictions_test.csv):

  <question>-euclid_<answer>            = model predicted Dirichlet concentration
  <question>-euclid_<answer>_fraction   = TRUE volunteer vote fraction
  <question>-euclid_total-votes         = TRUE number of volunteers asked this question

Relevance rule (reverse-engineered to exactly reproduce
zoobot_project/gz_euclid_cnn_pernode_accuracy.csv galaxy counts):
  a question is relevant for a galaxy when at least 50% of that galaxy's
  volunteers were routed to it, i.e.
      total-votes(question) / total-votes(smooth-or-featured) >= 0.5
  smooth-or-featured itself is relevant whenever total-votes >= 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# question -> allowed answers, in bundle column order
QUESTIONS = {
    "smooth-or-featured": ["smooth", "featured-or-disk", "problem"],
    "disk-edge-on":       ["yes", "no"],
    "has-spiral-arms":    ["yes", "no"],
    "bar":                ["strong", "weak", "no"],
    "bulge-size":         ["dominant", "large", "moderate", "small", "none"],
    "how-rounded":        ["round", "in-between", "cigar-shaped"],
    "edge-on-bulge":      ["boxy", "none", "rounded"],
    "spiral-winding":     ["tight", "medium", "loose"],
    "spiral-arm-count":   ["1", "2", "3", "4", "more-than-4", "cant-tell"],
    "merging":            ["none", "minor-disturbance", "major-disturbance", "merger"],
    "clumps":             ["yes", "no"],
    "problem":            ["star", "artifact", "zoom"],
    "artifact":           ["satellite", "scattered", "diffraction", "ray", "saturation", "other", "ghost"],
}

# questions with so few relevant galaxies that a random subsample would
# leave almost nothing — all their relevant galaxies are force-included
RARE_QUESTIONS = ["edge-on-bulge", "problem", "artifact"]

BASE_QUESTION = "smooth-or-featured"

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PREDICTIONS = (
    ROOT
    / "gz_evo_downstream_predictions_bundle_2026-07-02"
    / "gz_evo_downstream_predictions_bundle_2026-07-02"
    / "predictions"
)


def concentration_cols(question):
    return [f"{question}-euclid_{a}" for a in QUESTIONS[question]]


def fraction_cols(question):
    return [f"{question}-euclid_{a}_fraction" for a in QUESTIONS[question]]


def votes_col(question):
    return f"{question}-euclid_total-votes"


def relevance_mask(df, question):
    """Boolean Series: is `question` relevant for each galaxy in df."""
    votes = df[votes_col(question)]
    if question == BASE_QUESTION:
        return votes >= 1
    base = df[votes_col(BASE_QUESTION)]
    return votes / base >= 0.5


def true_answers(df, question):
    """Volunteers' majority answer per galaxy (argmax of true fractions)."""
    fracs = df[fraction_cols(question)].to_numpy()
    idx = np.argmax(fracs, axis=1)
    answers = np.array(QUESTIONS[question])
    return pd.Series(answers[idx], index=df.index)


def true_confidence(df, question):
    """Vote fraction of the majority answer per galaxy."""
    return df[fraction_cols(question)].max(axis=1)


def cnn_answers(df, question):
    """CNN's answer per galaxy (argmax of predicted concentrations)."""
    conc = df[concentration_cols(question)].to_numpy()
    idx = np.argmax(conc, axis=1)
    answers = np.array(QUESTIONS[question])
    return pd.Series(answers[idx], index=df.index)


def load_bundle_seed(model="gz_euclid_convnext_base", seed=0):
    path = BUNDLE_PREDICTIONS / model / f"seed_{seed}" / "predictions_test.csv"
    return pd.read_csv(path)
