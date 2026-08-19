"""Accuracy vs volunteer agreement: do models fail where humans disagree?

Every relevant (galaxy, question) pair (same >=50% routing rule as
score_euclid.py) is binned by the volunteers' agreement on the majority
answer; per bin, accuracy = fraction of answers matching that majority.
CNN is mean over the 6 bundle seeds. Recomputes from the data, then plots
a dot-line chart in the same palette/specs as plot_results.py.

Run:  python plot_agreement.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from euclid_schema import (
    QUESTIONS, BUNDLE_PREDICTIONS,
    relevance_mask, true_answers, true_confidence, cnn_answers, load_bundle_seed,
)

MODEL = "gz_euclid_maxvit_base"
GEMINI_CSV = "gemini_euclid_predictions_structured.csv"
IDS = "euclid_sample_ids.txt"

BIN_EDGES = [0, .5, .6, .7, .8, .9, 1.0001]
BIN_LABELS = ["<50%", "50–60%", "60–70%", "70–80%",
              "80–90%", "90–100%"]

# palette (validated reference instance, light mode; same entities as plot_results.py)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
CNN_C     = "#2a78d6"   # categorical slot 1 (blue)
GEMINI_C  = "#1baf7a"   # categorical slot 2 (aqua)


def long_table(truth_df):
    """One row per relevant (galaxy, question) pair: id, question, agreement, majority."""
    rows = []
    for q in QUESTIONS:
        m = relevance_mask(truth_df, q)
        sub = truth_df.loc[m]
        rows.append(pd.DataFrame({
            "id": sub.index, "q": q,
            "conf": true_confidence(sub, q).values,
            "truth": true_answers(sub, q).values,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["bin"] = pd.cut(out["conf"], BIN_EDGES, labels=BIN_LABELS, right=False)
    return out


def bin_accuracy(long, correct):
    return (pd.Series(correct, index=long.index)
            .groupby(long["bin"], observed=True).mean().reindex(BIN_LABELS))


def main():
    ids = [l.strip() for l in open(IDS, encoding="utf-8") if l.strip()]
    truth_df = load_bundle_seed(model=MODEL).set_index("id_str").loc[ids]
    long = long_table(truth_df)
    n_per_bin = long["bin"].value_counts().reindex(BIN_LABELS)

    g = (pd.read_csv(GEMINI_CSV, dtype=str).drop_duplicates("id_str", keep="last")
         .set_index("id_str"))
    g_pred = np.array([g.at[i, q] for i, q in zip(long["id"], long["q"])])
    gem_acc = bin_accuracy(long, g_pred == long["truth"].values)

    seed_accs = []
    for sd in sorted(p for p in (BUNDLE_PREDICTIONS / MODEL).iterdir() if p.is_dir()):
        df = pd.read_csv(sd / "predictions_test.csv").set_index("id_str").loc[ids]
        preds = {q: cnn_answers(df, q) for q in QUESTIONS}
        p = np.array([preds[q][i] for i, q in zip(long["id"], long["q"])])
        seed_accs.append(bin_accuracy(long, p == long["truth"].values))
    cnn_acc = pd.concat(seed_accs, axis=1).mean(axis=1)
    cnn_std = pd.concat(seed_accs, axis=1).std(axis=1)

    print(pd.DataFrame({"n": n_per_bin, "cnn": (cnn_acc * 100).round(1),
                        "cnn_seed_std": (cnn_std * 100).round(2),
                        "gemini": (gem_acc * 100).round(1)}).to_string())

    # --- plot: dot-line, one series per model ---
    x = np.arange(len(BIN_LABELS))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for acc, color in [(cnn_acc, CNN_C), (gem_acc, GEMINI_C)]:
        ax.plot(x, acc * 100, color=color, linewidth=2, zorder=3,
                marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        # selective direct labels: first and last point only
        for i in (0, len(x) - 1):
            ax.text(x[i], acc.iloc[i] * 100 + 3, f"{acc.iloc[i]:.1%}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="semibold", color=INK, zorder=4)

    ax.set_xlim(-0.4, len(BIN_LABELS) - 0.6)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(BIN_LABELS, n_per_bin)],
                       fontsize=9.5, color=INK_2)
    ax.set_xlabel("Volunteer agreement on the majority answer", fontsize=10,
                  color=INK_2, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)

    handles = [plt.Line2D([], [], color=CNN_C, linewidth=2, marker="o",
                          markersize=7, markeredgecolor=SURFACE),
               plt.Line2D([], [], color=GEMINI_C, linewidth=2, marker="o",
                          markersize=7, markeredgecolor=SURFACE)]
    ax.legend(handles, ["Fine-tuned CNN (maxvit, mean of 6 seeds)",
                        "Gemini zero-shot (structured output)"],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2,
              handlelength=1.6)

    fig.text(0.06, 0.965, "Models fail where humans disagree",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.06, 0.915, "Answer accuracy vs volunteer agreement — "
             "4,400 question-answers on 1,000 GZ Euclid galaxies",
             fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.08, right=0.97)

    out = "gz_euclid_accuracy_vs_agreement.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
