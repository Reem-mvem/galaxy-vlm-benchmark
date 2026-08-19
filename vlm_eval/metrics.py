"""Results saving, metrics, and confusion-matrix plotting."""

import csv
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


def save_results_csv(records, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} rows -> {output_csv}")


def compute_metrics_and_plot(records, class_names, task_label, model_name,
                             model_label, output_csv, prompt):
    """Balanced accuracy + classification report + normalized confusion PNG.

    Records need ``true_name`` and ``pred_name`` keys; rows whose prediction
    could not be parsed (pred_name None/NaN) are dropped and counted.
    Returns the balanced accuracy, or None if no valid predictions.
    """
    valid = [r for r in records
             if r.get("pred_name") is not None and r["pred_name"] == r["pred_name"]]
    n_invalid = len(records) - len(valid)
    print(f"\n{n_invalid} unparseable responses out of {len(records)}")

    if not valid:
        return None

    label_map = {name: i for i, name in enumerate(class_names)}
    true_idx = [label_map[r["true_name"]] for r in valid]
    pred_idx = [label_map.get(r["pred_name"], -1) for r in valid]

    bal_acc = balanced_accuracy_score(true_idx, pred_idx)

    print(f"\n{'=' * 60}")
    print("EVALUATION METRICS")
    print(f"{'=' * 60}")
    print(f"  Model:              {model_name}")
    print(f"  Valid predictions:  {len(valid)} / {len(records)}")
    print(f"  Balanced Accuracy:  {bal_acc:.4f}  ({bal_acc:.2%})")
    print()
    print(classification_report(true_idx, pred_idx, target_names=class_names, zero_division=0))

    k = len(class_names)
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(k)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5 + 0.4 * k, 5 + 0.3 * k))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_label} — {task_label}\nBal. Acc: {bal_acc:.1%}  (model={model_name})",
                 fontsize=10)
    for r in range(k):
        for c in range(k):
            ax.text(c, r, f"{cm_norm[r, c]:.2f}\n({cm[r, c]})",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_norm[r, c] > 0.55 else "black")
    plt.colorbar(im, ax=ax)
    prompt_wrapped = "\n".join(textwrap.wrap(prompt.replace("\n", " "), width=90))
    fig.text(0.5, 0.01, f"Prompt: {prompt_wrapped}",
             ha="center", va="bottom", fontsize=6, style="italic",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    cm_path = str(output_csv).replace(".csv", "_confusion.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix -> {cm_path}")
    return bal_acc
