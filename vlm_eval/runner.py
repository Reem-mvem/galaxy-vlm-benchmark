"""Generic evaluation loop and CLI shared by every task and provider.

A per-task entry point is three lines:

    from task import TASK
    from vlm_eval.runner import main
    main(TASK, provider="gemini", default_model="gemini-3.5-flash")
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from vlm_eval.metrics import compute_metrics_and_plot, save_results_csv
from vlm_eval.providers import TokenUsage, get_provider
from vlm_eval.task import TaskSpec

RETRY_DELAY_S = 10


def _query_one(task, provider, img_path):
    """Prepare one image, query the model once, parse. Returns (pred, text, usage)."""
    image_bytes = task.prepare_image(str(img_path))
    text, usage = provider.query(image_bytes, task.prompt)
    return task.parse_prediction(text), text, usage


def run_eval(task: TaskSpec, provider, output_csv, n_sample=None, seed=42,
             sleep_s=1.0):
    catalog = task.load_or_build_catalog()
    sample = task.get_sample(catalog, n_sample, seed)
    records = []
    n_total = len(sample)
    total = TokenUsage()

    try:
        for i, (_, row) in enumerate(sample.iterrows()):
            img_path = Path(row["file_loc"])
            if not img_path.exists():
                print(f"[{i + 1}/{n_total}] SKIPPED: {img_path}", file=sys.stderr)
                continue
            print(f"[{i + 1}/{n_total}] {row['id_str']} (true={row['label_name']})",
                  end=" ", flush=True)
            try:
                try:
                    pred, text, usage = _query_one(task, provider, img_path)
                except Exception as e:
                    print(f"\n  ERROR (retrying in {RETRY_DELAY_S}s): {e}", file=sys.stderr)
                    time.sleep(RETRY_DELAY_S)
                    pred, text, usage = _query_one(task, provider, img_path)
                total += usage
                records.append({"id_str": row["id_str"],
                                **task.record_extras(row),
                                "true_name": row["label_name"],
                                "pred_name": pred,
                                "raw_response": text.strip()})
                print(f"-> pred={str(pred):12s} [in={usage.input}]")
                time.sleep(sleep_s)
            except Exception as e:
                print(f"\n  RETRY FAILED: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n\nStopped early — saving {len(records)} records...")

    if not records:
        print("No results.")
        return
    save_results_csv(records, output_csv)
    print(f"Token usage — in: {total.input:,}  out: {total.output:,}  think: {total.thinking:,}")
    if hasattr(provider, "print_cost"):
        provider.print_cost(total)
    compute_metrics_and_plot(records, task.class_names, task.label,
                             provider.model, provider.label, output_csv,
                             prompt=task.prompt)


def fill_missing(task: TaskSpec, provider, output_csv, sleep_s=1.0):
    """Re-query only the rows of an existing results CSV with no parsed prediction."""
    if not Path(output_csv).exists():
        print(f"ERROR: {output_csv} not found. Run the eval first.", file=sys.stderr)
        return
    existing = pd.read_csv(output_csv)
    missing = existing[existing["pred_name"].isna()]
    print(f"Found {len(missing)} missing predictions out of {len(existing)} total")
    if missing.empty:
        return

    catalog = task.load_or_build_catalog()
    id_to_file = dict(zip(catalog["id_str"], catalog["file_loc"]))
    total = TokenUsage()
    filled = 0

    try:
        for i, (idx, row) in enumerate(missing.iterrows()):
            img_path = Path(id_to_file.get(row["id_str"], ""))
            print(f"[{i + 1}/{len(missing)}] {row['id_str']} (true={row['true_name']})",
                  end=" ", flush=True)
            if not img_path.exists():
                print("SKIPPED — file not found", file=sys.stderr)
                continue
            try:
                pred, text, usage = _query_one(task, provider, img_path)
                total += usage
                existing.at[idx, "pred_name"] = pred
                existing.at[idx, "raw_response"] = text.strip()
                filled += 1
                print(f"-> pred={str(pred):12s}")
                time.sleep(sleep_s)
            except Exception as e:
                print(f"\n  ERROR: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n\nStopped early — saving progress...")

    existing.to_csv(output_csv, index=False)
    print(f"\nFilled {filled} predictions -> {output_csv}")
    print(f"Token usage — in: {total.input:,}  out: {total.output:,}")
    print(f"{existing['pred_name'].isna().sum()} still missing after fill")
    valid_records = existing[existing["pred_name"].notna()].to_dict("records")
    compute_metrics_and_plot(valid_records, task.class_names, task.label,
                             provider.model, provider.label, output_csv,
                             prompt=task.prompt)


def main(task: TaskSpec, provider: str, default_model: str,
         default_thinking=None, default_n_sample=None, sleep_s=1.0):
    """Standard CLI for one (task, provider) pair."""
    parser = argparse.ArgumentParser(
        description=f"Evaluate {provider} zero-shot on {task.label}")
    parser.add_argument("--output", default=None,
                        help=f"results CSV (default: results/{task.name}_results_{provider}.csv)")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--n_sample", type=int, default=default_n_sample,
                        help="evaluate a subsample instead of the full test set")
    parser.add_argument("--seed", type=int, default=42)
    if provider == "gemini":
        parser.add_argument("--thinking", default=default_thinking,
                            choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--build_catalog_only", action="store_true",
                        help="download images / build the test catalog, then exit")
    parser.add_argument("--fill_missing", action="store_true",
                        help="re-query only unparsed rows of an existing results CSV")
    args = parser.parse_args()

    if args.build_catalog_only:
        if task.build_catalog is None:
            parser.error(f"{task.name} has no downloadable catalog (see its README)")
        task.build_catalog()
        return

    output_csv = args.output or task.dir / "results" / f"{task.name}_results_{provider}.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    thinking = getattr(args, "thinking", None)
    prov = get_provider(provider, args.model, thinking=thinking)

    if args.fill_missing:
        fill_missing(task, prov, output_csv, sleep_s=sleep_s)
    else:
        run_eval(task, prov, output_csv, n_sample=args.n_sample, seed=args.seed,
                 sleep_s=sleep_s)
