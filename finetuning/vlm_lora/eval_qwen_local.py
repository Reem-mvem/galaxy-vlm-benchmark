#!/usr/bin/env python
"""Evaluate a locally-hosted Qwen2-VL on any task via the shared vlm_eval pipeline.

  python eval_qwen_local.py --task is_lsb --output <csv>

Loads tasks/<task>/task.py's TASK spec and runs the ``qwen_local`` provider (same
prompt / images / parsing / metrics as the Gemini runs). Zero-shot by default;
set QWEN_ADAPTER_PATH=<adapter dir> to serve a fine-tuned adapter instead.
All other flags (--output, --model, --n_sample, --seed, --fill_missing) pass
straight through to vlm_eval.runner.main.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_task(task_name):
    task_dir = REPO / "tasks" / task_name
    if not (task_dir / "task.py").exists():
        sys.exit(f"no task.py under {task_dir}")
    sys.path.insert(0, str(task_dir))
    import importlib
    return importlib.import_module("task").TASK


if __name__ == "__main__":
    # Peel off --task, leave the rest of argv for vlm_eval.runner.main's parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True)
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    TASK = _load_task(known.task)
    from vlm_eval.runner import main
    main(TASK, provider="qwen_local", default_model="qwen2vl-7b", sleep_s=0.0)
