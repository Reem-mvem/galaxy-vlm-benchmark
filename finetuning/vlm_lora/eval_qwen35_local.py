#!/usr/bin/env python
"""Evaluate the local Qwen3.5-VL on any task via the shared vlm_eval pipeline.
  python eval_qwen35_local.py --task which_lsb --output <csv>
Zero-shot by default; set QWEN_ADAPTER_PATH=<adapter dir> for the fine-tuned variant.
Uses the same tasks/<task>/task.py specs + catalogs as the Qwen2 runs (shared).
"""
import argparse, sys, importlib
from pathlib import Path

REPO = Path("/home/reem1/scratch/galaxy-vlm-benchmark")   # shared task specs + vlm_eval


def _load_task(task_name):
    task_dir = REPO / "tasks" / task_name
    if not (task_dir / "task.py").exists():
        sys.exit(f"no task.py under {task_dir}")
    sys.path.insert(0, str(task_dir))
    sys.path.insert(0, str(REPO))          # so `import vlm_eval` resolves
    return importlib.import_module("task").TASK


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True)
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    TASK = _load_task(known.task)
    from vlm_eval.runner import main
    main(TASK, provider="qwen35_local", default_model="qwen3.5-4b", sleep_s=0.0)
