"""Shared toolkit for evaluating vision-language models on galaxy morphology tasks.

Each task (tasks/<name>/task.py) declares a TaskSpec; the per-provider entry
points (eval_gemini.py, eval_openai.py, eval_qwen.py) call runner.main() with
that spec. Everything else — catalog building, sampling, the query loop,
metrics, and confusion-matrix plots — lives here and is shared.
"""

from vlm_eval.task import TaskSpec

__all__ = ["TaskSpec"]
