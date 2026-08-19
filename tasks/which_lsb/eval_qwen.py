#!/usr/bin/env python
"""Evaluate Qwen (via OpenRouter) zero-shot on which-lsb (4-class tidal debris morphology)."""

from task import TASK
from vlm_eval.runner import main

if __name__ == "__main__":
    main(TASK, provider="qwen", default_model="qwen/qwen3.5-397b-a17b")