#!/usr/bin/env python
"""Evaluate Qwen (via OpenRouter) zero-shot on is-lsb (binary: tidal / not_tidal)."""

from task import TASK
from vlm_eval.runner import main

if __name__ == "__main__":
    main(TASK, provider="qwen", default_model="qwen/qwen3.5-397b-a17b")