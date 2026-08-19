#!/usr/bin/env python
"""Evaluate OpenAI zero-shot on GZ Rings (binary: ring / no_ring)."""

from task import TASK
from vlm_eval.runner import main

if __name__ == "__main__":
    main(TASK, provider="openai", default_model="gpt-5.6-sol", default_n_sample=1000)