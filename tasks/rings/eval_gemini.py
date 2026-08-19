#!/usr/bin/env python
"""Evaluate Gemini zero-shot on GZ Rings (binary: ring / no_ring)."""

from task import TASK
from vlm_eval.runner import main

if __name__ == "__main__":
    main(TASK, provider="gemini", default_model="gemini-3.5-flash", default_thinking="minimal", default_n_sample=1000)