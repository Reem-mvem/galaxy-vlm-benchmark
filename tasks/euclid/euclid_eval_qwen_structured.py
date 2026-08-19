#!/usr/bin/env python
"""
Qwen (via OpenRouter) zero-shot on the GZ Euclid 13-question decision tree.

OpenRouter is asked for a strict json_schema response_format built from the
EuclidDecisionTree model. Not every OpenRouter backend honours the schema, so
the reply is always re-validated with Pydantic; if that fails the row is
recorded blank (counts as wrong) rather than crashing the run.

Run:  python euclid_eval_qwen_structured.py                (full 1000 sample)
      python euclid_eval_qwen_structured.py --limit 5      (smoke test)
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from euclid_eval_common import EuclidDecisionTree, Usage, run_eval, HERE

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

MODEL_NAME = "qwen/qwen3.5-397b-a17b"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 1200

_SCHEMA = EuclidDecisionTree.model_json_schema()


def _coerce(content):
    """Best-effort parse of the model text into an EuclidDecisionTree, or None."""
    if not content:
        return None
    try:
        return EuclidDecisionTree.model_validate_json(content)
    except Exception:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)  # dig a JSON object out of prose
    if m:
        try:
            return EuclidDecisionTree.model_validate(json.loads(m.group(0)))
        except Exception:
            return None
    return None


def make_query_one(api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def query_one(image_bytes, prompt):
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "reasoning": {"enabled": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "euclid_tree", "strict": True, "schema": _SCHEMA},
            },
        }
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage", {})
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 0)
        tree = _coerce(content)
        raw = tree.model_dump_json() if tree is not None else content
        return tree, usage, raw

    return query_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "qwen_euclid_predictions_structured.csv"))
    parser.add_argument("--ids",    default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only run the first N sample ids (smoke test)")
    parser.add_argument("--sleep",  type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    run_eval(make_query_one(api_key), "Qwen",
             args.output, args.ids, args.limit, args.sleep)
