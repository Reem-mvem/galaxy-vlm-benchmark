#!/usr/bin/env python
"""
OpenAI zero-shot on the GZ Euclid 13-question decision tree (structured output).

Uses the native structured-output parse endpoint with the EuclidDecisionTree
Pydantic model as response_format, so every answer is a valid schema value.

Run:  python euclid_eval_openai_structured.py                (full 1000 sample)
      python euclid_eval_openai_structured.py --limit 5      (smoke test)
"""

import argparse
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from euclid_eval_common import EuclidDecisionTree, Usage, run_eval, HERE

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

MODEL_NAME = "gpt-5.6-sol"
IMAGE_DETAIL = "high"          # fine morphology needs the detail
MAX_COMPLETION_TOKENS = 3000   # headroom for reasoning + the 13-field object
REASONING_EFFORT = "low"


def make_query_one(client):
    def query_one(image_bytes, prompt):
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": IMAGE_DETAIL}},
                {"type": "text", "text": prompt},
            ]}],
            response_format=EuclidDecisionTree,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=REASONING_EFFORT,
        )
        msg = resp.choices[0].message
        u = resp.usage
        reasoning = 0
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
        usage = Usage(u.prompt_tokens, (u.completion_tokens or 0) - reasoning, reasoning)

        tree = msg.parsed
        if tree is None:
            # refusal or truncation — record as unparseable so it counts wrong
            raw = msg.refusal or (msg.content or "")
            return None, usage, raw
        return tree, usage, tree.model_dump_json()

    return query_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "openai_euclid_predictions_structured.csv"))
    parser.add_argument("--ids",    default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only run the first N sample ids (smoke test)")
    parser.add_argument("--sleep",  type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    run_eval(make_query_one(client), "OpenAI",
             args.output, args.ids, args.limit, args.sleep)
