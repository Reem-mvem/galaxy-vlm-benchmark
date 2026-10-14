#!/usr/bin/env python
"""
Gemini zero-shot on the GZ Euclid 13-question decision tree (structured output).

A Pydantic model with Literal fields is passed as the response_schema, so the
API constrains decoding to the schema — every answer is guaranteed valid.

Run:  python euclid_eval_gemini_structured.py                (full 1000 sample)
      python euclid_eval_gemini_structured.py --limit 5      (smoke test)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from euclid_eval_common import EuclidDecisionTree, Usage, run_eval, HERE

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

MODEL_NAME = "gemini-3.5-flash"


def make_query_one(client, thinking_level):
    def query_one(image_bytes, prompt):
        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": EuclidDecisionTree,
        }
        if thinking_level:
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_level=thinking_level.upper())
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
        tree = response.parsed
        raw = None
        if tree is None:  # rare: fall back to validating the raw text
            tree = EuclidDecisionTree.model_validate_json(response.text)
        raw = tree.model_dump_json()
        u = response.usage_metadata
        usage = Usage(u.prompt_token_count or 0,
                      u.candidates_token_count or 0,
                      u.thoughts_token_count or 0)
        return tree, usage, raw

    return query_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default=str(HERE / "gemini_euclid_predictions_structured.csv"))
    parser.add_argument("--ids",      default=str(HERE / "euclid_sample_ids.txt"))
    parser.add_argument("--thinking", default="minimal",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--limit",    type=int, default=None,
                        help="Only run the first N sample ids (smoke test)")
    parser.add_argument("--sleep",    type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    run_eval(make_query_one(client, args.thinking), "Gemini",
             args.output, args.ids, args.limit, args.sleep)
