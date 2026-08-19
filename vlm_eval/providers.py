"""Thin wrappers around the model APIs, with one shared interface.

Every provider exposes ``query(image_bytes, prompt) -> (text, TokenUsage)``.
API keys are read from the repo-root .env (see .env.example):
  GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from vlm_eval.images import to_b64

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()  # also pick up a task-local .env / real environment


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    thinking: int = 0

    def __iadd__(self, other):
        self.input += other.input
        self.output += other.output
        self.thinking += other.thinking
        return self


def _require_key(env_var):
    key = os.getenv(env_var)
    if not key:
        print(f"ERROR: {env_var} not set — copy .env.example to .env and fill it in.",
              file=sys.stderr)
        sys.exit(1)
    return key


class GeminiProvider:
    """Google Gemini via the google-genai SDK."""

    label = "Gemini"

    def __init__(self, model, thinking=None):
        from google import genai
        self.genai = genai
        self.model = model
        self.thinking = thinking  # None => omit thinking_config entirely
        self.client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))

    def query(self, image_bytes, prompt):
        genai = self.genai
        config_kwargs = {}
        if self.thinking:
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_level=self.thinking.upper())
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text if hasattr(response, "text") else str(response)
        u = response.usage_metadata
        usage = TokenUsage(u.prompt_token_count or 0,
                           u.candidates_token_count or 0,
                           u.thoughts_token_count or 0)
        return text or "", usage

    def print_cost(self, usage: TokenUsage):
        """Estimated cost via genai_prices, if installed."""
        try:
            from genai_prices import Usage, calc_price
        except ImportError:
            return
        try:
            price = calc_price(
                Usage(input_tokens=usage.input, output_tokens=usage.output + usage.thinking),
                self.model, provider_id="google",
            )
            print(f"Estimated cost: ${price.total_price:.4f} USD")
        except LookupError:
            print(f"  (model '{self.model}' not in genai_prices — cost unknown)")


class OpenAIProvider:
    """OpenAI chat completions with an inline base64 image."""

    label = "OpenAI"

    def __init__(self, model, image_detail="low", max_completion_tokens=500):
        from openai import OpenAI
        self.model = model
        self.image_detail = image_detail
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))

    def query(self, image_bytes, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{to_b64(image_bytes)}",
                               "detail": self.image_detail}},
                {"type": "text", "text": prompt},
            ]}],
            max_completion_tokens=self.max_completion_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = TokenUsage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return text, usage


class OpenRouterProvider:
    """Any OpenRouter-hosted model (used for Qwen), reasoning disabled."""

    label = "Qwen"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model, max_tokens=20):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = _require_key("OPENROUTER_API_KEY")

    def query(self, image_bytes, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{to_b64(image_bytes)}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "reasoning": {"enabled": False},
        }
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage", {})
        usage = TokenUsage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return text, usage


class QwenLocalProvider:
    """Locally-hosted Qwen2-VL, run offline on a compute node (unlike the
    OpenRouter 'qwen' provider, which needs the internet). Serves both the
    zero-shot base model and a LoRA-fine-tuned variant.

    Env vars:
      MODEL_LOCAL_PATH   dir of the base model (staged to $SLURM_TMPDIR)
      QWEN_ADAPTER_PATH  optional LoRA adapter dir; if set it is attached, so
                         the SAME provider serves zero-shot and fine-tuned.
    Heavy imports (torch/transformers) are lazy so an API-only run (Gemini,
    OpenAI, OpenRouter) never needs them installed.
    """

    label = "Qwen-local"

    def __init__(self, model, max_new_tokens=10):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        self.torch = torch
        model_path = os.environ.get("MODEL_LOCAL_PATH") or model
        self.m = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        adapter = os.environ.get("QWEN_ADAPTER_PATH")
        if adapter:
            self.m.load_adapter(adapter)
            self.model = "Qwen2-VL-7B+adapter"
            print(f"[qwen_local] loaded adapter: {adapter}")
        else:
            self.model = "Qwen2-VL-7B-zeroshot"
        self.m.eval()
        self.max_new_tokens = max_new_tokens

    def query(self, image_bytes, prompt):
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img],
                                return_tensors="pt").to(self.m.device)
        n_in = inputs.input_ids.shape[1]
        with self.torch.no_grad():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        gen = out[:, n_in:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return ans, TokenUsage(input=int(n_in), output=int(gen.shape[1]))


class Qwen35LocalProvider:
    """Locally-hosted Qwen3.5-VL (dense), offline. Same interface as QwenLocalProvider
    but uses the Auto classes (works for the Qwen3.5 arch). Serves zero-shot base and
    a LoRA-fine-tuned variant via QWEN_ADAPTER_PATH. Heavy imports are lazy.
    """

    label = "Qwen3.5-local"

    def __init__(self, model, max_new_tokens=10):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # QWEN_MAX_NEW_TOKENS lets us capture the full reasoning trace (e.g. 512) instead
        # of just the first few tokens; default stays short for normal parsing.
        max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", max_new_tokens))
        self.torch = torch
        model_path = os.environ.get("MODEL_LOCAL_PATH") or model
        self.m = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        adapter = os.environ.get("QWEN_ADAPTER_PATH")
        if adapter:
            self.m.load_adapter(adapter)
            self.model = "Qwen3.5+adapter"
            print(f"[qwen35_local] loaded adapter: {adapter}")
        else:
            self.model = "Qwen3.5-zeroshot"
        self.m.eval()
        self.max_new_tokens = max_new_tokens

    def query(self, image_bytes, prompt):
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)  # direct answer, no reasoning (match 7B)
        inputs = self.processor(text=[text], images=[img],
                                return_tensors="pt").to(self.m.device)
        n_in = inputs.input_ids.shape[1]
        with self.torch.no_grad():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        gen = out[:, n_in:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return ans, TokenUsage(input=int(n_in), output=int(gen.shape[1]))


PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "qwen": OpenRouterProvider,
    "qwen_local": QwenLocalProvider,
    "qwen35_local": Qwen35LocalProvider,
}


def get_provider(name, model, thinking=None):
    if name == "gemini":
        return GeminiProvider(model, thinking=thinking)
    return PROVIDERS[name](model)
