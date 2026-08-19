"""One-image Gemini 'show me the reasoning' — MATCHED to the Qwen which_lsb trace.
Same galaxy (test_32, true=arm), same prompt. thinking_level=HIGH, include_thoughts=True.
Prints Gemini's thought summary + final answer so it can sit next to Qwen's raw CoT.

Auth: reads GEMINI_API_KEY / GOOGLE_API_KEY from env, or a key file path in GEMINI_KEY_FILE
(or ~/.gemini_key). Internet works from this node; only the key is needed.
Run:  THINK_LEVEL=HIGH python gemini_show_thinking_test32.py
"""
import os, sys
from pathlib import Path
from google import genai
from google.genai import types

IMG    = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/data/images/test/32.jpg"
PROMPT = ("This galaxy image shows faint debris. Classify the debris morphology. "
          "Reply with one word: arm, stream, shell, or diffuse.")
MODEL  = "gemini-3.5-flash"
LEVEL  = os.environ.get("THINK_LEVEL", "HIGH").upper()
OUT    = "/home/reem1/scratch/galaxy-vlm-benchmark/analysis/poster_reasoning_GEMINI_test32.txt"

def get_key():
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        if os.environ.get(v):
            return os.environ[v].strip()
    kf = os.environ.get("GEMINI_KEY_FILE") or os.path.expanduser("~/.gemini_key")
    if Path(kf).is_file():
        return Path(kf).read_text().strip()
    envf = Path("/home/reem1/scratch/.env")           # dotenv copied onto Narval
    if envf.is_file():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No API key found (env / ~/.gemini_key / /home/reem1/scratch/.env).")

client = genai.Client(api_key=get_key())
img_bytes = open(IMG, "rb").read()
img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

# --- EXACTLY ONE API call: SAME prompt as Qwen, thinking HIGH, include the thought summary ---
resp = client.models.generate_content(
    model=MODEL,
    contents=[img_part, PROMPT],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=LEVEL, include_thoughts=True),
    ),
)
thoughts, answer = [], []
for part in resp.candidates[0].content.parts:
    if not getattr(part, "text", None):
        continue
    (thoughts if getattr(part, "thought", False) else answer).append(part.text)
u = resp.usage_metadata

print(f"=== Gemini {MODEL}  thinking={LEVEL}  (ONE call · image test_32 · TRUE answer = arm) ===\n")
print("########## THINKING SUMMARY ##########")
print("\n".join(thoughts).strip() or "(no thought summary returned)")
print("\n########## ANSWER ##########")
print("\n".join(answer).strip() or "(empty)")
print(f"\n[hidden raw-thinking tokens: {getattr(u,'thoughts_token_count',None)}]")

Path(OUT).write_text(
    "=== THINKING SUMMARY ===\n" + "\n".join(thoughts).strip() +
    "\n\n=== ANSWER ===\n" + "\n".join(answer).strip() +
    "\n[hidden raw-thinking tokens: %s]\n" % getattr(u, "thoughts_token_count", None))
print("\nsaved ->", OUT)
