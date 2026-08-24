# How to Run

This is an **evaluation project**: it benchmarks how well vision-language models (VLMs)
classify galaxy morphology compared to purpose-built CNNs, across several tasks and label
budgets. There is no single application to launch — you run **evaluation scripts** (and,
optionally, fine-tuning scripts) per task and per model, and each writes a results CSV.

This guide is self-contained: follow it top to bottom to install, provide API keys, get the
data, and run any part of the benchmark.

---

## 1. Requirements

| Part | Needs |
|---|---|
| Zero-shot **paid API** models (Gemini, GPT, Qwen-397B) | **API keys**, internet — **no GPU** |
| Local **open** VLM (Qwen-VL) zero-shot or fine-tuned | **1 GPU** (NVIDIA, ≥20 GB, e.g. A100) |
| **Fine-tuning** an open VLM (LoRA) | **1 GPU** (≥20 GB) |
| **CNN** baseline (gz-evo) | **1 GPU** |
| Plotting / analysis | CPU only |

- **Python ≥ 3.10**
- A GPU is required **only** for the local open-model steps (running/fine-tuning Qwen-VL and
  the CNN). The zero-shot API evaluations and all plotting run on CPU.

## 2. Install

From the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `vlm_eval` package and its dependencies (google-genai, openai, pandas,
numpy, scikit-learn, datasets, pillow, matplotlib, python-dotenv).

For the local open-model steps you also need PyTorch + Transformers with GPU support, plus
PEFT/TRL for fine-tuning:

```bash
pip install torch transformers accelerate peft trl
```

(Install the CUDA build of `torch` that matches your GPU/driver — see pytorch.org.)

## 3. API keys (required for the paid models)

The zero-shot commercial models are called over their APIs, so **you must supply your own
keys**. Copy the example file and fill it in:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_google_ai_studio_key
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key      # Qwen3.5-397B is served via OpenRouter
```

- **Gemini** → Google AI Studio key. **GPT** → OpenAI key. **Qwen-397B** → OpenRouter key.
- You only need the key for the model(s) you intend to run. **These calls cost money** (on
  the order of ~$0.15–0.17 per 100 images); the local open models are free to run once you
  have a GPU.
- `.env` is git-ignored and must **not** be shared.

## 4. Get the data

Image datasets are **not shipped** with the source — they are downloaded from HuggingFace on
first use (`mwalmsley/gz_rings`, `mwalmsley/gz_euclid`, `astronolan/streams_shells_neither`,
plus the Galaxy Zoo / Euclid catalogs). This needs internet access. A few datasets may require
a HuggingFace token — if so, add `HF_TOKEN=...` to your `.env` (also shown in `.env.example`).

Each task can build its test catalog + download its images once, e.g.:

```bash
cd tasks/rings
python eval_gemini.py --build_catalog_only     # downloads test images, writes the catalog
```

(Some tasks have a `build_*.py` script instead — see the task folder.)

## 5. Run a zero-shot evaluation (paid APIs, no GPU)

Each task in `tasks/<task>/` has one script per provider. Run it from inside the task folder:

```bash
cd tasks/rings
python eval_gemini.py          # Gemini 3.5 Flash
python eval_openai.py          # GPT
python eval_qwen.py            # Qwen3.5-397B (via OpenRouter)
```

Common flags (see `vlm_eval/runner.py`):

| Flag | Meaning | Default |
|---|---|---|
| `--model` | model id to call | task's default (e.g. `gemini-3.5-flash`) |
| `--n_sample` | number of test images to evaluate | task default (e.g. 1000) |
| `--seed` | sampling seed | 42 |
| `--thinking` | reasoning level (Gemini): `minimal`/`low`/`medium`/`high` | task default |
| `--output` | path for the results CSV | auto |

Each run writes a `..._results_*.csv` with `true_name,pred_name` (and probabilities) under
the task's `results/` folder. Score it with the plotting/metric scripts in `analysis/`
(metric is **balanced accuracy** = mean per-class recall; euclid uses **path accuracy** via
`experiments/euclid_tree/path_scores.py`).

## 6. Run the local open VLM (GPU) — zero-shot or fine-tuned

Point the eval script at a locally downloaded Qwen-VL model; add an adapter path to evaluate a
fine-tuned run:

```bash
# zero-shot (no adapter)
MODEL_LOCAL_PATH=/path/to/Qwen3.5-VL-4B \
python finetuning/vlm_lora/eval_qwen35_local.py

# fine-tuned (with a LoRA adapter)
MODEL_LOCAL_PATH=/path/to/Qwen3.5-VL-4B \
QWEN_ADAPTER_PATH=/path/to/adapter_dir \
python finetuning/vlm_lora/eval_qwen35_local.py
```

## 7. Fine-tune an open VLM (GPU, LoRA)

```bash
TRAIN_SAMPLE_CSV=tasks/rings/data/train_100.csv \
ADAPTER_OUT=/path/to/output_adapter \
MODEL_LOCAL_PATH=/path/to/Qwen3.5-VL-4B \
TASK_PROMPT="Does this galaxy image show a ring? Reply yes or no." \
python finetuning/vlm_lora/train_lora_qwen35.py
```

The full recipe (ranks, epochs, batch sizes per task) is in
`finetuning/vlm_lora/RECIPE.md`. `train_lora.py` is the equivalent trainer for Qwen2-VL-7B.
Adapters retrain in ~15 min on one A100 and are **not** shipped with the source.

## 8. CNN baseline (GPU)

The specialist encoder baseline is the vendored **gz-evo / Zoobot** code under
`finetuning/encoders/`. See `finetuning/encoders/README.md` for the exact fine-tune commands
and the per-task encoder used (MaxViT / ConvNeXt / SoViT). Per-seed **test** predictions are
committed under `finetuning/encoders/results/` so the reported CNN numbers are verifiable
without re-running.

## 9. Reproduce the figures

```bash
python analysis/plot_combined_all_tasks.py     # main results figure
```

Other plotting scripts live in `analysis/`; generated figures are in `docs/figures/`.

---

## Notes

- **No GPU?** You can still run every zero-shot API evaluation and all plotting — only the
  local open-model and CNN steps need a GPU.
- **Config / paths:** machine-specific locations are read from environment variables
  (`GVB_DATA_DIR`, `GVB_MODEL_DIR`, `GVB_ADAPTER_DIR`, `GVB_SCRATCH`) — see `config.py`.
- **HPC / SLURM (Narval):** cluster-specific setup, module loads, and staging are documented
  in `docs/running-on-hpc.md`; the `finetuning/**/jobs/*.sbatch` files are the as-run record.
