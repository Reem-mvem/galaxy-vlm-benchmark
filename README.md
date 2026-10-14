# Galaxy VLM Benchmark

Can a vision-language model (VLM) classify galaxy morphology as well as a purpose-built
CNN — and does prompting or fine-tuning win when labels are scarce? This repo benchmarks
**fine-tuned CNNs**, **zero-shot APIs** (Gemini, GPT, Qwen-397B), and **fine-tuned open
VLMs** (Qwen2-VL / Qwen3.5-VL, LoRA) across six galaxy tasks, with a focus on the
low-label regime common to new surveys.

## Tasks

| Task | Type | Source |
|---|---|---|
| `which_lsb` | 4-class (tidal-debris morphology) | Galaxy Zoo / DESI |
| `is_lsb` | binary (tidal / not-tidal) | Galaxy Zoo / DESI |
| `strong_lens` | binary (lens / not) | Euclid |
| `rings` | binary (ring / not) | HuggingFace `mwalmsley/gz_rings` |
| `euclid` (Morphology) | 13-question decision tree, scored by **path accuracy** | HuggingFace `mwalmsley/gz_euclid` |
| `streams_shells_neither` | 3-class (real-world) | HuggingFace `astronolan/streams_shells_neither` (DESI DR10) |

Metric is **balanced accuracy** (mean per-class recall); euclid uses **path accuracy**
through the morphology tree. Every fine-tuned model is trained on the **same 100 labels
per task**; zero-shot models are untrained.

## Headline results

Balanced accuracy averaged over the four balanced tasks (which_lsb, is_lsb, strong_lens, rings):

| Model | Avg. acc | Type |
|---|---|---|
| **Fine-tuned Qwen3.5-4B** (100 labels) | **72.7** | local, LoRA |
| **Fine-tuned CNN** (100 labels) | **71.1** | local (baseline) |
| Gemini 3.5 Flash | 69.8 | zero-shot API |
| Qwen3.5-397B-A17B | 56.3 | zero-shot API |
| Qwen3.5-4B | 55.9 | zero-shot, local |
| GPT-5.6-sol | 53.3 | zero-shot API |

**Takeaways:** fine-tuning a small open VLM on ~100 labels is the most accurate *and*
free-to-run option; the specialist CNN stays strongest on the hardest tasks; zero-shot
Gemini is a strong, label-free fallback; and **more reasoning does not mean higher
accuracy** — a Gemini thinking-budget sweep *helped* the 3-class real-world task but
*hurt* the binary ring task. See `docs/` and `figures/`.

## Repository layout

```
vlm_eval/            # shared eval framework (providers, runner, metrics, catalogs)
tasks/<task>/        # task.py + data/ (test/train ID lists; images not committed) + results/
    results/
      VLMs_results/                    # zero-shot: Gemini, GPT, Qwen-397B, local zero-shot
      Fine-tuning_results_Qwen2VL-7B/  # per-model fine-tune results
      Fine-tuning_results_Qwen3.5-4B/
      Fine-tuning_results_Qwen3.5-9B/
tasks/euclid/
    euclid_VLMs/     # zero-shot API runs (Gemini, GPT, Qwen-397B) + shared schema and path scoring
    fine-tuned/      # local Qwen3.5-4B (zero-shot and fine-tuned) runs
    encoder_baseline/# encoder path accuracy and label-efficiency curve
finetuning/
    vlm_lora/        # LoRA trainers (Qwen2-VL-7B and Qwen3.5-VL-4B/9B) + eval runners
    adapters/        # reported Qwen3.5-4B adapters (configs + logs) and their jobs
    encoders/        # gz-evo clone (specialist encoder baseline) + jobs + per-seed test predictions
experiments/         # side experiments not in the headline results (reasoning sweeps,
                     # prompt variants, ablations, cost measurement), one folder per topic
analysis/            # plotting scripts
figures/             # generated figures
config.py            # central paths (env-overridable)
docs/                # methodology, running-on-hpc, results
```

## Data & weights

- **Images are not committed**; they come from HuggingFace (`mwalmsley/gz_rings`,
  `mwalmsley/gz_euclid`, `astronolan/streams_shells_neither`, plus the Galaxy Zoo / Euclid
  catalogs). The exact test and training IDs used are committed in `tasks/<task>/data/`.
- **LoRA adapter weights are not committed**; their configs, jobs and training/eval logs are
  in `finetuning/adapters/`. Recipe in `finetuning/vlm_lora/RECIPE.md`.
- **Per-seed encoder test predictions are committed** under `finetuning/encoders/results/` so
  every reported test number is verifiable; the large train/validation predictions are omitted.

## Install

```bash
pip install -e .        # installs the vlm_eval package + deps (requirements.txt)
```

Set API keys in a `.env` (see `.env.example`): `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY` (Qwen-397B is served via OpenRouter).

## Compute needs

| Step | GPU |
|---|---|
| Zero-shot APIs (Gemini/GPT/Qwen-397B) | none (API) |
| VLM fine-tune (LoRA, per task) | 1 GPU (A100 20–40 GB), ~15 min |
| VLM local eval (zero-shot or fine-tuned) | 1 GPU / MIG |
| CNN train + predict (gz-evo) | 1 GPU |

HPC (Narval/SLURM) specifics are in `docs/running-on-hpc.md`.

## Attribution

The specialist encoder baseline uses **gz-evo / Zoobot** (Mike Walmsley et al.) — vendored under
`finetuning/encoders/gz-evo/` with its original `LICENSE`; upstream: https://github.com/mwalmsley/gz-evo
(commit `8a0bdca`). Our changes are listed in `finetuning/encoders/README.md`.
The `streams_shells_neither` task uses the mentor-provided DESI dataset published at
`astronolan/streams_shells_neither`.
