# Running on HPC (Narval / SLURM)

The reusable code reads all machine-specific locations from environment variables, so the
same scripts run on a laptop, a workstation, or the cluster. The `finetuning/**/jobs/` and
`finetuning/encoders/jobs/` sbatch files are the **as-run record** on Narval and keep their
original paths for provenance; change the exports at the top to your machine.

## Paths (see `config.py`)

| Variable | Meaning |
|---|---|
| `GVB_DATA_DIR` | task catalogs + images (default `<repo>/tasks`) |
| `GVB_MODEL_DIR` | local VLM weights |
| `GVB_ADAPTER_DIR` | LoRA adapters (off-git) |
| `GVB_SCRATCH` | fast scratch for staging (`$SLURM_TMPDIR` on Narval) |

## Per-script env vars

**VLM fine-tune** (`finetuning/vlm_lora/train_lora*.py`): `TRAIN_SAMPLE_CSV`,
`ADAPTER_OUT`, `TASK_PROMPT`, `MODEL_LOCAL_PATH`, optional `IMAGES_DIR`, `TARGET_MODULES`
(default `all-linear`), `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`, `LORA_R`, `ANSWER_YES_LABEL`
(binary tasks only).

**VLM eval** (`finetuning/vlm_lora/eval_qwen*_local.py`): `MODEL_LOCAL_PATH`,
`QWEN_ADAPTER_PATH` (set = fine-tuned, unset = zero-shot).

**Encoders / CNN baseline** (`finetuning/encoders/`): `GZEVO_TRAIN_CSV` (force a matched training subset by id),
`RINGS_CATALOG`, `divisor` (label-efficiency subsampling).

## GPU requirements

| Step | GPU | Wall time |
|---|---|---|
| Zero-shot API (Gemini / GPT / Qwen-397B) | none | minutes (network-bound) |
| VLM LoRA fine-tune (per task, 100 labels) | 1 × A100 (20–40 GB) | ~15 min |
| VLM local eval (1000 test images) | 1 × A100 / MIG (20 GB) | binary ~30 min · 13-question ~1.5 h |
| CNN train + predict (gz-evo) | 1 × A100 | ~7–45 min (scales with label count) |

## Narval-specific gotchas

- **Modules:** `module load StdEnv/2023 gcc python/3.11` then `module load arrow/24.0.0`
  (pyarrow comes from the arrow module, **not** pip). Do not pipe `module load | tail` — a
  subshell loses the environment.
- **Lustre is slow for many-file imports.** Stage the model and venv to `$SLURM_TMPDIR`
  (NVMe): extract a `venv_full.tar` and `cp -rL` the model dir before running. Never run the
  venv directly off `/scratch`.
- **Offline compute nodes:** set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`
  and pre-download datasets/models on a login node. API calls (Gemini/OpenAI/OpenRouter) run
  from a login node (or off-cluster), not compute nodes.
- No `bitsandbytes` / no quantization — training is bf16.
