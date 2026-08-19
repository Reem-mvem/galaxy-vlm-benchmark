# VLM LoRA fine-tuning recipe

All open VLMs are fine-tuned with the same lightweight LoRA recipe (TRL `SFTTrainer` +
PEFT), differing only by the model/architecture classes.

## Two trainers, one recipe
- `train_lora.py` — **Qwen2-VL-7B** (`Qwen2VLForConditionalGeneration` / `Qwen2VLProcessor`).
- `train_lora_qwen35.py` — **Qwen3.5-VL-4B and 9B** (generic `AutoModelForImageTextToText` /
  `AutoProcessor`; `image_token_id` read from config). The 9B uses this same script, just
  pointed at the 9B weights via `MODEL_LOCAL_PATH`.

The Qwen3.5 trainer is adapted from the Qwen2-VL one — only the model/processor classes and
the image-token source changed; the env-driven interface, collator, and LoRA config are
identical.

## Hyperparameters
- **LoRA:** rank `r = 8`, `α = 2·r` (= 16), dropout 0.05, `target_modules = all-linear`
  (adapts the vision layers as well as the language ones), `task_type = CAUSAL_LM`.
- **Optimisation:** lr `2e-4`, warmup 0.03, bf16, gradient checkpointing, `max_grad_norm 0.3`.
- **Objective:** direct answer, **no reasoning** (`enable_thinking=False`); loss is masked to
  the answer tokens only (prompt / padding / image placeholders are ignored).
- **Data:** 100 labelled images per task (matched to the CNN and across model sizes).

### Per-task batch / epochs (effective batch = micro × accum)
| Task(s) | eff. batch | epochs | ≈ updates |
|---|---|---|---|
| which_lsb, is_lsb | 8 (4×2) | 6 | ~75 |
| strong_lens, rings | 4 (4×1) | 10 | ~250 |
| euclid (99 labels, 13-line target) | 8 (4×2) | 12 | ~150 |

Ablation: the 4B was re-run at `r = 16` on rings + strong_lens (`LORA_R=16`).

## Run
Set `TRAIN_SAMPLE_CSV`, `ADAPTER_OUT`, `TASK_PROMPT`, `MODEL_LOCAL_PATH` (+ `ANSWER_YES_LABEL`
for binary tasks), then `python train_lora_qwen35.py`. Example jobs in `../jobs/`. Adapters
(~78 MB) are not committed — retrain in ~15 min on one GPU.
