# Euclid — local Qwen3.5-4B (zero-shot and fine-tuned)

Qwen3.5-4B run locally on the GZ Euclid 13-question decision tree, as the zero-shot base model
and fine-tuned with LoRA on the same 99 labelled galaxies the CNN (seed 0) was trained on.

## Files

| File | Purpose |
|---|---|
| `build_euclid_vlm_matched.py` | builds the training set from `cnn_seed0_ids.txt` (the CNN's 79 train + 20 val galaxies) → `finetune_train_99.csv` |
| `cnn_seed0_ids.txt` | the 99 galaxy ids |
| `finetune_train_99.csv` | training images + 13-line `question: answer` targets (volunteer majority answers) |
| `prompt.txt` | the 13-question prompt, used for training and evaluation |
| `train_lora_qwen35.py` | LoRA trainer |
| `train_euclid.sbatch` | training job |
| `eval_euclid_vlm.py` | evaluation: generates the 13 answers per galaxy, parses them, scores path accuracy |
| `eval_euclid.sbatch` | evaluation job (`SIZE=100` fine-tuned, `SIZE=zeroshot` base model) |
| `eval_1000.csv` | the 1000 evaluation galaxies |
| `results/` | per-galaxy raw reply + the 13 parsed answers, for each model |

Scoring uses the shared Euclid code (`euclid_schema.py`, `path_scores.py`), the same used for
the API models and the CNN (see `../euclid_VLMs/shared/`). The trained adapter is in
`/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/adapters/gz_euclid_qwen35_cnn99`.

## Settings

- **Fine-tuning:** LoRA rank 8, alpha 16, dropout 0.05, all linear layers (vision tower included),
  effective batch 8 (4 × 2), 12 epochs, learning rate 2e-4, bf16.
- **Evaluation:** thinking off, greedy decoding, up to 220 new tokens. An answer is read from each
  `question: answer` line; a missing line or an answer outside the allowed options is recorded as
  `INVALID` and scored wrong.

## How to run

```bash
sbatch train_euclid.sbatch                                   # fine-tune
sbatch --export=ALL,SIZE=100 eval_euclid.sbatch              # evaluate fine-tuned
sbatch --export=ALL,SIZE=zeroshot eval_euclid.sbatch         # evaluate zero-shot
```

## Results (path accuracy, 1000 galaxies)

| Model | Path accuracy | INVALID answers (of 13,000) |
|---|---|---|
| Qwen3.5-4B fine-tuned | **0.7370** | 0 |
| Qwen3.5-4B zero-shot | **0.6077** | 2,850 (179 on the volunteers' path) |
