# Qwen3.5-4B LoRA adapters (reported results)

The fine-tuned Qwen3.5-4B adapters behind the reported results, one folder per task, each with
its training and evaluation logs. All were trained with `../vlm_lora/train_lora_qwen35.py`
(LoRA rank 8, alpha 16, dropout 0.05, all linear layers, learning rate 2e-4, bf16, thinking off).

| Task | Adapter | Training images | Batch × epochs | Result file | Score |
|---|---|---|---|---|---|
| which_lsb | `which_lsb/which_lsb_qwen35_100` | 100 | 8 × 6 | `tasks/which_lsb/results/Fine-tuning_results_Qwen3.5-4B/which_lsb_results_qwen35_vision100.csv` | 63.6 |
| is_lsb | `is_lsb/is_lsb_qwen35_100` | 100 | 8 × 6 | `tasks/is_lsb/results/Fine-tuning_results_Qwen3.5-4B/is_lsb_results_qwen35_vision100.csv` | 94.3 |
| strong_lens | `strong_lens/strong_lens_qwen35_100` | 100 | 4 × 10 | `tasks/strong_lens/results/Fine-tuning_results_Qwen3.5-4B/strong_lens_results_qwen35_vision100.csv` | 64.1 |
| rings | `rings/rings_qwen35_100` | 100 | 4 × 10 | `tasks/rings/results/Fine-tuning_results_Qwen3.5-4B/rings_results_qwen35_vision100.csv` | 68.6 |
| streams | `streams_shells_neither/streams_shells_neither_qwen35_72` | 72 | 8 × 10 | `tasks/streams_shells_neither/results/Fine-tuning_results_Qwen3.5-4B/streams_shells_neither_results_qwen35_vision72.csv` | 83.3 |
| Euclid | `euclid/gz_euclid_qwen35_cnn99` | 99 | 8 × 12 | `tasks/euclid/fine-tuned/results/euclid_vlm_finetuned_answers.csv` | 73.7 (path accuracy) |

Scores are balanced accuracy (%) unless noted. Each task folder's `logs/` holds the SLURM
training log (`train_<job id>.out`) and evaluation log (`eval_<job id>.out`) of that run.

## Jobs (`jobs/`)

`train_qwen35.sbatch` / `eval_qwen35.sbatch` serve every classification task (the task is passed
at submit time); `train_euclid.sbatch` / `eval_euclid.sbatch` are the Euclid jobs. Submit
commands, from the logs:

```bash
# which_lsb, is_lsb (defaults: batch 4 x accum 2, 6 epochs)
sbatch --export=ALL,TASK=which_lsb,SIZE=100 train_qwen35.sbatch
sbatch --export=ALL,TASK=which_lsb,SIZE=100 eval_qwen35.sbatch
# strong_lens
sbatch --export=ALL,TASK=strong_lens,SIZE=100,GRAD_ACCUM=1,EPOCHS=10 train_qwen35.sbatch
# rings
sbatch --export=ALL,TASK=rings,SIZE=100,GRAD_ACCUM=1,EPOCHS=10 train_qwen35.sbatch
# streams
sbatch --export=ALL,TASK=streams_shells_neither,SIZE=72,EPOCHS=10 train_qwen35.sbatch
# Euclid
sbatch train_euclid.sbatch
sbatch --export=ALL,SIZE=100 eval_euclid.sbatch
```

(Evaluation uses the same `TASK`/`SIZE` with `eval_qwen35.sbatch`.) In the logs, the which_lsb,
is_lsb and strong_lens adapters appear under their original run names, ending in `_nt`. The job files use
the Narval paths of the original runs.

## Loading an adapter

```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("Qwen3.5-4B", torch_dtype="bfloat16")
model.load_adapter("finetuning/adapters/is_lsb/is_lsb_qwen35_100")
```
