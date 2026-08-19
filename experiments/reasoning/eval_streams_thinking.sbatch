#!/usr/bin/env bash
#SBATCH --job-name=ss_qwen_think
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/reem1/streams_shells_neither/results/ss_qwen_think_%j.out

pwd; hostname; date
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
Q2=/home/reem1/scratch/finetuning_Qwen2-VL-7B
NEW=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B
SS=/scratch/reem1/streams_shells_neither

echo "=== extract venv ==="; time tar xf "$Q2/venv_full.tar" -C "$SLURM_TMPDIR"
VENV="$SLURM_TMPDIR/.venv"
echo "=== stage 4B model ==="; mkdir -p "$SLURM_TMPDIR/q35"; time cp -rL "$NEW/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export ENABLE_THINKING=1               # thinking ON
export FORCE_CONCLUDE=1                 # budget-force a committed answer if it never closes </think>
export QWEN_MAX_NEW_TOKENS=2048
export OUT="$SS/results/qwen_zeroshot_THINK.csv"     # resumable; compare vs qwen_zeroshot_predictions.csv (33.3)

nvidia-smi -L
echo "=== Qwen3.5-4B zero-shot THINKING on streams/shells (30 imgs) ==="
"$VENV/bin/python" -u "$SS/scripts/eval_streams_thinking.py"
