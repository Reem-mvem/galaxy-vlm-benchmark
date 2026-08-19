#!/usr/bin/env bash
#SBATCH --job-name=qwen_lora_vision2938w
#SBATCH --account=def-bovy_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/train_vision2938w_%j.out

pwd; hostname; date
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

echo "=== staging model ==="
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/

source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"

# --- experiment config: unfreeze vision (all-linear) + 788 balanced samples ---
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV=/scratch/reem1/finetuning/strong_lens_train_2938.csv
export ADAPTER_OUT=/scratch/reem1/finetuning/adapters/strong_lens_vision2938w
export EPOCHS=3
export POS_WEIGHT=6.46

nvidia-smi | head -12
echo "=== training (all-linear, 788 samples) ==="
python -u /scratch/reem1/finetuning/train_lora.py
