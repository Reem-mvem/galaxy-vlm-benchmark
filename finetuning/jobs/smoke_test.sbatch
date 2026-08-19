#!/usr/bin/env bash
#SBATCH --job-name=vlm_smoke
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/reem1/finetuning/logs/smoke_%j.out

pwd; hostname; date

module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

# --- stage BOTH venv and model to node-local NVMe ---
# Lustre is pathologically slow at the random/mmap reads of importing torch and
# loading a sharded model; sequential cp to $SLURM_TMPDIR is fast, and then all
# reads happen from local disk.
# NOTE: do NOT stage the venv — copying its thousands of small files off Lustre
# is pathologically slow (failed at 34min and 17min). Importing from it in place
# works (proven in run 1). Only the model is staged (large files, ~48s).
echo "=== staging model -> $SLURM_TMPDIR/qwen2vl ==="
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"
time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/
echo "staged model=$(du -sh $SLURM_TMPDIR/qwen2vl|cut -f1)"

source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"

nvidia-smi | head -12
echo "=== running smoke test ==="
python -u /scratch/reem1/finetuning/smoke_test.py
