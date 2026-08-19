#!/usr/bin/env bash
#SBATCH --job-name=sl100_retry_ev
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/sl100_retry_ev_%j.out
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/
source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
export QWEN_ADAPTER_PATH="/scratch/reem1/finetuning/adapters/strong_lens_vision_100_retry"
cd /home/reem1/scratch/finetuning/strong_lens
echo "=== strong_lens 100-retry eval (999) ==="
python -u eval_qwen_local.py --output results/strong_lens_results_qwen_vision100_retry.csv
