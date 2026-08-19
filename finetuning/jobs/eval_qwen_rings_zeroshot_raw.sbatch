#!/usr/bin/env bash
#SBATCH --job-name=qwen_rings_zs_raw
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_rings_zs_raw_%j.out

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
# no QWEN_ADAPTER_PATH -> zero-shot

nvidia-smi | head -12
cd /home/reem1/scratch/galaxy-vlm-benchmark
echo "=== rings zero-shot (RAW images), same balanced-1000 seed=42 as Gemini ==="
python -u finetuning/eval_qwen_local.py --task rings --n_sample 1000 --seed 42 \
    --output tasks/rings/results/rings_results_qwen_local_raw.csv
