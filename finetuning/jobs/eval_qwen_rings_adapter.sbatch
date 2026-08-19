#!/usr/bin/env bash
#SBATCH --job-name=qwen_rings_ev
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_rings_ev_%j.out

# usage: sbatch --export=ALL,SIZE=500 eval_qwen_rings_adapter.sbatch
: "${SIZE:?set SIZE=100|250|500|1000|2000}"
pwd; hostname; date; echo "SIZE=$SIZE"
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

echo "=== staging model ==="
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/

source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
export QWEN_ADAPTER_PATH="/scratch/reem1/finetuning/adapters/rings_vision_${SIZE}"

nvidia-smi | head -12
cd /home/reem1/scratch/galaxy-vlm-benchmark
echo "=== rings base+adapter eval (balanced 1000, seed 42, RAW) — adapter: $QWEN_ADAPTER_PATH ==="
python -u finetuning/eval_qwen_local.py --task rings --n_sample 1000 --seed 42 \
    --output "tasks/rings/results/rings_results_qwen_vision${SIZE}.csv"
