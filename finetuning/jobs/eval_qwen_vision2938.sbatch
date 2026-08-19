#!/usr/bin/env bash
#SBATCH --job-name=qwen_vision2938
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_vision2938_%j.out

pwd; hostname; date

module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

# stage the model to node-local NVMe (large files, fast; see memory: never stage the venv)
echo "=== staging model ==="
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"
time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/

source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
export QWEN_ADAPTER_PATH=/scratch/reem1/finetuning/adapters/strong_lens_vision2938

nvidia-smi | head -12
cd /home/reem1/scratch/finetuning/strong_lens
echo "=== running zero-shot eval (n_sample=${N_SAMPLE:-all}) ==="
python -u eval_qwen_local.py --output results/strong_lens_results_qwen_vision2938.csv
