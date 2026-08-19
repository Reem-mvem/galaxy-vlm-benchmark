#!/usr/bin/env bash
#SBATCH --job-name=qwen_islsb_tr
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_islsb_tr_%j.out

# usage: sbatch --export=ALL,SIZE=300 train_is_lsb.sbatch
: "${SIZE:?set SIZE=300 (balanced training subset)}"
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

REPO=/home/reem1/scratch/galaxy-vlm-benchmark
export TRAIN_SAMPLE_CSV="$REPO/tasks/is_lsb/data/train_${SIZE}.csv"
export ADAPTER_OUT="/scratch/reem1/finetuning/adapters/is_lsb_vision_${SIZE}"
export TASK_PROMPT="Does this galaxy image show faint debris (low-surface-brightness features around the main galaxy body)? Reply with one word: yes or no."
export ANSWER_YES_LABEL="tidal"
export TARGET_MODULES="all-linear"
# batch/epochs from train_lora.py defaults (micro 4 x accum 4 = eff 16, 4 epochs)

nvidia-smi | head -12
cd "$REPO"
echo "=== is_lsb fine-tune, SIZE=$SIZE (balanced) ==="
python -u finetuning/train_lora.py
echo "=== adapter contents ==="; ls -la "$ADAPTER_OUT" 2>/dev/null && echo "ADAPTER SAVED ✓"
