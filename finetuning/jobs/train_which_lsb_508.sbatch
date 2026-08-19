#!/usr/bin/env bash
#SBATCH --job-name=qwen_whichlsb_train
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/train_whichlsb508_%j.out

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

# --- which_lsb full 508, 4-class, vision unfrozen ---
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV=/home/reem1/scratch/finetuning/which_lsb/data/train_catalog.csv
export IMAGES_DIR=/home/reem1/scratch/finetuning/which_lsb/data/images/train
export ADAPTER_OUT=/scratch/reem1/finetuning/adapters/which_lsb_vision508
export EPOCHS=3
export TASK_PROMPT="This galaxy image shows faint debris. Classify the debris morphology. Reply with one word: arm, stream, shell, or diffuse."
# multi-class: no ANSWER_YES_LABEL -> answer = the class word

nvidia-smi | head -12
echo "=== training which_lsb (all-linear, 508, 4-class) ==="
python -u /scratch/reem1/finetuning/train_lora.py
