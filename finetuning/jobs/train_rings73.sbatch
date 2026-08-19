#!/usr/bin/env bash
#SBATCH --job-name=rings73_tr
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/rings73_tr_%j.out
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/
source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on the MIG
REPO=/home/reem1/scratch/galaxy-vlm-benchmark
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV="$REPO/tasks/rings/data/train_73.csv"
export ADAPTER_OUT="/scratch/reem1/finetuning/adapters/rings_vision_73"
# leaner for the 20GB MIG: micro-batch 2 x accum 2 = eff batch 4, 10 epochs
export BATCH_SIZE=4 GRAD_ACCUM=1 EPOCHS=10
export TASK_PROMPT="Does this galaxy image show a ring (a circular feature within or around the galaxy)? Reply with one word: yes or no."
export ANSWER_YES_LABEL="ring"
nvidia-smi -L
cd "$REPO"; echo "=== rings@73 train (micro-batch 2, eff 4, 10 ep) ==="
python -u finetuning/train_lora.py
# real success check: adapter file must exist, else exit non-zero (so afterok blocks the eval)
test -f "$ADAPTER_OUT/adapter_model.safetensors" && echo "ADAPTER SAVED ✓" || { echo "ADAPTER MISSING — training failed"; exit 1; }
