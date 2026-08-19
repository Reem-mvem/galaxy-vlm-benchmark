#!/usr/bin/env bash
#SBATCH --job-name=sl100_retry
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/sl100_retry_%j.out

pwd; hostname; date
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/
source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
REPO=/home/reem1/scratch/galaxy-vlm-benchmark
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV="$REPO/tasks/strong_lens/data/train_100.csv"
export ADAPTER_OUT="/scratch/reem1/finetuning/adapters/strong_lens_vision_100_retry"
# stronger low-data config: eff batch 4 (4x1), 10 epochs -> ~250 updates on 100 imgs
export BATCH_SIZE=4 GRAD_ACCUM=1 EPOCHS=10
export TASK_PROMPT="This image is from the Euclid space telescope. Does it show a strong gravitational lens (a confident or probable lens candidate, as an expert would judge)? Reply with one word: yes or no."
export ANSWER_YES_LABEL="grade_a_or_b_lens"
nvidia-smi -L
cd "$REPO"
echo "=== strong_lens 100 RETRY: eff batch 4, 10 epochs ==="
python -u finetuning/train_lora.py
echo "=== done ==="; ls -la "$ADAPTER_OUT" 2>/dev/null && echo "ADAPTER SAVED ✓"
