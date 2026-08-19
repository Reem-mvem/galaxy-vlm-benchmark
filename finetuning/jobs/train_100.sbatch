#!/usr/bin/env bash
#SBATCH --job-name=qwen_tr100
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_tr100_%x_%j.out

# usage: sbatch --export=ALL,TASK=is_lsb ...   TASK in {is_lsb,which_lsb,strong_lens}
: "${TASK:?set TASK=is_lsb|which_lsb|strong_lens}"
pwd; hostname; date; echo "TASK=$TASK (100 samples)"
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
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV="$REPO/tasks/${TASK}/data/train_100.csv"
export ADAPTER_OUT="/scratch/reem1/finetuning/adapters/${TASK}_vision_100"
# low-data config: effective batch 8 (4x2), 6 epochs (~75 updates on 100 samples)
export BATCH_SIZE=4 GRAD_ACCUM=2 EPOCHS=6

case "$TASK" in
  is_lsb)
    export TASK_PROMPT="Does this galaxy image show faint debris (low-surface-brightness features around the main galaxy body)? Reply with one word: yes or no."
    export ANSWER_YES_LABEL="tidal" ;;
  strong_lens)
    export TASK_PROMPT="This image is from the Euclid space telescope. Does it show a strong gravitational lens (a confident or probable lens candidate, as an expert would judge)? Reply with one word: yes or no."
    export ANSWER_YES_LABEL="grade_a_or_b_lens" ;;
  which_lsb)
    export TASK_PROMPT="This galaxy image shows faint debris. Classify the debris morphology. Reply with one word: arm, stream, shell, or diffuse."
    unset ANSWER_YES_LABEL ;;   # multi-class: answer = class word
esac

nvidia-smi | head -12
cd "$REPO"
echo "=== fine-tune $TASK on 100 samples ==="
python -u finetuning/train_lora.py
echo "=== done ==="; ls -la "$ADAPTER_OUT" 2>/dev/null && echo "ADAPTER SAVED ✓"
