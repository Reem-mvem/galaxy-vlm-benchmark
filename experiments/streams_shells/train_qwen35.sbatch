#!/usr/bin/env bash
#SBATCH --job-name=q35_train
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/jobs/train_%x_%j.out

# usage: sbatch --export=ALL,TASK=which_lsb,SIZE=100 train_qwen35.sbatch
: "${TASK:?set TASK=is_lsb|which_lsb|strong_lens|rings}"
: "${SIZE:?set SIZE=100 (train_<SIZE>.csv)}"
pwd; hostname; date; echo "TASK=$TASK SIZE=$SIZE"
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

Q2=/home/reem1/scratch/finetuning_Qwen2-VL-7B
NEW=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B
REPO=/home/reem1/scratch/galaxy-vlm-benchmark

echo "=== extract venv from tarball -> local NVMe (fast imports) ==="; time tar xf "$Q2/venv_full.tar" -C "$SLURM_TMPDIR"
VENV="$SLURM_TMPDIR/.venv"
echo "=== stage 4B model ==="; mkdir -p "$SLURM_TMPDIR/q35"; time cp -rL "$NEW/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV="$REPO/tasks/${TASK}/data/train_${SIZE}.csv"
export ADAPTER_OUT="$NEW/adapters/${TASK}_qwen35_${SIZE}${RUNTAG:-}"
# config (override on the command line to push rank/epochs). Default matches the 7B which_lsb@100 recipe.
export BATCH_SIZE="${BATCH_SIZE:-4}" GRAD_ACCUM="${GRAD_ACCUM:-2}" EPOCHS="${EPOCHS:-6}" LORA_R="${LORA_R:-8}"

case "$TASK" in
  is_lsb)      export TASK_PROMPT="Does this galaxy image show faint debris (low-surface-brightness features around the main galaxy body)? Reply with one word: yes or no."; export ANSWER_YES_LABEL="tidal" ;;
  strong_lens) export TASK_PROMPT="This image is from the Euclid space telescope. Does it show a strong gravitational lens (a confident or probable lens candidate, as an expert would judge)? Reply with one word: yes or no."; export ANSWER_YES_LABEL="grade_a_or_b_lens" ;;
  rings)       export TASK_PROMPT="Does this galaxy image show a ring (a circular feature within or around the galaxy)? Reply with one word: yes or no."; export ANSWER_YES_LABEL="ring" ;;
  which_lsb)   export TASK_PROMPT="This galaxy image shows faint debris. Classify the debris morphology. Reply with one word: arm, stream, shell, or diffuse."; unset ANSWER_YES_LABEL ;;
  streams_shells_neither) export TASK_PROMPT="This is a galaxy image from an astronomical survey. Does it show a stellar stream, a shell, or neither? Reply with one word: stream, shell, or neither."; unset ANSWER_YES_LABEL ;;
esac

nvidia-smi -L
cd "$NEW"
echo "=== fine-tune Qwen3.5-4B: $TASK @ $SIZE ==="
"$VENV/bin/python" -u train_lora_qwen35.py
test -f "$ADAPTER_OUT/adapter_model.safetensors" && echo "ADAPTER SAVED ✓" || { echo "ADAPTER MISSING — training failed"; exit 1; }
