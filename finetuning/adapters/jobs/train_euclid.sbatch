#!/usr/bin/env bash
#SBATCH --job-name=euclid_vlm_tr
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm/train_%j.out

pwd; hostname; date
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
Q2=/home/reem1/scratch/finetuning_Qwen2-VL-7B
NEW=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B
EU=$NEW/euclid_vlm

echo "=== extract venv ==="; time tar xf "$Q2/venv_full.tar" -C "$SLURM_TMPDIR"
VENV="$SLURM_TMPDIR/.venv"
echo "=== stage 4B model ==="; mkdir -p "$SLURM_TMPDIR/q35"; time cp -rL "$NEW/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export TARGET_MODULES=all-linear
export TRAIN_SAMPLE_CSV="$EU/train_cnn99.csv"                # SAME 99 labels the CNN (seed0) got
export ADAPTER_OUT="$NEW/adapters/gz_euclid_qwen35_cnn99"
export BATCH_SIZE=4 GRAD_ACCUM=2 EPOCHS=12 LORA_R=8       # eff-batch 8, ~150 updates on 100
export TASK_PROMPT="$(cat "$EU/prompt.txt")"              # 13-question prompt (matches eval)
unset ANSWER_YES_LABEL                                     # target = the 13-line answer text

nvidia-smi -L
cd "$NEW"
echo "=== fine-tune Qwen3.5-4B on gz_euclid (13-question tree) ==="
"$VENV/bin/python" -u train_lora_qwen35.py
test -f "$ADAPTER_OUT/adapter_model.safetensors" && echo "ADAPTER SAVED ✓" || { echo "ADAPTER MISSING"; exit 1; }
