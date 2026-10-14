#!/usr/bin/env bash
#SBATCH --job-name=q35_eval
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/jobs/eval_%x_%j.out

# usage: sbatch --export=ALL,TASK=which_lsb,SIZE=100 eval_qwen35.sbatch   (SIZE=zeroshot for base)
: "${TASK:?set TASK=...}"; : "${SIZE:?set SIZE=100|zeroshot}"
pwd; hostname; date; echo "TASK=$TASK SIZE=$SIZE"
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
Q2=/home/reem1/scratch/finetuning_Qwen2-VL-7B
NEW=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B
REPO=/home/reem1/scratch/galaxy-vlm-benchmark

echo "=== extract venv from tarball ==="; time tar xf "$Q2/venv_full.tar" -C "$SLURM_TMPDIR"
VENV="$SLURM_TMPDIR/.venv"
echo "=== stage 4B model ==="; mkdir -p "$SLURM_TMPDIR/q35"; time cp -rL "$NEW/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
if [ "$SIZE" != "zeroshot" ]; then
  export QWEN_ADAPTER_PATH="$NEW/adapters/${TASK}_qwen35_${SIZE}${RUNTAG:-}"
  OUT="$REPO/tasks/${TASK}/results/${TASK}_results_qwen35_vision${SIZE}${RUNTAG:-}.csv"
else
  OUT="$REPO/tasks/${TASK}/results/${TASK}_results_qwen35_zeroshot.csv"
fi
nvidia-smi -L
cd "$NEW"
echo "=== eval Qwen3.5-4B: $TASK ($SIZE) -> $OUT ==="
"$VENV/bin/python" -u eval_qwen35_local.py --task "$TASK" --output "$OUT"
