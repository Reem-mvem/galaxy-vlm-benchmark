#!/usr/bin/env bash
#SBATCH --job-name=think_which
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/jobs/think_%x_%j.out

# usage: sbatch --export=ALL,ENABLE_THINKING=1 thinking_which.sbatch   (0 = no-think baseline)
: "${ENABLE_THINKING:?set ENABLE_THINKING=1 (think) or 0 (no-think)}"
pwd; hostname; date; echo "ENABLE_THINKING=$ENABLE_THINKING"
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0
Q2=/home/reem1/scratch/finetuning_Qwen2-VL-7B
NEW=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B

echo "=== extract venv ==="; time tar xf "$Q2/venv_full.tar" -C "$SLURM_TMPDIR"
VENV="$SLURM_TMPDIR/.venv"
echo "=== stage 4B base model (adapter, if any, is loaded later by the python script) ==="; mkdir -p "$SLURM_TMPDIR/q35"; time cp -rL "$NEW/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/
[ -n "$QWEN_ADAPTER_PATH" ] && echo "  will load adapter: $QWEN_ADAPTER_PATH" || echo "  no adapter -> zero-shot"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
# QWEN_ADAPTER_PATH (optional) + OUT can be passed via --export; else default to zero-shot names.
TAG="zeroshot"; [ -n "$QWEN_ADAPTER_PATH" ] && TAG="ft"
if [ "$ENABLE_THINKING" = "1" ]; then
  export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-2048}"
  : "${OUT:=/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/results/which_lsb_4b_${TAG}_THINK.csv}"
else
  export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-10}"
  : "${OUT:=/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/results/which_lsb_4b_${TAG}_NOTHINK.csv}"
fi
export OUT
nvidia-smi -L
cd "$NEW"
echo "=== which_lsb 4B zero-shot, thinking=$ENABLE_THINKING -> $OUT ==="
"$VENV/bin/python" -u eval_which_thinking.py
