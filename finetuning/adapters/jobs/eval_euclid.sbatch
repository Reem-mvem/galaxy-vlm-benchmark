#!/usr/bin/env bash
#SBATCH --job-name=euclid_vlm_ev
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/euclid_vlm/eval_%j.out

# usage: sbatch --export=ALL,SIZE=100 eval_euclid.sbatch   (SIZE=zeroshot for base model)
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
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export EVAL_CSV="$EU/eval_1000.csv"
if [ "${SIZE:-100}" = "zeroshot" ]; then
  export OUT="$EU/euclid_vlm_zeroshot_answers.csv"
else
  export QWEN_ADAPTER_PATH="$NEW/adapters/gz_euclid_qwen35_cnn99"
  export OUT="$EU/euclid_vlm_finetuned_answers.csv"
fi
nvidia-smi -L
cd "$EU"
echo "=== eval euclid VLM -> $OUT ==="
"$VENV/bin/python" -u eval_euclid_vlm.py
