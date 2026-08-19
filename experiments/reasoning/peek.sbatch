#!/usr/bin/env bash
#SBATCH --job-name=q35_peek
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/jobs/peek_%j.out
module load StdEnv/2023 gcc python/3.11; module load arrow/24.0.0
tar xf /home/reem1/scratch/finetuning_Qwen2-VL-7B/venv_full.tar -C "$SLURM_TMPDIR"
mkdir -p "$SLURM_TMPDIR/q35"; cp -rL "/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export QWEN_ADAPTER_PATH=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/adapters/which_lsb_qwen35_100_r16
"$SLURM_TMPDIR/.venv/bin/python" -u /home/reem1/scratch/finetuning_Qwen3.5-VL-4B/peek_reasoning.py
