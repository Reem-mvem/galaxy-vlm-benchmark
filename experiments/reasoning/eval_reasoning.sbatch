#!/usr/bin/env bash
#SBATCH --job-name=q35_reason
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/jobs/eval_reasoning_%j.out
module load StdEnv/2023 gcc python/3.11; module load arrow/24.0.0
tar xf /home/reem1/scratch/finetuning_Qwen2-VL-7B/venv_full.tar -C "$SLURM_TMPDIR"
mkdir -p "$SLURM_TMPDIR/q35"; cp -rL "/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/Qwen3.5-4B"/. "$SLURM_TMPDIR/q35"/
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/q35"
export QWEN_ADAPTER_PATH=/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/adapters/which_lsb_qwen35_100_r16
export QWEN_MAX_NEW_TOKENS=512
cd /home/reem1/scratch/finetuning_Qwen3.5-VL-4B
echo "=== which_lsb full-reasoning eval (r16 adapter, 512 tokens, all test) ==="; date
"$SLURM_TMPDIR/.venv/bin/python" -u eval_qwen35_local.py --task which_lsb     --output /home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/results/which_lsb_results_qwen35_r16_reasoning.csv
date
