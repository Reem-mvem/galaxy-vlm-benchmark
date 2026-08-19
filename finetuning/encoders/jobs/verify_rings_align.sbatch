#!/usr/bin/env bash
#SBATCH --job-name=rings_align
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/reem1/rings_runs/logs/rings_align_%j.out

pwd; hostname; date
module load StdEnv/2023 gcc cuda/12.2 opencv/4.11.0 python/3.11
module load arrow/24.0.0
source /scratch/reem1/rings_runs/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python /scratch/reem1/rings_runs/jobs/verify_rings_align.py
