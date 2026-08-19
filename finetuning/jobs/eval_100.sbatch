#!/usr/bin/env bash
#SBATCH --job-name=qwen_ev100
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/reem1/finetuning/logs/qwen_ev100_%x_%j.out

# usage: sbatch --export=ALL,TASK=is_lsb ...   TASK in {is_lsb,which_lsb,strong_lens}
: "${TASK:?set TASK=is_lsb|which_lsb|strong_lens}"
pwd; hostname; date; echo "TASK=$TASK (eval 100-sample adapter, on 3g.20gb MIG)"
module load StdEnv/2023 gcc python/3.11
module load arrow/24.0.0

echo "=== staging model ==="
SNAP=/scratch/reem1/rings_runs/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
mkdir -p "$SLURM_TMPDIR/qwen2vl"; time cp -rL "$SNAP"/. "$SLURM_TMPDIR/qwen2vl"/

source /scratch/reem1/finetuning/.venv/bin/activate
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_LOCAL_PATH="$SLURM_TMPDIR/qwen2vl"
export QWEN_ADAPTER_PATH="/scratch/reem1/finetuning/adapters/${TASK}_vision_100"

nvidia-smi -L
case "$TASK" in
  is_lsb)
    cd /home/reem1/scratch/galaxy-vlm-benchmark
    python -u finetuning/eval_qwen_local.py --task is_lsb \
      --output tasks/is_lsb/results/is_lsb_results_qwen_vision100.csv ;;
  which_lsb)
    cd /home/reem1/scratch/finetuning/which_lsb
    python -u eval_qwen_local.py --output results/which_lsb_results_qwen_vision100.csv ;;
  strong_lens)
    cd /home/reem1/scratch/finetuning/strong_lens
    python -u eval_qwen_local.py --output results/strong_lens_results_qwen_vision100.csv ;;
esac
echo "=== done ==="
