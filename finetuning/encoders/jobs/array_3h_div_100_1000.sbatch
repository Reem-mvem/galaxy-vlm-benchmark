#!/usr/bin/env bash
#SBATCH --job-name=rings_div_100_1000
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/reem1/rings_runs/logs/array_3h_%A_%a.out

# 2 divisors x 6 seeds = 12 tasks (array indices 0-11)
DIVISORS=(100 1000)
SEEDS=(0 1 2 3 4 42)
NUM_SEEDS=6

DIVISOR_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
DIVISOR=${DIVISORS[$DIVISOR_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "DIVISOR=${DIVISOR}"
echo "SEED=${SEED}"
pwd; hostname; date
nvidia-smi

module load StdEnv/2023 gcc cuda/12.2 opencv/4.11.0 python/3.11
module load arrow/24.0.0
source /scratch/reem1/rings_runs/.venv/bin/activate

export SCRATCH=/scratch/reem1
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

REPO_DIR=/scratch/reem1/rings_runs/gz-evo

python $REPO_DIR/gz_evo/downstream/finetune.py \
    +learner=convnext_base +hardware=cluster \
    ++dataset=gz_rings \
    ++seed=${SEED} \
    ++run_id=gz_rings_convnext_base_seed${SEED} \
    ++version=narval_repro_public_best_6seed_2026-06-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=8 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=${DIVISOR} \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.2 \
    ++learner.batch_size=32
