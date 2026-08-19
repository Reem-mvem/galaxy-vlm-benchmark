#!/usr/bin/env bash
#SBATCH --job-name=gz_euclid_sweep
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/reem1/rings_runs/logs/gz_euclid_%A_%a.out

# gz_euclid (GZ Euclid), MaxViT-Base (best model, 2.1282): 3 divisors x 6 seeds = 18 tasks
# Bundle reference command uses batch_size=16, num_workers=4, prefetch_factor=2.
DIVISORS=(2 4 7)
SEEDS=(0 1 2 3 4 42)
NUM_SEEDS=6

DIVISOR=${DIVISORS[$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))]}

echo "DIVISOR=${DIVISOR}"
echo "SEED=${SEED}"
pwd; hostname; date
nvidia-smi

# arrow must be loaded LAST, in its own command, AFTER python/3.11 — otherwise the
# module silently fails to attach and pyarrow is missing (datasets load crashes).
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
    +learner=maxvit_base +hardware=cluster \
    ++dataset=gz_euclid \
    ++seed=${SEED} \
    ++run_id=gz_euclid_maxvit_base_seed${SEED} \
    ++version=gz_euclid/narval_repro_public_best_6seed_2026-06-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=4 ++hardware.prefetch_factor=2 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=${DIVISOR} \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.2 \
    ++learner.batch_size=16
