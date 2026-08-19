#!/usr/bin/env bash
#SBATCH --job-name=which_lsb_sweep
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/reem1/rings_runs/logs/which_lsb_%A_%a.out

# which-lsb (Which Debris), MaxViT-Tiny: 2 divisors x 6 seeds = 12 tasks
# batch_size=32 per the reference command; if div4 breaks on batch counts
# (small dataset), rerun those tasks with batch_size=8.
DIVISORS=(2 4)
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
    +learner=maxvit_tiny_rw_224 +hardware=cluster \
    ++dataset=which-lsb \
    ++seed=${SEED} \
    ++run_id=which_lsb_maxvit_tiny_seed${SEED} \
    ++version=which_lsb/narval_repro_public_best_6seed_2026-06-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=8 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=${DIVISOR} \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.2 \
    ++learner.batch_size=32
