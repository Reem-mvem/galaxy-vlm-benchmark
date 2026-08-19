#!/usr/bin/env bash
#SBATCH --job-name=strong_lens_sweep
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/reem1/rings_runs/logs/strong_lens_%A_%a.out

# strong lens (euclid_strong_lens_expert_judges), SoViT-400m/14: 3 divisors x 6 seeds = 18 tasks
# NOTE: bundle reference command uses batch_size=8 and layer_decay=0.8 for this model.
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
    +learner=vit_so400m_siglip +hardware=cluster \
    ++dataset=euclid_strong_lens_expert_judges \
    ++seed=${SEED} \
    ++run_id=strong_lenses_sovit400_seed${SEED} \
    ++version=strong_lens/narval_repro_public_best_6seed_2026-06-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=8 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=${DIVISOR} \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.8 \
    ++learner.batch_size=8
