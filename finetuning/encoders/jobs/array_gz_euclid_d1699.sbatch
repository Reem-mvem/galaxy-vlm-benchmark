#!/usr/bin/env bash
#SBATCH --job-name=euclid_d1699
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/reem1/rings_runs/logs/euclid_d1699_%A_%a.out

# gz_euclid ~100-image point for the label-efficiency curve, MaxViT-Base.
# divisor 1699 -> 169898 // 1699 ~= 100 images. 6 seeds, same config as the existing curve.
SEEDS=(0 1 2 3 4 42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DIVISOR=1699
echo "DIVISOR=${DIVISOR} SEED=${SEED}"; pwd; hostname; date; nvidia-smi -L
module load StdEnv/2023 gcc cuda/12.2 opencv/4.11.0 python/3.11
module load arrow/24.0.0
source /scratch/reem1/rings_runs/.venv/bin/activate
export SCRATCH=/scratch/reem1
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
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
