#!/usr/bin/env bash
#SBATCH --job-name=cnn_sl_bal100
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/reem1/rings_runs/logs/cnn_sl_bal100_%A_%a.out

# CNN (SoViT-400m) on the EXACT balanced-100 the VLM used (via GZEVO_TRAIN_CSV), 2 seeds.
SEEDS=(0 1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "SEED=$SEED"; pwd; hostname; date; nvidia-smi -L

module load StdEnv/2023 gcc cuda/12.2 opencv/4.11.0 python/3.11
module load arrow/24.0.0
source /scratch/reem1/rings_runs/.venv/bin/activate

export SCRATCH=/scratch/reem1
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
# train on the identical balanced 100 images the VLM saw
export GZEVO_TRAIN_CSV=/home/reem1/scratch/galaxy-vlm-benchmark/tasks/strong_lens/data/train_100.csv

REPO_DIR=/scratch/reem1/rings_runs/gz-evo
python $REPO_DIR/gz_evo/downstream/finetune.py \
    +learner=vit_so400m_siglip +hardware=cluster \
    ++dataset=euclid_strong_lens_expert_judges \
    ++seed=${SEED} \
    ++run_id=strong_lenses_sovit400_bal100_seed${SEED} \
    ++version=strong_lens/balanced100_2seed_2026-07-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=6 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=1 \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.8 \
    ++learner.batch_size=8
