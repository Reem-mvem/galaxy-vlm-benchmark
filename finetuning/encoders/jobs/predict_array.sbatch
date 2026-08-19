#!/usr/bin/env bash
#SBATCH --job-name=rings_predict
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-29
#SBATCH --output=/scratch/reem1/rings_runs/logs/predict_%A_%a.out

MANIFEST=/scratch/reem1/rings_runs/jobs/predict_manifest.txt
# array index -> line (1-based)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
DIVISOR=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')
CKPT=$(echo "$LINE" | awk '{print $3}')

echo "DIVISOR=${DIVISOR}"
echo "SEED=${SEED}"
echo "CKPT=${CKPT}"
pwd; hostname; date

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
cd $REPO_DIR

python $REPO_DIR/gz_evo/downstream/predict_rings_simple.py \
    +learner=convnext_base +hardware=cluster \
    ++dataset=gz_rings \
    ++seed=${SEED} \
    ++version=narval_repro_public_best_6seed_2026-06-30 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=0 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=${DIVISOR} \
    ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 \
    ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.2 \
    ++learner.batch_size=32 \
    +ckpt_path=${CKPT}
