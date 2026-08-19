#!/usr/bin/env bash
#SBATCH --job-name=cnn_rings_bal100
#SBATCH --account=def-bovy_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/reem1/rings_runs/logs/cnn_rings_bal100_%A_%a.out

SEEDS=(0 1); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "TASK=rings SEED=$SEED"; pwd; hostname; date; nvidia-smi -L
module load StdEnv/2023 gcc cuda/12.2 opencv/4.11.0 python/3.11
module load arrow/24.0.0
source /scratch/reem1/rings_runs/.venv/bin/activate
export SCRATCH=/scratch/reem1
export HF_HOME=/scratch/reem1/rings_runs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline

# same balanced 100 the VLM trained on (classification-config train_<i> ids)
export GZEVO_TRAIN_CSV=/home/reem1/scratch/galaxy-vlm-benchmark/tasks/rings/data/train_100.csv
REPO_DIR=/scratch/reem1/rings_runs/gz-evo

# pre-flight: confirm the 100 ids map to the SAME galaxies finetune.py will load, and that
# the classification named-config loads offline at all. Abort before training if not.
echo "=== pre-flight: verify balanced-100 id/label alignment ==="
python /scratch/reem1/rings_runs/jobs/verify_rings_align.py || { echo "alignment/offline-load check FAILED — aborting"; exit 1; }

# rings label-eff study used ConvNeXt-Base; classification head (num_classes=2) via gz-rings-binary.
# config matched to the other tasks' balanced-100 CNN runs (full finetune, 2 seeds, fp32).
python $REPO_DIR/gz_evo/downstream/finetune.py \
    +learner=convnext_base +hardware=cluster \
    ++dataset=gz-rings-binary ++seed=${SEED} \
    ++run_id=rings_convnext_bal100_seed${SEED} \
    ++version=rings/balanced100_2seed_2026-08-04 \
    ++hardware.devices=1 ++hardware.num_nodes=1 ++hardware.num_workers=6 ++hardware.strategy=auto \
    ++wandb=False ++debug=False \
    ++precision=32-true ++hardware.precision=32-true \
    ++divisor=1 ++learner.training_mode=full \
    ++learner.learning_rate=0.00001 ++learner.weight_decay=0.2 \
    ++learner.layer_decay=0.2 ++learner.batch_size=32
