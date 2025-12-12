#!/bin/bash
#SBATCH -J anixploreexp2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-19%4
#SBATCH -o /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.out
#SBATCH -e /gpfs/milgram/scratch60/gerstein/yz2483/logs/animel2m/%x_%A_%a.err

set -eo pipefail

module load miniconda
source activate /gpfs/milgram/pi/holmes/yz2483/conda_envs/animel2m

cd /gpfs/milgram/home/yz2483/animel2m
pwd


FAKE_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images"
REAL_ROOT="/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img"
SEG_PATH="./segformer_mit-b0.pth"
NUM_MODELS=4
NUM_FOLDS=5

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge $((NUM_MODELS * NUM_FOLDS)) ]; then
    echo "Error: TASK_ID $TASK_ID out of range (0-$((NUM_MODELS * NUM_FOLDS - 1)))"
    exit 1
fi

MODEL_IDX=$((TASK_ID / NUM_FOLDS))
FOLD=$((TASK_ID % NUM_FOLDS))       
echo "[Task $TASK_ID] MODEL_IDX=$MODEL_IDX, FOLD=$FOLD"

# MODEL_IDX: 0 -> AniXplore
# MODEL_IDX: 1 -> Baseline: ConvNeXt
# MODEL_IDX: 2 -> Baseline: ResNet
# MODEL_IDX: 3 -> Baseline: ViT

if [ $MODEL_IDX -eq 0 ]; then
    MODE="anixplore"
    MODEL_NAME="pgd" # placeholder, not used in AniXplore mode
    echo "[Task $TASK_ID] Running Model: AniXplore, Fold: $FOLD"

elif [ $MODEL_IDX -eq 1 ]; then
    MODE="baseline"
    MODEL_NAME="resnet"
    echo "[Task $TASK_ID] Running Baseline: ConvNeXt, Fold: $FOLD"

elif [ $MODEL_IDX -eq 2 ]; then
    MODE="baseline"
    MODEL_NAME="vit"
    echo "[Task $TASK_ID] Running Baseline: FrequencyAwareBaseline, Fold: $FOLD"

elif [ $MODEL_IDX -eq 3 ]; then
    MODE="baseline"
    MODEL_NAME="lightweight"
    echo "[Task $TASK_ID] Running Baseline: EfficientNetBaseline, Fold: $FOLD"

else
    echo "Error: MODEL_IDX $MODEL_IDX out of range (0-$((NUM_MODELS - 1)))"
    exit 1
fi

srun python learner.py \
  --mode "$MODE" \
  --model_name "$MODEL_NAME" \
  --fold "$FOLD" \
  --fake_root "$FAKE_ROOT" \
  --real_root "$REAL_ROOT" \
  --seg_path "$SEG_PATH"

echo "[Task $TASK_ID] Finished MODEL=$MODEL_NAME, FOLD=$FOLD."